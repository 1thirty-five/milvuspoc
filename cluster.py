"""
Cluster the stored document vectors with KMeans and write each document's
cluster label back into Milvus.

Reads every vector from the live Milvus `documents` collection (so it clusters
exactly what is stored), runs KMeans (default k=10, configurable with --k),
prints each cluster with its member documents, then re-stores every row with a
`cluster` label so you can filter by it later, e.g.:

    client.query(collection_name="documents", filter="cluster == 3",
                 output_fields=["text", "cluster"])

`cluster` rides in the collection's dynamic field (the schema already has
enable_dynamic_field=True), so no schema change is needed and it stays
filterable like any scalar field. If result.py exists (from
`loadmilvus.py --full`), each entry there also gets a `cluster` label beside
its text.

Prerequisites:
    python loadmilvus.py     # embed + store the documents first
    pip install scikit-learn # clustering backend

Run:
    python cluster.py            # k = 10 (matches the 10 topics in input.md)
    python cluster.py --k 6      # choose a different number of clusters
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from loadmilvus import (DEFAULT_COLLECTION, DEFAULT_RESULT, INSERT_BATCH, connect,
                        ensure_collection, insert_batched)

# Default cluster count. input.md holds 10 topical groups of ~10 docs each, so
# k=10 is the natural choice; override per run with `--k <n>`.
DEFAULT_K = 10

# The catch-all bucket. Every clustering run emits it, whether or not anything
# lands in it, so downstream consumers can rely on it existing rather than
# discovering it the first time a corpus happens to produce outliers.
# -1 follows the DBSCAN/HDBSCAN convention for noise, and sorts naturally to the
# end; `cluster` stays an int, so existing filters and plots keep working.
UNASSIGNED = -1
UNASSIGNED_NAME = "unassigned"

# How far below the mean a row's similarity to its own centroid must fall before
# it is called uncorrelated, in standard deviations.
#
# This is relative, not an absolute cosine cutoff, and that is deliberate:
# measured on this corpus, bge-m3 puts *every* row between 0.643 and 0.876 of its
# centroid (mean 0.794, sd 0.055). An absolute floor of 0.45 -- which sounds
# permissive -- catches exactly zero rows. Modern embedding models compress
# cosine into a narrow high band, so a fixed threshold is either inert or
# catastrophic depending on the model, whereas "two sigma below this run's mean"
# means the same thing on any model and any corpus.
DEFAULT_FLOOR_SIGMA = 2.0


def unassigned_floor(similarity, sigma=DEFAULT_FLOOR_SIGMA):
    """Return the similarity below which a row counts as uncorrelated."""
    values = np.asarray(similarity, dtype="float64")
    if values.size == 0:
        return float("-inf")
    return float(values.mean() - sigma * values.std())


def apply_floor(labels, similarity, floor):
    """Send rows below `floor` to UNASSIGNED. Returns (labels, count moved)."""
    labels = np.asarray(labels).copy()
    if floor is None or not np.isfinite(floor):
        return labels, 0
    labels[np.asarray(similarity) < floor] = UNASSIGNED
    return labels, int((labels == UNASSIGNED).sum())


def centroid_similarity(embeddings, labels, k):
    """Cosine of every row to the centroid of the cluster it was assigned.

    Embeddings are L2-normalized on ingest, so a dot product is cosine. The
    centroid of a set of unit vectors is not itself unit length, hence the
    explicit renormalization.
    """
    matrix = np.asarray(embeddings, dtype="float32")
    similarity = np.zeros(len(matrix), dtype="float32")
    for c in range(k):
        members = labels == c
        if not members.any():
            continue
        centroid = matrix[members].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroid = centroid / (norm if norm else 1.0)
        similarity[members] = matrix[members] @ centroid
    return similarity


def fetch_all(client, collection=DEFAULT_COLLECTION, batch_size=INSERT_BATCH):
    """Return every stored row (text + embedding) from `collection`.

    Pages with query_iterator rather than a single query: a plain `query` is
    capped (16384 by default, and Milvus refuses more), so on a corpus larger
    than the cap it would silently return a truncated slice. store_labels
    rebuilds the collection from whatever comes back, so a short read here
    destroys every row it missed. Paging keeps that from happening.

    The collection must be loaded before it can be queried; `filter="id >= 0"`
    matches all auto-id rows. Returns the raw list of row dicts from Milvus.
    """
    if not client.has_collection(collection):
        raise SystemExit(
            f"Collection '{collection}' does not exist. Run loadmilvus.py first.")
    client.load_collection(collection)

    iterator = client.query_iterator(
        collection_name=collection,
        filter="id >= 0",
        # Every field, not just the ones KMeans needs: store_labels rebuilds the
        # collection from these rows, so anything missing here is destroyed.
        output_fields=["text", "embedding", "source", "chunk_hash", "seq"],
        batch_size=batch_size,
    )
    rows = []
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows.extend(batch)
    finally:
        iterator.close()

    expected = client.get_collection_stats(collection)["row_count"]
    if len(rows) != expected:
        raise SystemExit(
            f"Read {len(rows)} rows but '{collection}' holds {expected}. "
            f"Refusing to continue: store_labels rebuilds the collection from "
            f"these rows, so clustering a partial read would destroy the rest.")

    print(f"Fetched {len(rows)} vectors from Milvus collection '{collection}'.")
    return rows


def cluster_vectors(embeddings, k=DEFAULT_K, sigma=DEFAULT_FLOOR_SIGMA):
    """Run KMeans over `embeddings` and return an integer label per vector.

    Vectors are L2-normalized (COSINE space), so Euclidean KMeans approximates
    cosine clustering well. n_init/random_state are fixed for reproducible runs.

    Rows that fit their assigned centroid poorly are moved to UNASSIGNED rather
    than left inside a cluster they only nominally belong to. Pass `sigma=0` to
    disable that and keep plain KMeans behaviour.
    """
    matrix = np.asarray(embeddings, dtype="float32")
    n = len(matrix)
    if k > n:
        raise SystemExit(f"Cannot make {k} clusters from {n} vectors (k > n).")
    print(f"Clustering {n} vectors into k={k} groups with KMeans...")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(matrix)

    # KMeans assigns every point to something -- it has no concept of "this one
    # belongs nowhere". So the outliers are found afterwards, by how well each
    # row actually fits the centroid it was handed, and moved to UNASSIGNED.
    if sigma <= 0:
        # Not "a floor at the mean" -- that would exile every below-average row,
        # roughly half the corpus. sigma<=0 means the check is off.
        print(f"  outlier floor disabled (sigma={sigma}); "
              f"{UNASSIGNED_NAME} left empty.")
        return labels

    similarity = centroid_similarity(matrix, labels, k)
    floor = unassigned_floor(similarity, sigma)
    labels, dropped = apply_floor(labels, similarity, floor)
    print(f"  floor={floor:.3f} (mean-{sigma}sd of centroid similarity): "
          f"{dropped} row(s) -> {UNASSIGNED_NAME}.")
    return labels


def store_labels(client, rows, labels, collection=DEFAULT_COLLECTION,
                 names=None, scheme=None):
    """Rebuild `collection` with each row's `cluster` label attached.

    The collection is reset and every row re-inserted with a `cluster` field
    (absorbed by the dynamic field). Reusing the stored embeddings means nothing
    is re-embedded. Returns the row count.

    Every field the row arrived with is carried back through the rebuild, not
    just the ones clustering cares about: this drops and recreates the
    collection, so any field left out here is silently destroyed for good.
    `source` is what search results are attributed to, and `chunk_hash` is the
    embedding cache's key -- losing it would force a full re-embed next ingest.

    `names` and `scheme` are what customcluster.py adds on top: a per-row human
    name for the cluster (`cluster_name`) and the criterion that produced it
    (`cluster_scheme`). `cluster` stays an int either way, so everything that
    already filters or plots on it -- search.py, visualize.py -- keeps working.
    """
    dim = len(rows[0]["embedding"])
    ensure_collection(client, dim=dim, collection=collection, reset=True)
    data = []
    for i, (row, label) in enumerate(zip(rows, labels)):
        entry = {
            "text": row["text"],
            "embedding": row["embedding"],
            "chunk_hash": row["chunk_hash"],
            "cluster": int(label),
        }
        if names is not None:
            entry["cluster_name"] = str(names[i])
        if scheme is not None:
            entry["cluster_scheme"] = str(scheme)
        # `source` and `seq` are dynamic and only set by the extractpdf path, so
        # they may be absent on rows that came from loadmilvus. Don't insert
        # nulls for them. `seq` is the chunk's position in its file, and it is
        # what visualize.py sorts on -- dropping it here would silently return
        # the plot to arbitrary segment order on the next clustering run.
        if row.get("source") is not None:
            entry["source"] = row["source"]
        if row.get("seq") is not None:
            entry["seq"] = row["seq"]
        data.append(entry)
    count = insert_batched(client, data, collection)   # also flushes
    print(f"Stored cluster labels for {count} rows in '{collection}'.")
    return count


def write_result_labels(rows, labels, path=DEFAULT_RESULT):
    """Add each document's cluster label beside its text in result.py.

    result.py (generated by `loadmilvus.py --full`) holds a `results` list of
    {"text", "vector"} dicts. This rewrites it in place so every entry gains a
    "cluster" key right after "text", matched to the KMeans labels by text.
    File order and the stored vectors are preserved. Skips gracefully if the
    file does not exist yet.
    """
    result_path = Path(path)
    if not result_path.exists():
        print(f"Skipping {path}: not found "
              f"(run `python loadmilvus.py --full` to generate it).")
        return

    # The file is plain data; exec it into a namespace rather than importing so
    # the `path` argument is honored and there's no module-cache surprise.
    namespace = {}
    exec(result_path.read_text(encoding="utf-8"), namespace)
    entries = namespace.get("results", [])
    dim = namespace.get("dim")

    label_by_text = {row["text"]: int(label) for row, label in zip(rows, labels)}

    missing = 0
    with open(result_path, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated by `loadmilvus.py --full`; cluster labels added by\n')
        f.write('`cluster.py`. Each entry is a sentence, its KMeans cluster id, and the\n')
        f.write('full embedding vector the model produced for it."""\n\n')
        if dim is not None:
            f.write(f"dim = {dim}\n\n")
        f.write("results = [\n")
        for entry in entries:
            label = label_by_text.get(entry["text"])
            if label is None:
                missing += 1
            f.write("    {\n")
            f.write(f"        {'text'!r}: {entry['text']!r},\n")
            f.write(f"        {'cluster'!r}: {label!r},\n")
            f.write(f"        {'vector'!r}: {entry['vector']!r},\n")
            f.write("    },\n")
        f.write("]\n")

    note = f" ({missing} had no matching label)" if missing else ""
    print(f"Updated {path} with cluster labels for {len(entries)} entries{note}.")


def print_clusters(rows, labels, k, limit=10):
    """Print each cluster id and the documents assigned to it.

    Only the first `limit` members of each cluster are printed. On a book-sized
    corpus the clusters run to thousands of chunks each, and printing them all
    just dumps the whole corpus to the console.
    """
    # UNASSIGNED is seeded here, not discovered from the data, so the bucket is
    # reported on every run even when empty. An absent category reads as "no
    # outlier detection happened"; an empty one states positively that the check
    # ran and everything cleared it.
    groups = {c: [] for c in range(k)}
    groups[UNASSIGNED] = []
    for row, label in zip(rows, labels):
        groups.setdefault(int(label), []).append(row["text"])

    print(f"\nClusters (k={k}):")
    for c in list(range(k)) + [UNASSIGNED]:
        members = groups[c]
        title = (f"Cluster {UNASSIGNED} ({UNASSIGNED_NAME})"
                 if c == UNASSIGNED else f"Cluster {c}")
        print(f"\n{title}  ({len(members)} docs)")
        for text in members[:limit]:
            print(f"  - {text}")
        if len(members) > limit:
            print(f"  ... and {len(members) - limit} more")


def parse_k_arg(argv, default=DEFAULT_K):
    """Read the cluster count from `--k <n>` in argv, else `default`."""
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 >= len(argv):
            raise SystemExit("--k needs a value, e.g. --k 6")
        try:
            return int(argv[i + 1])
        except ValueError:
            raise SystemExit(f"--k must be an integer, got {argv[i + 1]!r}")
    return default


def main():
    """Fetch vectors from Milvus, cluster them, print groups, store labels back."""
    # Cluster members are extracted document text, which can hold characters
    # outside the terminal's legacy code page (e.g. math glyphs like the sqrt
    # sign in "sqrt(dk)"). Don't let print_clusters crash the run on a Windows
    # cp1252 console; replace anything unencodable. (Same guard as extractpdf.py.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    k = parse_k_arg(sys.argv)

    client = connect()
    rows = fetch_all(client)
    if not rows:
        raise SystemExit(
            "No vectors found in Milvus. Run loadmilvus.py to store some first.")

    embeddings = [row["embedding"] for row in rows]
    labels = cluster_vectors(embeddings, k)

    print_clusters(rows, labels, k)
    store_labels(client, rows, labels)
    write_result_labels(rows, labels)

    print(f"\nDone. Cluster labels (k={k}) stored in Milvus collection "
          f"'{DEFAULT_COLLECTION}'.")
    print('Filter later with, e.g., '
          'client.query(filter="cluster == 0", output_fields=["text", "cluster"]).')


if __name__ == "__main__":
    main()
