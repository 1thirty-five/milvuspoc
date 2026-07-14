"""
Sweep the cluster count k and report how well the corpus actually clusters.

cluster.py defaults to k=10 because input.md held 10 hand-written topical
groups. On a corpus of PDF chunks that number is arbitrary: KMeans returns
exactly k clusters whether or not k clusters exist in the data, so a plot of
overlapping blobs may just mean "there is no 10-way split here."

This script answers the prior question -- is there any k worth using? -- without
touching what is stored. It reads the vectors, fits KMeans for a range of k, and
scores each fit:

    silhouette   mean over points of (b - a) / max(a, b), cosine distance, where
                 a = distance to own cluster, b = distance to nearest other one.
                 ~0.5+ is strong structure, ~0.25-0.5 is real but soft, and
                 anything near 0 means the clusters are arbitrary cuts through a
                 continuum -- which is what "the clusters look smeared together"
                 usually is. Higher is better; it needs no ground truth.
    inertia      KMeans' own objective (sum of squared distances to the assigned
                 centroid). Always falls as k rises, so it can't pick k on its
                 own -- look for the elbow where it stops falling steeply.

Read-only: it never writes labels back to Milvus. Run `python cluster.py --k <n>`
with the k you pick to actually apply it.

Prerequisites:
    python loadmilvus.py     # embed + store the documents first
    pip install scikit-learn

Run:
    python sweep.py                          # k = 2..20, on a 8000-vector sample
    python sweep.py --min-k 2 --max-k 40
    python sweep.py --sample 0               # use every vector (slow)
"""

import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from loadmilvus import DEFAULT_COLLECTION, connect

# Rows per query_iterator page. Only bounds memory per round-trip, not the total.
FETCH_BATCH = 1000

# KMeans on 30k x 1024 float32, refit for every k, is minutes of CPU per k. The
# shape of the silhouette curve is a property of the corpus, not of how many
# vectors you show it, so fit on a random sample and the curve lands in the same
# place for a fraction of the cost. --sample 0 opts into the full set.
DEFAULT_SAMPLE = 8000

# Silhouette is O(n^2) in distance computations, so it gets a smaller sample
# still -- scored on the same fitted labels, just fewer pairs.
SILHOUETTE_SAMPLE = 3000

DEFAULT_MIN_K = 2
DEFAULT_MAX_K = 20

# The floor below which a silhouette score means "no real structure at this k".
WEAK_SILHOUETTE = 0.10


def fetch_embeddings(client, collection=DEFAULT_COLLECTION, batch_size=FETCH_BATCH):
    """Return every stored embedding from `collection` as a float32 matrix.

    Pages with query_iterator because a plain `query` is capped at 16384 rows and
    would silently hand back a truncated slice of a book-sized corpus -- a sweep
    over half the data would score a corpus that isn't the one you have. Only the
    embeddings are read; this script never writes, so no other field is needed.
    """
    if not client.has_collection(collection):
        raise SystemExit(
            f"Collection '{collection}' does not exist. Run loadmilvus.py first.")
    client.load_collection(collection)

    iterator = client.query_iterator(
        collection_name=collection,
        filter="id >= 0",
        output_fields=["embedding"],
        batch_size=batch_size,
    )
    vectors = []
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            vectors.extend(row["embedding"] for row in batch)
    finally:
        iterator.close()

    expected = client.get_collection_stats(collection)["row_count"]
    if len(vectors) != expected:
        print(f"Warning: read {len(vectors)} rows but '{collection}' holds "
              f"{expected}. Sweeping the rows that came back.")

    print(f"Fetched {len(vectors)} vectors from Milvus collection '{collection}'.")
    return np.asarray(vectors, dtype="float32")


def subsample(matrix, size, seed=42):
    """Return `size` rows drawn at random from `matrix` (all of it if size >= n)."""
    n = len(matrix)
    if size <= 0 or size >= n:
        return matrix
    rng = np.random.default_rng(seed)
    return matrix[rng.choice(n, size=size, replace=False)]


def score_k(matrix, k, silhouette_sample=SILHOUETTE_SAMPLE, seed=42):
    """Fit KMeans with `k` clusters and return (silhouette, inertia, smallest cluster).

    Vectors are L2-normalized (loadmilvus embeds with normalize_embeddings=True),
    so Euclidean KMeans approximates cosine clustering -- but the silhouette is
    scored with the cosine metric to match the space the vectors were embedded
    and indexed in. n_init is 3 rather than cluster.py's 10: this is a survey of
    many k, not the final fit, and the curve doesn't move.

    The smallest cluster size is worth watching: KMeans hitting a k with no
    structure often parks a handful of outliers in a near-empty cluster and
    splits one real blob in half, which the silhouette alone won't show you.
    """
    km = KMeans(n_clusters=k, n_init=3, random_state=seed)
    labels = km.fit_predict(matrix)

    scored = subsample(matrix, silhouette_sample, seed=seed)
    if len(scored) < len(matrix):
        # Score a sample of points, but against their real labels: refitting on
        # the sample would measure a different clustering than the one reported.
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(matrix), size=silhouette_sample, replace=False)
        scored, scored_labels = matrix[idx], labels[idx]
    else:
        scored_labels = labels

    # A sample can happen to miss a tiny cluster entirely; silhouette needs >= 2
    # distinct labels present.
    if len(set(scored_labels)) < 2:
        return float("nan"), float(km.inertia_), int(np.bincount(labels).min())

    silhouette = silhouette_score(scored, scored_labels, metric="cosine")
    smallest = int(np.bincount(labels, minlength=k).min())
    return float(silhouette), float(km.inertia_), smallest


def sweep(matrix, min_k=DEFAULT_MIN_K, max_k=DEFAULT_MAX_K, step=1):
    """Score every k in [min_k, max_k] and return the rows, printing as it goes."""
    n = len(matrix)
    if max_k >= n:
        raise SystemExit(f"Cannot make {max_k} clusters from {n} vectors (k >= n).")

    print(f"\nSweeping k = {min_k}..{max_k} over {n} vectors "
          f"({matrix.shape[1]}-dim). This refits KMeans per k; give it a minute.\n")
    print(f"{'k':>4}  {'silhouette':>10}  {'inertia':>12}  {'smallest':>8}")
    print(f"{'-' * 4}  {'-' * 10}  {'-' * 12}  {'-' * 8}")

    rows = []
    for k in range(min_k, max_k + 1, step):
        silhouette, inertia, smallest = score_k(matrix, k)
        rows.append((k, silhouette, inertia, smallest))
        print(f"{k:>4}  {silhouette:>10.4f}  {inertia:>12.1f}  {smallest:>8}")
    return rows


def report(rows):
    """Print the best k by silhouette and say what the number actually means."""
    scored = [r for r in rows if not np.isnan(r[1])]
    if not scored:
        raise SystemExit("No k could be scored. Is the corpus large enough?")

    best_k, best_sil, _, best_smallest = max(scored, key=lambda r: r[1])
    print(f"\nBest k by silhouette: k={best_k} (silhouette {best_sil:.4f}, "
          f"smallest cluster {best_smallest} docs).")

    if best_sil < WEAK_SILHOUETTE:
        print(
            f"\nThat is a weak score (< {WEAK_SILHOUETTE}). The corpus has no crisp\n"
            f"k-way split: KMeans is cutting a continuous cloud, so every k is about\n"
            f"as arbitrary as any other and the plot will look smeared no matter what\n"
            f"you pick. Density-based clustering (HDBSCAN over a UMAP-reduced space)\n"
            f"can label the in-between chunks as noise instead of forcing them into a\n"
            f"cluster, which is usually the honest answer for chunked-PDF corpora.")
    else:
        print(f"\nApply it with:  python cluster.py --k {best_k}")

    print("\nAlso scan the inertia column for an elbow -- the k where it stops")
    print("dropping steeply is a second, independent vote for the same choice.")


def parse_int_arg(argv, flag, default):
    """Read an integer from `--flag <n>` in argv, else `default`."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value, e.g. {flag} {default}")
        try:
            return int(argv[i + 1])
        except ValueError:
            raise SystemExit(f"{flag} must be an integer, got {argv[i + 1]!r}")
    return default


def main():
    """Fetch the stored vectors, sweep k, and report which k the data supports."""
    min_k = parse_int_arg(sys.argv, "--min-k", DEFAULT_MIN_K)
    max_k = parse_int_arg(sys.argv, "--max-k", DEFAULT_MAX_K)
    sample = parse_int_arg(sys.argv, "--sample", DEFAULT_SAMPLE)
    if min_k < 2:
        raise SystemExit("--min-k must be at least 2 (silhouette needs 2 clusters).")
    if max_k < min_k:
        raise SystemExit(f"--max-k ({max_k}) must be >= --min-k ({min_k}).")

    client = connect()
    matrix = fetch_embeddings(client)
    if not len(matrix):
        raise SystemExit(
            "No vectors found in Milvus. Run loadmilvus.py to store some first.")

    fitted = subsample(matrix, sample)
    if len(fitted) < len(matrix):
        print(f"Sampling {len(fitted)} of {len(matrix)} vectors to fit "
              f"(pass --sample 0 to use all of them).")

    rows = sweep(fitted, min_k, max_k)
    report(rows)


if __name__ == "__main__":
    main()
