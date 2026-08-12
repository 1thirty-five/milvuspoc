"""
Cluster the stored document vectors by a criterion *you* define in words.

Plain `cluster.py` is criterion-free: KMeans on the raw embeddings groups by
whatever bge-m3 thinks dominates, and the only knob is k. This script reads a
criterion out of `criteria.md` and clusters on that instead. Nothing else in the
pipeline changes -- it reuses cluster.py's fetch/store/result-file plumbing, and
still writes an integer `cluster` field, so search.py and visualize.py keep
working untouched.

The whole idea in one line: **a criterion stated in words embeds into the same
space as the documents**, so it becomes geometry with no training and no LLM.

One mode, declared by the `# Mode` heading in criteria.md:

  anchor  You name the buckets. Each label (plus its description) is embedded,
          and every chunk is assigned to the label nearest it. `assign = hard`
          takes the nearest anchor and stops; `assign = seeded` (default) uses
          the anchors as KMeans init, so clusters settle onto the data's real
          density while staying named and ordered by your vocabulary.

Rows that fit nothing well go to the `unassigned` bucket (`cluster = -1`) rather
than being forced into the least-bad label. See `floor` in criteria.md.

An `axes` mode also existed, in which you named a contrast ("formal prose vs
casual explanation"), each axis became a direction `normalize(embed(left) -
embed(right))`, and KMeans ran on those projections alone. It was removed on
2026-08-12 by request. Worth knowing if it is ever reconsidered: it was the only
mode that could express a criterion *orthogonal to topic*. In raw embedding space
topic dominates, so nearest-anchor sends "how urgent is this" to whichever label
shares its subject matter and returns something plausible and wrong. Anchor mode
cannot do cross-topic criteria; it is good at topical ones.

Labels are stored three ways per row: `cluster` (int, unchanged semantics),
`cluster_name` (the human name -- "risk factors"), and `cluster_scheme` (which
criterion produced it). Filter later with, e.g.:

    client.query(collection_name="documents", filter='cluster_name == "results"',
                 output_fields=["text", "cluster_name"])

Prerequisites:
    python loadmilvus.py     # embed + store the documents first
    pip install scikit-learn

Run:
    python customcluster.py                     # read criteria.md
    python customcluster.py --criteria mine.md  # a different criterion file
    python customcluster.py --dry-run           # print groups, write nothing

Note it shares one limitation with cluster.py: `store_labels` rebuilds the
collection, so only one clustering lives in Milvus at a time and each run
replaces the last. `cluster_scheme` tells you which one is currently stored.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from cluster import (apply_floor, fetch_all, store_labels, unassigned_floor,
                     write_result_labels)
from loadmilvus import (DEFAULT_COLLECTION, DEFAULT_MODEL, connect, embed,
                        get_dim, get_model)

# Criterion file read when --criteria isn't given.
DEFAULT_CRITERIA = "criteria.md"

# Settings defaults; every one is overridable from the file's `# Settings`
# section.
DEFAULT_ASSIGN = "seeded"

# `auto` = two sigma below this run's mean similarity; see cluster.unassigned_floor
# for why a relative rule and not an absolute cosine cutoff. A number in
# criteria.md still works and is taken as an absolute floor; `0` disables the
# bucket's population (it is still reported, empty).
DEFAULT_FLOOR = "auto"
DEFAULT_SCHEME = "default"
DEFAULT_PREVIEW = 10

# Rows that fall below `floor` land here rather than being forced into the
# least-bad bucket. -1 keeps `cluster` an int, so visualize.py still sorts it.
UNASSIGNED = -1
UNASSIGNED_NAME = "unassigned"

MODES = ("anchor",)


# --------------------------------------------------------------------------
# criteria.md
# --------------------------------------------------------------------------

def strip_bullet(line):
    """Strip a leading `- `, `* `, `+ `, or `1. ` list marker from a line."""
    for marker in ("- ", "* ", "+ "):
        if line.startswith(marker):
            return line[len(marker):].strip()
    head, sep, rest = line.partition(". ")
    if sep and head.isdigit():
        return rest.strip()
    return line


def parse_criteria(path=DEFAULT_CRITERIA):
    """Parse a criteria.md file into a config dict.

    Format -- markdown headings, same shape as input.md, so the file stays
    readable as a document rather than turning into YAML:

        # Mode
        anchor

        # Labels
        - pricing: fees, discounts, billing terms
        - legal risk: liability, indemnity, compliance exposure

        # Settings
        assign = seeded
        floor = 0.35

    <!-- HTML comments --> are ignored anywhere in the file, which is what lets
    criteria.md document itself inline. Returns a dict with keys: mode, labels
    (list of (name, description)), assign, floor, model, scheme, preview.
    """
    criteria_path = Path(path)
    if not criteria_path.exists():
        raise SystemExit(
            f"Criterion file '{path}' not found. Create it (see criteria.md in "
            f"the repo for the format) or pass --criteria <path>.")

    mode, labels, settings = None, [], {}
    section, in_comment = None, False

    for raw in criteria_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if line.startswith("<!--"):
            if "-->" not in line:
                in_comment = True
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
            if heading.startswith("mode"):
                section = "mode"
            elif heading.startswith("label"):
                section = "labels"
            elif heading.startswith("setting"):
                section = "settings"
            else:
                section = None      # unknown heading: ignore its body
            continue

        line = strip_bullet(line)

        if section == "mode":
            if mode is None:
                mode = line.lower()
        elif section == "labels":
            name, _, description = line.partition(":")
            name = name.strip()
            if name:
                labels.append((name, description.strip()))
        elif section == "settings":
            key, sep, value = line.partition("=")
            if not sep:
                raise SystemExit(
                    f"Setting must be `key = value`, got {line!r} in {path}.")
            settings[key.strip().lower()] = value.strip()

    return validate(mode, labels, settings, path)


def setting_number(settings, key, default, cast):
    """Read one numeric setting, failing loudly rather than silently defaulting."""
    if key not in settings:
        return default
    try:
        return cast(settings[key])
    except ValueError:
        raise SystemExit(
            f"Setting `{key}` must be a {cast.__name__}, got {settings[key]!r}.")


def validate(mode, labels, settings, path):
    """Check the parsed criterion is usable and return it as a config dict."""
    if mode is None:
        raise SystemExit(
            f"No mode in {path}. Add a `# Mode` heading with one of: "
            f"{', '.join(MODES)}.")
    if mode not in MODES:
        raise SystemExit(
            f"Unknown mode {mode!r} in {path}. Use one of: {', '.join(MODES)}.")
    if mode == "anchor" and len(labels) < 2:
        raise SystemExit(
            f"Mode is `anchor` but {path} has {len(labels)} label(s) under "
            f"`# Labels`. Give it at least 2 -- one bucket isn't a clustering.")

    assign = settings.get("assign", DEFAULT_ASSIGN).lower()
    if assign not in ("seeded", "hard"):
        raise SystemExit(
            f"Setting `assign` must be `seeded` or `hard`, got {assign!r}.")

    # `floor` accepts the word `auto` as well as a number, so it cannot go
    # through setting_number's unconditional cast.
    floor = settings.get("floor", DEFAULT_FLOOR)
    if isinstance(floor, str) and floor.strip().lower() == "auto":
        floor = "auto"
    else:
        try:
            floor = float(floor)
        except (TypeError, ValueError):
            raise SystemExit(
                f"Setting `floor` must be a number or `auto`, got {floor!r}.")
        if not -1.0 <= floor <= 1.0:
            raise SystemExit(
                f"Setting `floor` is a cosine similarity, so it must be between "
                f"-1 and 1; got {floor}.")

    return {
        "mode": mode,
        "labels": labels,
        "k": len(labels),
        "assign": assign,
        "floor": floor,
        "model": settings.get("model", DEFAULT_MODEL),
        "scheme": settings.get("scheme", DEFAULT_SCHEME),
        "preview": setting_number(settings, "preview", DEFAULT_PREVIEW, int),
    }


# --------------------------------------------------------------------------
# anchor mode -- the user names the buckets
# --------------------------------------------------------------------------

def anchor_text(name, description):
    """The string actually embedded for one label.

    Name *and* description, because the description is what carries the signal:
    "Tier 1" embeds to noise, "Tier 1: capital adequacy requirements under Basel
    III" embeds to something a corpus can match.
    """
    return f"{name}: {description}" if description else name


def embed_anchors(model, labels):
    """Embed each label into the document space. Returns a (n_labels, dim) array."""
    texts = [anchor_text(name, description) for name, description in labels]
    print(f"Embedding {len(texts)} anchor label(s)...")
    return np.asarray(embed(model, texts), dtype="float32")


def resolve_floor(similarity, floor=DEFAULT_FLOOR):
    """Turn the configured `floor` into a concrete similarity cutoff.

    `auto` (the default) reads the cutoff off this run's own distribution --
    two sigma below the mean. An explicit number is used as-is. `0` or negative
    means never move anything, which still leaves the bucket reported but empty.
    """
    if isinstance(floor, str):
        if floor.strip().lower() == "auto":
            return unassigned_floor(similarity)
        floor = float(floor)
    return float(floor) if floor > 0 else None


def apply_unassigned(labels, similarity, floor=DEFAULT_FLOOR):
    """Move poorly-fitting rows to UNASSIGNED and report what moved."""
    cutoff = resolve_floor(similarity, floor)
    labels, dropped = apply_floor(labels, similarity, cutoff)
    if cutoff is None:
        print(f"  floor disabled -> {UNASSIGNED_NAME} left empty.")
    else:
        print(f"  floor={cutoff:.3f} -> {dropped} row(s) to {UNASSIGNED_NAME}.")
    return labels


def assign_anchors(embeddings, anchors, assign=DEFAULT_ASSIGN, floor=DEFAULT_FLOOR):
    """Assign every row to an anchor. Returns (labels, similarity per row).

    `hard` is pure nearest-anchor: every chunk goes to the label it is closest
    to and the buckets mean exactly what the user wrote. `seeded` hands the
    anchors to KMeans as `init` with `n_init=1`, so the centroids start on the
    user's vocabulary and then move to where the data actually is -- it surfaces
    structure the user didn't anticipate while keeping their names. Passing an
    init array is what preserves the label order: centroid i stays the one that
    started on anchor i.

    Both vectors are L2-normalized, so a dot product *is* cosine similarity.
    The returned similarity is each row's cosine to whatever it was assigned to
    (its anchor, or its fitted centroid), which is what `floor` is applied to
    and what the per-cluster report prints.
    """
    matrix = np.asarray(embeddings, dtype="float32")
    anchors = np.asarray(anchors, dtype="float32")

    if assign == "hard":
        print(f"Assigning {len(matrix)} vectors to nearest of "
              f"{len(anchors)} anchors...")
        similarities = matrix @ anchors.T
        labels = similarities.argmax(axis=1)
        best = similarities[np.arange(len(matrix)), labels]
    else:
        print(f"Running KMeans over {len(matrix)} vectors, seeded from "
              f"{len(anchors)} anchors...")
        km = KMeans(n_clusters=len(anchors), init=anchors, n_init=1,
                    random_state=42)
        labels = km.fit_predict(matrix)
        # Cosine to the *fitted* centroid, not the original anchor: after the
        # centroids move, distance to the anchor no longer describes the fit.
        centroids = km.cluster_centers_
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        best = np.einsum("ij,ij->i", matrix, (centroids / norms)[labels])

    labels = apply_unassigned(labels, best, floor)
    return labels, best


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_clusters(rows, labels, names, preview=DEFAULT_PREVIEW, stats=None):
    """Print each cluster by name with its member documents.

    Only the first `preview` members of each cluster are shown; on a book-sized
    corpus the clusters run to thousands of chunks and printing them all just
    dumps the corpus to the console. `stats` is an optional {cluster_id: str}
    of extra per-cluster detail (mean/min cosine per cluster).
    """
    # UNASSIGNED is seeded rather than discovered, so the bucket is reported on
    # every run even when nothing fell into it. An absent category reads as "no
    # outlier check ran"; an empty one says the check ran and everything passed.
    groups = {UNASSIGNED: []}
    for row, label in zip(rows, labels):
        groups.setdefault(int(label), []).append(row["text"])

    # Unassigned last: it's a leftovers bucket, not cluster "-1".
    order = sorted(groups, key=lambda c: (c == UNASSIGNED, c))

    populated = sum(1 for c in groups if groups[c])
    print(f"\nClusters ({populated} non-empty):")
    for c in order:
        members = groups[c]
        name = UNASSIGNED_NAME if c == UNASSIGNED else names[c]
        detail = f"  {stats[c]}" if stats and c in stats else ""
        print(f"\n[{c}] {name}  ({len(members)} docs){detail}")
        for text in members[:preview]:
            print(f"  - {text}")
        if len(members) > preview:
            print(f"  ... and {len(members) - preview} more")


def anchor_stats(labels, similarity, k):
    """Mean/min cosine per cluster -- the numbers you set `floor` from."""
    stats = {}
    for c in list(range(k)) + [UNASSIGNED]:
        selected = similarity[labels == c]
        if len(selected):
            stats[c] = (f"cosine mean {selected.mean():.3f}, "
                        f"min {selected.min():.3f}")
    return stats


def arg_value(argv, flag, default=None):
    """Read `--flag <value>` out of argv, else `default`."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value.")
        return argv[i + 1]
    return default


def describe(config, path):
    """Echo the criterion back before doing any work, so a typo is visible."""
    print(f"\nCriterion from {path}:")
    print(f"  mode    : {config['mode']}")
    print(f"  assign  : {config['assign']}")
    print(f"  labels  : {len(config['labels'])}")
    for name, description in config["labels"]:
        print(f"    - {anchor_text(name, description)}")
    print(f"  floor   : {config['floor']}")
    print(f"  model   : {config['model']}")
    print(f"  scheme  : {config['scheme']}\n")


def main():
    """Read the criterion, classify every stored vector by it, store the labels."""
    # Cluster members are extracted document text, which can hold characters
    # outside the terminal's legacy code page. Don't let printing crash the run
    # on a Windows cp1252 console. (Same guard as cluster.py / extractpdf.py.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    argv = sys.argv
    criteria_path = arg_value(argv, "--criteria", DEFAULT_CRITERIA)
    dry_run = "--dry-run" in argv

    config = parse_criteria(criteria_path)

    # --k is kept only to fail loudly. With axes mode gone the cluster count is
    # always the number of labels, so silently ignoring a --k someone typed out
    # of habit would run a different clustering than they asked for.
    if "--k" in argv:
        raise SystemExit(
            "--k is not accepted: the cluster count is the number of labels in "
            "your `# Labels` section. Add or remove a label instead.")

    describe(config, criteria_path)

    client = connect()
    rows = fetch_all(client)
    if not rows:
        raise SystemExit(
            "No vectors found in Milvus. Run loadmilvus.py to store some first.")
    embeddings = [row["embedding"] for row in rows]

    if config["k"] > len(rows):
        raise SystemExit(
            f"Cannot make {config['k']} clusters from {len(rows)} vectors.")

    model = get_model(config["model"])
    # A criterion embedded by a different model lands in a different space, and
    # the assignment is then meaningless rather than merely wrong. Dimension is
    # the cheap half of that check; it catches the realistic mistake (running
    # this with the default after ingesting under `--model minilm`).
    stored_dim, model_dim = len(embeddings[0]), get_dim(model)
    if stored_dim != model_dim:
        raise SystemExit(
            f"Model '{config['model']}' embeds to {model_dim} dims but the "
            f"stored vectors are {stored_dim}. Set `model = <the model you "
            f"ingested with>` in {criteria_path}.")

    anchors = embed_anchors(model, config["labels"])
    labels, similarity = assign_anchors(
        embeddings, anchors, config["assign"], config["floor"])
    names = [name for name, _ in config["labels"]]
    stats = anchor_stats(labels, similarity, config["k"])

    print_clusters(rows, labels, names, config["preview"], stats)

    if dry_run:
        print("\n--dry-run: nothing written to Milvus or result.py.")
        return

    row_names = [UNASSIGNED_NAME if int(label) == UNASSIGNED else names[int(label)]
                 for label in labels]
    store_labels(client, rows, labels, names=row_names, scheme=config["scheme"])
    write_result_labels(rows, labels)

    print(f"\nDone. Scheme '{config['scheme']}' ({config['mode']} mode) stored "
          f"in Milvus collection '{DEFAULT_COLLECTION}'.")
    print('Filter later with, e.g., client.query('
          f'filter=\'cluster_name == "{names[0]}"\', '
          'output_fields=["text", "cluster_name"]).')


if __name__ == "__main__":
    main()
