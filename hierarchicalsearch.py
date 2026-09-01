"""
Hierarchical search -- rank the *clusters* first, then search inside the winners.

Every technique in search.py searches the whole pot: one flat ANN sweep over all
stored vectors, and whatever comes back top-k comes back. This one searches in
two stages instead:

    stage 1   score each cluster against the query (cosine of the query to the
              cluster's centroid) and rank the clusters.
    stage 2   run a normal dense search *restricted to one cluster at a time*,
              pulling a decreasing quota from each: 5 chunks from the best
              cluster, 4 from the second, 3 from the third, 2 from the fourth,
              1 from the fifth. 15 results, drawn from five different regions
              of the corpus.

Why bother, when flat dense search would just return the 15 nearest vectors?
Because "nearest" collapses. A flat top-15 on a corpus with one dominant topic
is routinely 15 chunks of that one topic -- often 15 chunks of the same two
pages. The quota is a hard structural guarantee that the answer set spans the
five most relevant *regions*, weighted toward the best one but never owned by
it. It is diversification like MMR, but at cluster granularity and by
construction rather than by a per-result penalty: MMR pushes apart individual
vectors, this reserves seats per cluster before the search runs.

The cost is the obvious one: if the answer lives entirely in cluster 1, six of
your fifteen seats are spent elsewhere. Use it for broad/exploratory queries
("what does this corpus say about X") and survey-style retrieval; use plain
dense or hybrid for pinpoint lookups.

Cluster labels come from cluster.py or customcluster.py, so the *meaning* of a
"cluster" here is whatever you last clustered by -- KMeans topics, or your own
named criterion from criteria.md. With customcluster the ranking is legible:
you see "risk factors" beat "pricing" for this query, by cosine.

Three ways to score a cluster, `--rank-by`:
    probe     (default) one flat dense sweep `--probe` deep, then score each
              cluster by the mean of its best 3 hits inside it -- "how good is
              the best material this cluster has for this query". Sharper than
              centroid *and* cheaper (one ANN search, no stored vector read).
    centroid  mean of the cluster's member vectors, renormalized. Describes
              where the cluster sits on average, which is the problem: a large
              heterogeneous cluster averages out toward the corpus mean and
              ranks low even when it holds the single best chunk. Kept for
              comparison. Reads every stored vector once per run.
    name      embed the cluster's label and use that. The label is taken as
              `name: description`, with the description read back out of
              criteria.md (`--criteria`) -- Milvus stores only the name, and
              three words rank far worse than the sentence customcluster.py
              actually placed the rows with. Needs customcluster.py labels
              (KMeans clusters are called "0".."9", which embed to nothing
              useful), but reads no vectors -- much cheaper on a large corpus,
              and it ranks by what you *said* the bucket was rather than by
              where its members drifted.

The `unassigned` bucket (cluster -1) is skipped by default -- it is a leftovers
pile, not a topic. `--include-unassigned` lets it compete for a slot.

`--floor auto` makes each quota a *cap* rather than a mandate: a cluster told to
produce 5 chunks when it only has 3 worth having leaves the other seats empty
instead of padding them with whatever it had left. `--hybrid` adds a BM25 pass
inside each selected cluster, RRF-fused with the dense one, for queries that
hinge on an exact token.

`--adaptive` derives the funnel's *shape* from the cluster scores instead of
using the fixed 5/4/3/2/1: flat when the leading clusters are within noise of
each other, steep when one clearly wins. The allocation still sets the total
(its sum) and how many clusters are searched (its length), and every selected
cluster keeps at least one slot, so the coverage guarantee is untouched.
`--sharpness` controls how hard score gaps are amplified (0 = even split).

Stage 2 takes its chunks straight from the probe pool when that pool already
covers a cluster's quota, falling back to a per-cluster filtered search only when
it doesn't -- 7 Milvus round trips per query become 2. Measured byte-identical to
the filtered path on 75 of 75 query/config combinations; `--no-pool-reuse` forces
the old behaviour. Reads of the collection that don't depend on the query (the
cluster texts, their BM25 indexes, whether rows are named) are cached per
process; call clear_caches() if you re-ingest without restarting.

Prerequisites:
    python loadmilvus.py        # or extractpdf.py --store
    python cluster.py           # or: python customcluster.py   <- required

Run (PowerShell):
    .venv\\Scripts\\python.exe hierarchicalsearch.py "how is risk handled?"
    .venv\\Scripts\\python.exe hierarchicalsearch.py "attention" --allocation 8,4,2,1
    .venv\\Scripts\\python.exe hierarchicalsearch.py "pricing" --rank-by name
    .venv\\Scripts\\python.exe hierarchicalsearch.py "BLEU" --sort score --rerank

Also reachable from the normal front-end as `search.py "..." --method hierarchical`.
"""

import sys
import textwrap

import numpy as np

from loadmilvus import (DEFAULT_COLLECTION, INSERT_BATCH, connect, embed,
                        get_dim, get_model, parse_model_arg)

# The funnel: how many chunks to take from the 1st, 2nd, ... ranked cluster.
# Position i is the quota for the (i+1)-th best cluster, so len() is also how
# many clusters get searched at all. Override with --allocation 5,4,3,2,1.
DEFAULT_ALLOCATION = (5, 4, 3, 2, 1)

# Criterion file read by --rank-by name, for the label descriptions Milvus does
# not store. Mirrors customcluster.DEFAULT_CRITERIA, spelled out here so the
# common (centroid) path never pays for importing customcluster and sklearn.
DEFAULT_CRITERIA = "criteria.md"

# Cluster id used by cluster.py/customcluster.py for rows that fit nothing well.
UNASSIGNED = -1
UNASSIGNED_NAME = "unassigned"

# HNSW search width floor. Per-cluster limits are tiny (1-5), but a filtered
# search has to walk past every non-matching neighbour it meets, so a too-small
# ef can come back short on a small cluster. Cheap insurance at this scale.
MIN_EF = 64

# --rank-by probe: how deep the one flat sweep goes, and how many of a cluster's
# hits inside it are averaged into that cluster's score. Mean-of-top-3 rather
# than the single best hit, so one lucky chunk can't carry a whole cluster --
# but only among clusters that HAVE three hits; a thinner one is scored on what
# it has, on purpose. probe_clusters documents why penalising the shortfall
# measured worse.
DEFAULT_PROBE = 300
PROBE_TOP = 3

# --adaptive: how hard score differences are amplified before slots are shared
# out. Scores are z-scored first (cosines sit in a narrow band, so raw
# differences are tiny), then softmaxed; 0 gives an even split, higher values
# concentrate the budget on the leaders.
DEFAULT_SHARPNESS = 1.0

# `--floor auto`: a cluster's quota is a cap, not a mandate. The cutoff is the
# score of the (FLOOR_POOL_MULTIPLE x budget)-th best chunk in the probe pool --
# "don't hand a seat to a chunk that wouldn't even have made the flat top-22".
# Read off this query's own distribution rather than being an absolute cosine,
# for the reason cluster.py documents at DEFAULT_FLOOR_SIGMA: modern embedding
# models compress cosine into a narrow high band, so a fixed threshold is either
# inert or catastrophic depending on the model.
#
# This is a precision-vs-coverage dial, and 1.5 is a measured middle rather than
# a round number. It must stay above 1.0: at exactly 1.0 the floor admits
# nothing a flat search wouldn't already return, which would collapse the
# technique into flat dense. Calibrated on this repo's Nintendo corpus over
# three queries, of 15 seats it keeps 15/12/12 at 1.5, 14/11/12 at 1.2 and
# 15/13/15 at 2.0 -- i.e. 2.0 is nearly inert, 1.2 bites hard. Override per run
# with `--floor auto:1.2`.
FLOOR_POOL_MULTIPLE = 1.5

# `--hybrid`: how many candidates each stage pulls per cluster before RRF fuses
# them, as a multiple of that cluster's quota (with a small absolute floor).
# Fusion needs more candidates than seats or there is nothing to reorder.
HYBRID_DEPTH_MULTIPLE = 4
HYBRID_MIN_DEPTH = 20


# --------------------------------------------------------------------------
# per-process caches
# --------------------------------------------------------------------------
# Everything cached here is a property of the *collection*, not of the query:
# whether the rows carry `cluster_name`, which scheme they were clustered under,
# and the text of each cluster (plus the BM25 index built over it). A CLI run
# issues one query per process, so these change nothing there. A long-lived
# caller -- a Streamlit app, a notebook, a sweep -- issues hundreds against a
# static collection, and without them every query re-pages whole clusters out of
# Milvus and re-tokenizes them from scratch.
#
# The tradeoff is the honest one for a process-lifetime cache: re-ingesting or
# re-clustering *while the same process is running* leaves these stale. Call
# clear_caches() after any write to the collection.
_NAMED_CACHE = {}
_SCHEME_CACHE = {}
_CLUSTER_CORPUS_CACHE = {}
_BM25_CACHE = {}


def clear_caches():
    """Drop every cached read of the collection. Call after re-ingesting."""
    for cache in (_NAMED_CACHE, _SCHEME_CACHE, _CLUSTER_CORPUS_CACHE, _BM25_CACHE):
        cache.clear()


# --------------------------------------------------------------------------
# stage 1 -- rank the clusters
# --------------------------------------------------------------------------

def has_cluster_names(client, collection=DEFAULT_COLLECTION):
    """True if rows carry a `cluster_name` (i.e. customcluster.py labelled them).

    `cluster_name` is a dynamic field, and only customcluster.py writes it --
    after a plain cluster.py run no row has the key. Asking for a dynamic field
    that nothing has is rejected outright by some Milvus versions, so probe once
    here and leave it out of every output_fields list when it isn't there,
    rather than letting the real search fail on a collection clustered the
    ordinary way.

    Cached per collection: the answer is a property of how the rows were written
    and cannot change under a running query, so paying a round trip for it on
    every search is pure overhead.
    """
    if collection in _NAMED_CACHE:
        return _NAMED_CACHE[collection]
    try:
        client.query(collection_name=collection, filter="id >= 0",
                     output_fields=["cluster_name"], limit=1)
        answer = True
    except Exception:
        answer = False
    _NAMED_CACHE[collection] = answer
    return answer


def stored_scheme(client, collection=DEFAULT_COLLECTION):
    """Return the `cluster_scheme` these rows were clustered under, or None.

    Only customcluster.py writes it, so plain cluster.py collections have none.
    Used to check the criterion file on disk still matches what was stored.
    Cached per collection for the same reason has_cluster_names is.
    """
    if collection in _SCHEME_CACHE:
        return _SCHEME_CACHE[collection]
    try:
        rows = client.query(collection_name=collection, filter="id >= 0",
                            output_fields=["cluster_scheme"], limit=1)
    except Exception:
        rows = None
    scheme = rows[0].get("cluster_scheme") if rows else None
    _SCHEME_CACHE[collection] = scheme
    return scheme


def fetch_cluster_rows(client, with_vectors=True, collection=DEFAULT_COLLECTION,
                       named=True):
    """Return every row's cluster label (and optionally its vector).

    Pages with query_iterator for the same reason cluster.fetch_all does: a
    plain query is capped at 16384 rows, and a silently truncated read here
    would produce centroids computed from a slice of each cluster.

    `with_vectors=False` skips the `embedding` field entirely, which is what
    makes --rank-by name cheap: it moves ints and short strings instead of a
    float array per row.
    """
    if not client.has_collection(collection):
        raise SystemExit(
            f"Collection '{collection}' does not exist. Run loadmilvus.py "
            f"(or extractpdf.py --store) first.")
    client.load_collection(collection)

    fields = ["cluster"] + (["cluster_name"] if named else []) \
        + (["embedding"] if with_vectors else [])
    iterator = client.query_iterator(
        collection_name=collection,
        filter="id >= 0",
        output_fields=fields,
        batch_size=INSERT_BATCH,
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

    if not rows:
        raise SystemExit(f"Collection '{collection}' is empty.")
    if all(row.get("cluster") is None for row in rows):
        raise SystemExit(
            "No `cluster` labels found in the collection. Hierarchical search "
            "ranks clusters, so it needs them:\n"
            "    python cluster.py            # KMeans topics\n"
            "    python customcluster.py      # your own named criterion")
    return rows


def cluster_profiles(rows, with_vectors=True):
    """Group rows into {cluster_id: {name, size, centroid}}.

    The centroid is the mean of the cluster's (already L2-normalized) member
    vectors, renormalized so a dot product against it is cosine again -- the
    mean of unit vectors is not itself unit length. `centroid` is None when
    vectors weren't fetched.
    """
    profiles = {}
    for row in rows:
        cid = row.get("cluster")
        if cid is None:
            continue
        slot = profiles.setdefault(int(cid), {"name": None, "vectors": []})
        if slot["name"] is None:
            slot["name"] = row.get("cluster_name")
        if with_vectors:
            slot["vectors"].append(row["embedding"])

    out = {}
    for cid, slot in profiles.items():
        centroid = None
        if with_vectors and slot["vectors"]:
            centroid = np.asarray(slot["vectors"], dtype="float32").mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid = centroid / (norm if norm else 1.0)
        name = slot["name"] or (UNASSIGNED_NAME if cid == UNASSIGNED else str(cid))
        out[cid] = {"name": name, "size": len(slot["vectors"]), "centroid": centroid}
    return out


def label_descriptions(path=DEFAULT_CRITERIA):
    """Return ({label name: description}, scheme) from a criteria.md.

    Milvus stores only the label's *name* ("history and group structure"), never
    the description that customcluster.py actually embedded to place the rows.
    That description is where nearly all the signal lives -- the name is three
    words, the description is the sentence that says "subsidiaries and
    associates, consolidated group companies" -- so name-mode ranking reads it
    back out of the criterion file and ranks on the same anchor text the
    clustering used.

    Returns ({}, None) if the file is missing or unparseable: the descriptions
    are an enrichment, and losing them should degrade the ranking to bare names,
    not kill the search.
    """
    try:
        from customcluster import parse_criteria
        config = parse_criteria(path)
    except SystemExit:
        return {}, None
    return dict(config["labels"]), config["scheme"]


def name_centroids(model, profiles, criteria=DEFAULT_CRITERIA, scheme=None):
    """Embed each cluster's name (+ its criteria.md description) as its centroid.

    The cheap ranking path: no stored vector is read, only the handful of label
    strings. Meaningful for customcluster.py labels, where the name is a phrase
    that lands somewhere sensible in the embedding space; plain cluster.py names
    are the digits "0".."9" and embed to noise, so this refuses to run on them.

    `scheme` is the criterion the *collection* was clustered under. If the
    criterion file on disk has since been edited to a different scheme, its
    descriptions no longer describe these rows, so say so rather than ranking
    on a stale sentence.
    """
    ids = sorted(profiles)
    names = [profiles[cid]["name"] for cid in ids]
    if all(name.lstrip("-").isdigit() or name == UNASSIGNED_NAME for name in names):
        raise SystemExit(
            "--rank-by name needs named clusters, but this collection's clusters "
            "are numbered (they came from cluster.py). Use --rank-by centroid, "
            "or re-cluster with customcluster.py to get names.")

    from customcluster import anchor_text          # same "name: description" form

    descriptions, criteria_scheme = label_descriptions(criteria)
    if descriptions and scheme and criteria_scheme and scheme != criteria_scheme:
        print(f"  Warning: '{criteria}' now holds scheme '{criteria_scheme}' but "
              f"these rows were clustered under '{scheme}'. Ranking on the bare "
              f"names instead of its descriptions.")
        descriptions = {}

    texts = [anchor_text(name, descriptions.get(name, "")) for name in names]
    matched = sum(1 for name in names if descriptions.get(name))
    if descriptions:
        print(f"Embedding {len(texts)} cluster label(s) "
              f"({matched}/{len(texts)} enriched with descriptions from {criteria})...")
    else:
        print(f"Embedding {len(texts)} cluster name(s) "
              f"(no usable descriptions in {criteria}; names only)...")
    vectors = np.asarray(embed(model, texts), dtype="float32")
    return {cid: vectors[i] for i, cid in enumerate(ids)}


def probe_clusters(client, query_vec, pool=DEFAULT_PROBE, ef=MIN_EF,
                   collection=DEFAULT_COLLECTION, named=True, with_records=True):
    """Score clusters by one flat sweep.

    Returns ({cluster: score}, profiles, pool_scores, pool_records).

    Runs a single unfiltered dense search `pool` deep, buckets the hits by
    cluster, and scores each cluster by the mean of its best PROBE_TOP hits --
    "how good is the best material this cluster actually has for this query",
    rather than "where does this cluster sit on average".

    That distinction is the whole point. A centroid is the mean of every member,
    so a large heterogeneous cluster averages out toward the corpus mean and
    ranks low even when it holds the single best chunk in the corpus; measured
    on this repo's Nintendo corpus, centroid ranking put `corporate governance`
    8th of 12 while it held the 2nd-best chunk overall. Centroid also spread all
    12 clusters across a 0.13 cosine band, where consecutive ranks differ by
    less than the noise. Probing is both sharper and cheaper: one ANN search
    instead of reading every stored vector.

    **A cluster with fewer than PROBE_TOP hits is scored on what it has**, not
    penalised for the shortfall -- `np.mean` divides by the hits present. That
    looks like a bug (a cluster with one hit is scored by that single hit, which
    is the "one lucky chunk carries the cluster" case the mean is meant to
    prevent) and penalising it was tried. It is wrong, and measurably so.

    Hit count conflates relevance with cluster *size*: a small cluster cannot
    place many chunks in the pool however well it matches. Over 25 queries on
    this repo's Nintendo corpus, padding the missing slots with the pool's
    weakest score changed the top-5 on 3 and the top-3 on 2 -- and the clusters
    it demoted out of rank 1 were `financial exchange rate and disaster risks`
    for "foreign exchange rate risk" and `climate and environmental initiatives`
    for "greenhouse gas emissions". Both are exactly the right cluster; both have
    only 2 pool hits because both are small. Any penalty proportional to hit
    count demotes the narrow, precisely-matching cluster, which is the one case
    the technique most needs to get right.

    The genuine cost -- a 2-hit cluster taking a quota of 5 and filling the rest
    from chunks that never made the pool -- is real but belongs to allocation,
    not ranking, and `--floor auto` already addresses it by making the quota a
    cap. See resolve_pool_floor.

    A cluster with no hit in the pool gets no score and cannot be selected --
    correct, since nothing it holds ranked in the top `pool` for this query.

    `with_records=True` also asks for each hit's text/source/seq, so stage 2 can
    take its chunks straight from this pool instead of re-searching per cluster
    (see hierarchical_search). Costs one text payload for the pool; saves one
    round trip per selected cluster.
    """
    client.load_collection(collection)
    fields = ["cluster"] + (["cluster_name"] if named else [])
    if with_records:
        fields += ["text", "source", "seq"]
    hits = client.search(
        collection_name=collection,
        data=[query_vec],
        anns_field="embedding",
        limit=pool,
        output_fields=fields,
        search_params={"metric_type": "COSINE", "params": {"ef": max(ef, pool)}},
    )[0]

    buckets = {}
    for hit in hits:
        cid = hit["entity"].get("cluster")
        if cid is None:
            continue
        slot = buckets.setdefault(int(cid), {"scores": [], "name": None,
                                             "records": []})
        slot["scores"].append(hit["distance"])
        if slot["name"] is None:
            slot["name"] = hit["entity"].get("cluster_name")
        if with_records:
            slot["records"].append({
                "id": hit["id"],
                "text": hit["entity"].get("text", ""),
                "source": hit["entity"].get("source"),
                "seq": hit["entity"].get("seq"),
                "cluster": cid,
                "cluster_name": hit["entity"].get("cluster_name"),
                "score": hit["distance"],
                "cosine": hit["distance"],
            })

    if not buckets:
        raise SystemExit(
            "No `cluster` labels found in the collection. Hierarchical search "
            "ranks clusters, so it needs them:\n"
            "    python cluster.py            # KMeans topics\n"
            "    python customcluster.py      # your own named criterion")

    scores, profiles = {}, {}
    for cid, slot in buckets.items():
        # Mean over the hits present -- deliberately not padded to PROBE_TOP.
        # See the docstring: penalising thin clusters demotes small ones that
        # match precisely, which measured worse.
        best = sorted(slot["scores"], reverse=True)[:PROBE_TOP]
        scores[cid] = float(np.mean(best))
        profiles[cid] = {
            "name": slot["name"] or (UNASSIGNED_NAME if cid == UNASSIGNED
                                     else str(cid)),
            # Corpus-wide size is unknown here -- nothing was read but the pool.
            "size": None,
            "hits": len(slot["scores"]),
            "centroid": None,
        }
    print(f"Probed top {len(hits)} chunks; {len(profiles)} cluster(s) represented.")
    # The pool doubles as this query's score distribution, which is what
    # resolve_pool_floor reads the `--floor auto` cutoff off.
    pool_records = {cid: slot["records"] for cid, slot in buckets.items()}
    return scores, profiles, [h["distance"] for h in hits], pool_records


def resolve_pool_floor(pool_scores, budget, floor, multiple=FLOOR_POOL_MULTIPLE):
    """Turn the configured `floor` into a concrete cosine cutoff, or None.

    `off`/None disables it. A number is used as an absolute cosine. `auto` (or
    `auto:M` for a one-off multiple) takes the score of the (M x budget)-th best
    chunk in the probe pool: a chunk below that did not merit a seat on any
    reading, and the only reason it got one is that its cluster was told to
    produce more chunks than it had worth giving.
    """
    if floor in (None, "off"):
        return None
    if isinstance(floor, str):
        head, sep, tail = floor.strip().lower().partition(":")
        if head == "auto":
            if not pool_scores:
                return None
            if sep:
                try:
                    multiple = float(tail)
                except ValueError:
                    raise SystemExit(
                        f"--floor auto:M needs a number for M, got {tail!r}.")
            rank = min(len(pool_scores) - 1, max(1, int(multiple * budget)) - 1)
            return float(sorted(pool_scores, reverse=True)[rank])
        floor = float(floor)
    return float(floor) if floor > 0 else None


def apply_score_floor(records, floor):
    """Drop records whose cosine is below `floor`. Returns (kept, dropped count).

    Records with no cosine are kept: under `--hybrid` those are BM25-only hits,
    and a chunk BM25 ranks top for an exact term is precisely what the lexical
    pass was added to surface. Nothing is backfilled to replace what is dropped
    -- the quota is a cap, so a cluster that cannot fill it simply returns less.
    """
    if floor is None:
        return records, 0
    kept = [r for r in records if r.get("cosine") is None or r["cosine"] >= floor]
    return kept, len(records) - len(kept)


def adaptive_allocation(scores, budget, sharpness=DEFAULT_SHARPNESS, minimum=1):
    """Share `budget` slots across ranked clusters in proportion to their scores.

    A fixed 5/4/3/2/1 commits to the same steep funnel whether stage 1 was
    decisive or a coin-flip. This derives the shape from the scores instead: it
    z-scores them (cosines live in a narrow band, so absolute gaps are tiny and
    meaningless on their own), softmaxes with `sharpness`, and hands out the
    budget by weight -- flat when the leaders are within noise of each other,
    steep when one cluster clearly wins.

    Every selected cluster keeps at least `minimum` slot, so the technique's
    coverage guarantee survives; only the *shape* adapts, never the total.
    Quotas are sorted descending before returning, so a lower-ranked cluster can
    never out-draw a better one after rounding.
    """
    n = len(scores)
    if n == 0:
        return []
    if budget <= n * minimum:
        # Too few slots to give everyone the minimum: the best `budget` clusters
        # take one each and the tail gets nothing.
        return [1] * budget + [0] * (n - budget)

    values = np.asarray(scores, dtype="float64")
    spread = values.std()
    z = (values - values.mean()) / spread if spread > 1e-9 else np.zeros(n)
    weights = np.exp(sharpness * z)
    weights /= weights.sum()

    free = budget - n * minimum
    raw = weights * free
    quotas = np.floor(raw).astype(int)
    # Largest-remainder rounding, so the quotas sum to exactly `free`.
    for i in np.argsort(-(raw - np.floor(raw)))[:int(free - quotas.sum())]:
        quotas[i] += 1
    quotas = np.sort(quotas + minimum)[::-1]
    return [int(q) for q in quotas]


def rank_clusters(profiles, query_vec, top_n, include_unassigned=False,
                  centroids=None, scores=None):
    """Score every cluster against the query and return the best `top_n`.

    Returns a list of (cluster_id, score) sorted best-first. Score is cosine:
    query and centroids are both unit vectors, so it is a plain dot product.

    `unassigned` is excluded unless asked for -- it is the bucket for rows that
    matched nothing well, so a high score there means the query missed the
    corpus, not that the bucket is a good place to search.
    """
    query_vec = np.asarray(query_vec, dtype="float32")
    scored = []
    for cid, profile in profiles.items():
        if cid == UNASSIGNED and not include_unassigned:
            continue
        if scores is not None:                      # probe mode: already scored
            if cid not in scores:
                continue
            scored.append((cid, scores[cid]))
            continue
        centroid = centroids[cid] if centroids is not None else profile["centroid"]
        if centroid is None:
            continue
        scored.append((cid, float(query_vec @ centroid)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


# --------------------------------------------------------------------------
# stage 2 -- search inside each winning cluster
# --------------------------------------------------------------------------

def search_in_cluster(client, query_vec, cluster_id, limit, ef,
                      collection=DEFAULT_COLLECTION, named=True):
    """Dense search restricted to one cluster. Returns up to `limit` records.

    Same ANN search as search.dense_search, plus a `cluster == n` filter, so the
    per-cluster quota is enforced by Milvus rather than by post-filtering a flat
    result set (which would need an unbounded pool to guarantee the small
    clusters were represented at all).
    """
    hits = client.search(
        collection_name=collection,
        data=[query_vec],
        anns_field="embedding",
        limit=limit,
        filter=f"cluster == {int(cluster_id)}",
        output_fields=["text", "source", "seq", "cluster"]
                      + (["cluster_name"] if named else []),
        search_params={"metric_type": "COSINE", "params": {"ef": ef}},
    )[0]
    return [{
        "id": h["id"],
        "text": h["entity"].get("text", ""),
        "source": h["entity"].get("source"),
        "seq": h["entity"].get("seq"),
        "cluster": h["entity"].get("cluster"),
        "cluster_name": h["entity"].get("cluster_name"),
        "score": h["distance"],
        # Kept separately from `score`: RRF overwrites `score` under --hybrid,
        # and the floor has to stay a cosine comparison.
        "cosine": h["distance"],
    } for h in hits]


def cluster_corpus(client, cluster_id, collection=DEFAULT_COLLECTION, named=True,
                   with_vectors=False):
    """Return every chunk in one cluster, for BM25 to rank locally.

    The cluster-restricted equivalent of search.fetch_corpus: BM25 runs
    client-side over the stored text (the collection has no sparse field), but
    only over the one cluster being searched, so the term statistics are the
    cluster's own rather than the whole corpus's. Pages with query_iterator so a
    cluster larger than the query cap is not silently truncated.

    Cached per (collection, cluster, with_vectors) -- see the note at
    _CLUSTER_CORPUS_CACHE. **The returned rows are the cached objects, so callers
    must not mutate them**; build new dicts instead.
    """
    key = (collection, int(cluster_id), bool(with_vectors), bool(named))
    if key in _CLUSTER_CORPUS_CACHE:
        return _CLUSTER_CORPUS_CACHE[key]

    iterator = client.query_iterator(
        collection_name=collection,
        filter=f"cluster == {int(cluster_id)}",
        output_fields=["text", "source", "seq", "cluster"]
                      + (["cluster_name"] if named else [])
                      + (["embedding"] if with_vectors else []),
        batch_size=INSERT_BATCH,
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
    _CLUSTER_CORPUS_CACHE[key] = rows
    return rows


def cluster_bm25(corpus, collection, cluster_id):
    """Return a BM25 index over `corpus`, built once per cluster and reused.

    Tokenizing and indexing a cluster is the same work every query, and it is
    the expensive half of the lexical pass -- rank_bm25 rebuilds its term
    statistics from scratch on construction. Keyed by the row ids actually being
    indexed, so a corpus filtered differently (by a different `--floor` cutoff,
    say) builds its own index rather than silently reusing one over other rows.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit(
            "Hybrid search needs rank-bm25. Install it:\n"
            "    .venv\\Scripts\\python.exe -m pip install rank-bm25")
    from search import tokenize

    key = (collection, int(cluster_id), tuple(row["id"] for row in corpus))
    index = _BM25_CACHE.get(key)
    if index is None:
        index = BM25Okapi([tokenize(row["text"]) for row in corpus])
        _BM25_CACHE[key] = index
    return index


def hybrid_in_cluster(client, query, query_vec, cluster_id, limit, ef,
                      collection=DEFAULT_COLLECTION, named=True, cutoff=None):
    """Dense + BM25 inside one cluster, fused by RRF. Returns up to `limit`.

    Stage 2 is otherwise pure dense, which is blind to exact tokens -- and a
    statutory corpus is full of queries that hinge on one ("subsidiaries",
    "Tier 1", a company name). This runs both retrievers against the same single
    cluster and fuses them with Reciprocal Rank Fusion, the same way
    search.py's `hybrid` method does for the whole collection.

    `score` on the returned records is the RRF score (rank-based, not a cosine);
    each record's `cosine` is preserved where it has one, so the floor and any
    later reporting still have the real similarity to work with.

    `cutoff` is applied to *both* candidate lists before fusion, never to the
    fused output. Two reasons. RRF reorders by rank, so filtering afterwards
    would strike out whichever weak chunk happened to land in the top `limit`
    and leave the seat empty while an eligible chunk sat at rank 8. And BM25
    hits need a cosine to be judged on at all: measured on the Nintendo corpus,
    letting lexical-only hits bypass the floor filled 9 of 15 seats with chunks
    the floor had just emptied ("no major customers", profit totals) because
    they shared a token with the query. So the cluster's chunks are fetched with
    their vectors, scored against the query, and filtered before BM25 ranks
    them -- BM25 reorders what is semantically eligible, it does not smuggle in
    what isn't.

    Filtering BM25's input rather than its output does change the term
    statistics it computes from (IDF and average document length are
    corpus-wide), which sounds like a reason not to. Measured on cluster 2 of
    the Nintendo corpus for "where are the subsidiaries", it is a clear win: the
    unfiltered cluster's BM25 top 5 is dominated by chunks that merely repeat
    the token (cosines 0.24-0.33), while the chunk listing "Nintendo (Hong Kong)
    Limited; Nintendo of Taiwan Co., Ltd." -- the literal answer -- sits at BM25
    rank 11. Restricting the corpus to floor-eligible chunks lifts it to rank 3.
    """
    from search import reciprocal_rank_fusion, tokenize

    depth = max(limit * HYBRID_DEPTH_MULTIPLE, HYBRID_MIN_DEPTH)
    dense = search_in_cluster(client, query_vec, cluster_id, depth,
                              max(ef, depth), collection, named)
    dense, _ = apply_score_floor(dense, cutoff)

    corpus = cluster_corpus(client, cluster_id, collection, named,
                            with_vectors=cutoff is not None)
    if cutoff is not None and corpus:
        # Stored vectors are L2-normalized, so a dot product is cosine. Done as
        # one matmul rather than a per-row asarray: the cluster can be thousands
        # of rows and this runs once per selected cluster per query.
        #
        # New dicts, never a mutation of `corpus` -- those rows are the cached
        # objects (see cluster_corpus), and popping `embedding` out of them would
        # corrupt the cache for every later query.
        matrix = np.asarray([row["embedding"] for row in corpus], dtype="float32")
        cosines = matrix @ np.asarray(query_vec, dtype="float32")
        corpus = [
            {**{k: v for k, v in row.items() if k != "embedding"},
             "cosine": float(cosine)}
            for row, cosine in zip(corpus, cosines)
        ]
        corpus, _ = apply_score_floor(corpus, cutoff)

    # A floor can leave a cluster with nothing eligible, and BM25 divides by its
    # document count. Nothing to rank means nothing for BM25 to contribute.
    if corpus:
        bm25 = cluster_bm25(corpus, collection, cluster_id)
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
        lexical = [{**row, "score": float(score)} for row, score in ranked[:depth]]
    else:
        lexical = []
    return reciprocal_rank_fusion([dense, lexical], limit)


def hierarchical_search(client, model, query, allocation=DEFAULT_ALLOCATION,
                        ef=MIN_EF, rank_by="probe", include_unassigned=False,
                        sort="cluster", collection=DEFAULT_COLLECTION,
                        criteria=DEFAULT_CRITERIA, adaptive=False,
                        sharpness=DEFAULT_SHARPNESS, probe=DEFAULT_PROBE,
                        floor=None, hybrid=False, reuse_pool=True):
    """Rank clusters against `query`, then take a quota of chunks from each.

    `allocation[i]` chunks come from the (i+1)-th best cluster, so the default
    (5, 4, 3, 2, 1) returns 15 results spread over 5 clusters. Fewer clusters
    than quotas (or a cluster smaller than its quota) just yields fewer results;
    nothing is backfilled from another cluster, because backfilling would quietly
    undo the guarantee the allocation exists to make.

    With `adaptive=True` the allocation supplies only the *budget* (its sum) and
    the number of clusters (its length); the shape is derived from the cluster
    scores by adaptive_allocation.

    Returns (records, ranked_clusters, profiles, quotas) -- `quotas` is the
    allocation actually used, which is what the caller should report.

    Each record carries `cluster`, `cluster_name` and `cluster_rank` alongside
    the usual fields, so a caller can see which region every hit came from.
    `sort="cluster"` keeps the funnel order (best cluster's hits first);
    `sort="score"` re-sorts everything by cosine, ignoring provenance.

    `reuse_pool=True` (default, `--rank-by probe` and non-hybrid only) takes each
    cluster's chunks from the probe pool that already ranked them, falling back
    to a filtered search only for a cluster the pool did not cover to its quota.
    See the note in the stage-2 loop for why that is sound.
    """
    query_vec = np.asarray(embed(model, [query])[0], dtype="float32")
    named = has_cluster_names(client, collection)

    scores, centroids, pool_scores = None, None, []
    pool_records = {}
    if rank_by == "probe":
        # One flat sweep. No stored vector is read, and nothing is fetched for
        # clusters that had no hit in the pool.
        scores, profiles, pool_scores, pool_records = probe_clusters(
            client, query_vec.tolist(), probe, ef, collection, named,
            with_records=reuse_pool and not hybrid)
    else:
        if isinstance(floor, str) and floor.strip().lower().startswith("auto"):
            raise SystemExit(
                "`--floor auto` reads its cutoff off the probe pool, which only "
                "--rank-by probe builds. Use --rank-by probe, or give --floor an "
                "explicit cosine.")
        with_vectors = rank_by == "centroid"
        rows = fetch_cluster_rows(client, with_vectors, collection, named)
        profiles = cluster_profiles(rows, with_vectors)
        centroids = None if with_vectors else name_centroids(
            model, profiles, criteria, stored_scheme(client, collection))

    ranked = rank_clusters(profiles, query_vec, len(allocation),
                           include_unassigned, centroids, scores)
    if not ranked:
        raise SystemExit("No clusters available to search.")

    if adaptive:
        quotas = adaptive_allocation([score for _, score in ranked],
                                     sum(allocation), sharpness)
    else:
        quotas = list(allocation)
        if len(ranked) < len(quotas):
            print(f"Note: only {len(ranked)} cluster(s) available; using the "
                  f"first {len(ranked)} quota(s) of {list(allocation)}.")

    cutoff = resolve_pool_floor(pool_scores, sum(allocation), floor)
    if cutoff is not None:
        print(f"Score floor: {cutoff:.4f} -- quotas are caps, not mandates.")

    results, dropped = [], 0
    served_from_pool = 0
    for rank, ((cid, cluster_score), quota) in enumerate(zip(ranked, quotas),
                                                         start=1):
        if quota <= 0:
            continue
        if hybrid:
            # The floor is applied inside, before RRF reorders the candidates.
            hits = hybrid_in_cluster(client, query, query_vec.tolist(), cid,
                                     quota, ef, collection, named, cutoff)
            cut = quota - len(hits)
        else:
            # The probe pool is already a cosine-ranked list of this cluster's
            # best chunks -- that is how the cluster got ranked at all -- so when
            # it covers the quota, re-asking Milvus for the same chunks under a
            # `cluster == n` filter buys nothing but a round trip. The top `quota`
            # of a cluster's pool hits is what a cluster-restricted search of that
            # depth returns, and the flat sweep saw them at ef=`probe` where the
            # filtered search would use ef=64, so if the two ever disagree the
            # pool is the higher-recall read.
            #
            # The fallback matters and is not rare: it fires for any cluster whose
            # pool coverage fell short of its quota, which is exactly the thin
            # tail of the funnel. Those clusters get the original search.
            available = pool_records.get(cid, []) if reuse_pool else []
            if len(available) >= quota:
                hits = [dict(rec) for rec in available[:quota]]
                served_from_pool += 1
            else:
                hits = search_in_cluster(client, query_vec.tolist(), cid, quota,
                                         max(ef, quota), collection, named)
            hits, cut = apply_score_floor(hits, cutoff)
        dropped += max(cut, 0)
        for hit in hits:
            hit["cluster_rank"] = rank
            hit["cluster_score"] = cluster_score
            # cluster_name is absent on cluster.py labels; fall back to the id.
            hit["cluster_name"] = hit.get("cluster_name") or profiles[cid]["name"]
            results.append(hit)

    if dropped:
        print(f"  {dropped} seat(s) left empty: no chunk in that cluster cleared "
              f"the floor.")
    if served_from_pool:
        searched = sum(1 for q in quotas[:len(ranked)] if q > 0) - served_from_pool
        print(f"  {served_from_pool} cluster(s) served from the probe pool; "
              f"{searched} needed a filtered search.")
    if sort == "score":
        results.sort(key=lambda rec: rec["score"], reverse=True)
    return results, ranked, profiles, quotas


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_ranking(ranked, profiles, allocation, adaptive=False):
    """Print stage 1: the cluster leaderboard and the quota each one won."""
    how = "adaptive" if adaptive else "fixed"
    print(f"\nCluster ranking (stage 1, {how} allocation)\n" + "-" * 60)
    for rank, (cid, score) in enumerate(ranked, start=1):
        quota = allocation[rank - 1] if rank - 1 < len(allocation) else 0
        profile = profiles[cid]
        # Probe mode never reads the corpus, so it knows how many of a cluster's
        # chunks made the pool but not how many exist.
        if profile.get("size"):
            detail = f"{profile['size']} docs"
        elif profile.get("hits"):
            detail = f"{profile['hits']} in pool"
        else:
            detail = "size n/a"
        print(f"  #{rank}  cosine={score:.4f}  take {quota:>2}  "
              f"[{cid}] {profile['name']}  ({detail})")


def print_results(query, records, allocation, reranked=False, preview=None,
                  width=96):
    """Print stage 2: the retrieved chunks, tagged with the cluster they came from.

    Full chunk text by default, wrapped -- same reasoning as search.print_results:
    a truncated hit tells you something matched but not whether it answers the
    question. `preview=N` truncates when scanning a lot at once.
    """
    label = "hierarchical" + (" + rerank" if reranked else "")
    plan = "/".join(str(a) for a in allocation)
    print(f"\nTop {len(records)} for {query!r}  (method: {label}, plan: {plan})\n"
          + "-" * width)
    for rank, rec in enumerate(records, start=1):
        text = " ".join(rec["text"].split())
        truncated = preview is not None and len(text) > preview
        if preview is not None:
            text = text[:preview]

        seq = rec.get("seq")
        where = f"  source={rec.get('source')}" + (
            f"  chunk #{seq}" if seq is not None and int(seq) >= 0 else "")
        origin = (f"  cluster #{rec.get('cluster_rank')} "
                  f"[{rec.get('cluster')}] {rec.get('cluster_name')}")
        # Under --hybrid `score` is an RRF score, which says nothing on its own;
        # show the cosine beside it, and mark BM25-only hits as having none.
        cosine = rec.get("cosine")
        extra = ""
        if cosine is not None and abs(cosine - rec["score"]) > 1e-9:
            extra = f" cosine={cosine:.4f}"
        elif cosine is None:
            extra = " cosine=n/a (BM25 only)"
        print(f"\n[{rank}] score={rec['score']:.4f}{extra}{origin}{where}  "
              f"({len(rec['text'])} chars)")
        for line in textwrap.wrap(text + ("..." if truncated else ""),
                                  width=width - 4) or [""]:
            print(f"    {line}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_allocation(text):
    """Parse `--allocation 5,4,3,2,1` into a tuple of per-rank quotas."""
    try:
        quotas = tuple(int(part) for part in text.replace(" ", "").split(","))
    except ValueError:
        raise SystemExit(
            f"--allocation must be comma-separated integers, got {text!r} "
            f"(e.g. 5,4,3,2,1).")
    if not quotas or any(q < 0 for q in quotas):
        raise SystemExit("--allocation must be one or more non-negative integers.")
    if all(q == 0 for q in quotas):
        raise SystemExit("--allocation is all zeros; nothing would be retrieved.")
    return quotas


def arg_value(argv, flag, default=None):
    """Return the value following `flag` in argv, else `default`."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1]
    return default


def parse_query(argv):
    """Return the first positional (non-flag) argument as the search query."""
    flags_with_values = {"--model", "--allocation", "--rank-by", "--sort", "--ef",
                         "--preview", "--k", "--rerank-model", "--criteria",
                         "--sharpness", "--probe", "--floor"}
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if arg in flags_with_values:
            skip = True
            continue
        if arg.startswith("--"):
            continue
        return arg
    return None


def load_query_model(client, model_name, collection=DEFAULT_COLLECTION):
    """Load the embedding model, checking its dim against the stored vectors.

    The query, the documents and (in centroid mode) the centroids all have to
    live in one space; a dim mismatch is the cheap, realistic half of that check.
    """
    info = client.describe_collection(collection)
    stored = next((f["params"]["dim"] for f in info["fields"]
                   if f["name"] == "embedding"), None)
    if stored is None:
        raise SystemExit(
            "Collection has no 'embedding' field -- was it built by loadmilvus?")
    model = get_model(model_name)
    if get_dim(model) != stored:
        raise SystemExit(
            f"Model '{model_name}' is {get_dim(model)}-dim but the stored vectors "
            f"are {stored}-dim. Query with the model they were built with.")
    return model


def main():
    """Parse options, run the two-stage search, print the ranking and results."""
    # Chunk text can hold glyphs outside the console's legacy code page; don't
    # let printing results crash the run. (Same guard as search.py / cluster.py.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    argv = sys.argv
    query = parse_query(argv)
    if not query:
        raise SystemExit(
            'Usage: hierarchicalsearch.py "your query" [--allocation 5,4,3,2,1] '
            '[--adaptive] [--sharpness F] [--floor auto|off|F] [--hybrid] '
            '[--rank-by probe|centroid|name] [--probe N] [--criteria <path>] '
            '[--sort cluster|score] [--include-unassigned] [--model <key>] '
            '[--rerank] [--rerank-model <id>] [--k N] [--ef N] [--preview N] '
            '[--no-pool-reuse]')

    allocation = parse_allocation(arg_value(argv, "--allocation",
                                            ",".join(map(str, DEFAULT_ALLOCATION))))
    rank_by = arg_value(argv, "--rank-by", "probe").lower()
    if rank_by not in ("probe", "centroid", "name"):
        raise SystemExit(
            f"--rank-by must be probe, centroid or name, got {rank_by!r}.")
    criteria = arg_value(argv, "--criteria", DEFAULT_CRITERIA)
    adaptive = "--adaptive" in argv
    sharpness = float(arg_value(argv, "--sharpness", DEFAULT_SHARPNESS))
    probe = int(arg_value(argv, "--probe", DEFAULT_PROBE))
    hybrid = "--hybrid" in argv
    floor = arg_value(argv, "--floor", "off")
    if not floor.strip().lower().startswith(("off", "auto")):
        try:
            float(floor)
        except ValueError:
            raise SystemExit(
                f"--floor must be `auto`, `auto:M`, `off`, or a cosine, "
                f"got {floor!r}.")
    sort = arg_value(argv, "--sort", "cluster").lower()
    if sort not in ("cluster", "score"):
        raise SystemExit(f"--sort must be cluster or score, got {sort!r}.")

    model_name = parse_model_arg(argv)
    include_unassigned = "--include-unassigned" in argv
    ef = int(arg_value(argv, "--ef", MIN_EF))
    do_rerank = "--rerank" in argv
    # Default k is the whole allocation: the funnel decides how many results
    # there are, and --k only matters as "how many survive the reranker".
    k = int(arg_value(argv, "--k", sum(allocation)))
    preview_arg = arg_value(argv, "--preview")
    preview = int(preview_arg) if preview_arg is not None else None

    client = connect()
    model = load_query_model(client, model_name)

    results, ranked, profiles, quotas = hierarchical_search(
        client, model, query, allocation, ef, rank_by, include_unassigned, sort,
        DEFAULT_COLLECTION, criteria, adaptive, sharpness, probe, floor, hybrid,
        reuse_pool="--no-pool-reuse" not in argv)

    print_ranking(ranked, profiles, quotas, adaptive)

    if do_rerank:
        from search import DEFAULT_RERANK_MODEL, rerank
        rerank_model = arg_value(argv, "--rerank-model", DEFAULT_RERANK_MODEL)
        results = rerank(query, results, rerank_model, k)
    elif k < len(results):
        results = results[:k]

    print_results(query, results, quotas, do_rerank, preview=preview)


if __name__ == "__main__":
    main()
