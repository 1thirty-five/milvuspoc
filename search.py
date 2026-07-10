"""
Search the stored document vectors — choose the retrieval technique AND model.

This is the retrieval front-end for the collection built by loadmilvus/extractpdf.
Six techniques, selectable with `--method`, plus an optional reranker on top:

    dense     semantic vector search over the HNSW index (COSINE). Embeds the
              query with `--model` and asks Milvus for the nearest chunks. Great
              at meaning/paraphrase, weak on exact tokens.
    lexical   BM25 keyword search over the stored `text` (no model, no vectors).
              Great at exact terms (codes, names, "WMT 2014"), blind to paraphrase.
    tfidf     TF-IDF cosine keyword search (the classic lexical baseline; a
              simpler, purely term-frequency alternative to BM25).
    hybrid    run dense AND lexical, then fuse with Reciprocal Rank Fusion (RRF,
              rank-based). The strong default — semantic recall + keyword precision.
    weighted  hybrid via weighted SUM of min-max-normalized dense & lexical
              scores; tune the balance with `--alpha` (1.0 = all dense, 0 = all
              lexical). Use when you want an explicit dial instead of RRF.
    mmr       dense retrieval + Maximal Marginal Relevance: greedily pick results
              that are relevant to the query but DIVERSE from each other, so the
              top-k aren't near-duplicates. Balance with `--lambda` (1.0 = pure
              relevance, 0 = pure diversity).

    --rerank  re-score the retrieved candidates with a cross-encoder that reads
              the query and each chunk TOGETHER (joint attention), then keep the
              top `--k`. More accurate ordering; applied only to the shortlist,
              and it stacks on ANY method above.

Each function does one thing and returns plain records ({id, text, source,
score}) so a UI layer can call any stage independently.

Prerequisites:
    python extractpdf.py --store   # build the collection first
    pip install rank-bm25          # only for lexical / hybrid

Run (PowerShell):
    .venv\\Scripts\\python.exe search.py "how does the model avoid recurrence?"
    .venv\\Scripts\\python.exe search.py "multi-head attention" --method lexical
    .venv\\Scripts\\python.exe search.py "scaled dot-product" --method hybrid --rerank
    .venv\\Scripts\\python.exe search.py "attention" --method weighted --alpha 0.7
    .venv\\Scripts\\python.exe search.py "attention" --method mmr --lambda 0.5 --k 5
    .venv\\Scripts\\python.exe search.py "BLEU score" --method dense --model minilm --k 3
"""

import re
import sys

from loadmilvus import (
    DEFAULT_COLLECTION,
    connect,
    embed,
    get_dim,
    get_model,
    parse_model_arg,
)

# Final results returned, and how deep each technique retrieves before
# fusing/reranking. candidates > k so RRF and the reranker have room to reorder.
DEFAULT_K = 5
DEFAULT_CANDIDATES = 50

# Default cross-encoder reranker: small and CPU-friendly. Override with
# `--rerank-model <hf-id>` (e.g. BAAI/bge-reranker-v2-m3 for higher quality).
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RRF damping constant. The standard value from Cormack et al. (2009); larger
# flattens the contribution of top ranks, smaller sharpens it.
RRF_K = 60

# Weighted-fusion default: dense-vs-lexical weight (1.0 = all dense, 0 = all lexical).
DEFAULT_ALPHA = 0.5

# MMR default: relevance-vs-diversity trade-off (1.0 = pure relevance, 0 = pure diversity).
DEFAULT_LAMBDA = 0.5


def get_stored_dim(client, collection=DEFAULT_COLLECTION):
    """Return the dimension of the collection's stored `embedding` vectors.

    Dense/hybrid must embed the query with the SAME model that produced the
    stored vectors, so we read the stored dim to validate the chosen --model
    (a mismatch would make Milvus reject the search).
    """
    info = client.describe_collection(collection)
    for field in info["fields"]:
        if field["name"] == "embedding":
            return field["params"]["dim"]
    raise SystemExit("Collection has no 'embedding' field — was it built by loadmilvus?")


def dense_search(client, model, query, limit, ef, collection=DEFAULT_COLLECTION,
                 with_vectors=False):
    """Semantic search: embed the query, return the `limit` nearest chunks.

    Uses the HNSW index with COSINE, so `score` is cosine similarity (higher is
    closer). `ef` is HNSW's search-time candidate width (must be >= limit); a
    larger ef trades speed for recall. Set `with_vectors=True` to also return
    each chunk's stored `embedding` (used by MMR to measure result-to-result
    similarity).
    """
    client.load_collection(collection)
    query_vec = embed(model, [query])[0].tolist()
    fields = ["text", "source"] + (["embedding"] if with_vectors else [])
    hits = client.search(
        collection_name=collection,
        data=[query_vec],
        anns_field="embedding",
        limit=limit,
        output_fields=fields,
        search_params={"metric_type": "COSINE", "params": {"ef": ef}},
    )[0]
    records = []
    for h in hits:
        rec = {
            "id": h["id"],
            "text": h["entity"].get("text", ""),
            "source": h["entity"].get("source"),
            "score": h["distance"],
        }
        if with_vectors:
            rec["vector"] = h["entity"].get("embedding")
        records.append(rec)
    return records


def fetch_corpus(client, collection=DEFAULT_COLLECTION):
    """Return every stored row ({id, text, source}) for BM25 to rank locally.

    Lexical search runs client-side over the stored text (the collection has no
    sparse/BM25 field), so we pull the whole corpus once. Fine at POC scale.
    """
    client.load_collection(collection)
    rows = client.query(
        collection_name=collection,
        filter="id >= 0",
        output_fields=["text", "source"],
        limit=16384,
    )
    return [{"id": r["id"], "text": r["text"], "source": r.get("source")} for r in rows]


def tokenize(text):
    """Lowercase word/number tokens for BM25 (drops punctuation and symbols)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def lexical_search(corpus, query, limit):
    """BM25 keyword ranking over `corpus`, returning the top `limit` records.

    Scores by term frequency x inverse document frequency with length
    normalization — rewards rare query terms appearing in short chunks. `score`
    is the raw BM25 score (unbounded; comparable only within this result set).
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise SystemExit(
            "Lexical/hybrid search needs rank-bm25. Install it:\n"
            "    .venv\\Scripts\\python.exe -m pip install rank-bm25")

    bm25 = BM25Okapi([tokenize(row["text"]) for row in corpus])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(corpus, scores), key=lambda pair: pair[1], reverse=True)
    return [{**row, "score": float(score)} for row, score in ranked[:limit]]


def reciprocal_rank_fusion(ranked_lists, limit, rrf_k=RRF_K):
    """Fuse several ranked record lists into one by Reciprocal Rank Fusion.

    Each list contributes 1/(rrf_k + rank) per document (rank is 1-based within
    that list); scores are summed across lists by document id. RRF needs only
    the rank positions, not the (incomparable) raw dense/BM25 scores, which is
    why it fuses cleanly across different retrievers. Returns the top `limit`.
    """
    fused = {}
    for records in ranked_lists:
        for rank, rec in enumerate(records, start=1):
            slot = fused.setdefault(rec["id"], {"record": rec, "score": 0.0})
            slot["score"] += 1.0 / (rrf_k + rank)
    ordered = sorted(fused.values(), key=lambda s: s["score"], reverse=True)
    return [{**s["record"], "score": s["score"]} for s in ordered[:limit]]


def tfidf_search(corpus, query, limit):
    """TF-IDF cosine keyword ranking over `corpus`, returning the top `limit`.

    The classic lexical baseline: each chunk and the query become sparse
    term-frequency x inverse-document-frequency vectors, ranked by cosine.
    Simpler than BM25 (no TF saturation / length-normalization tuning); a useful
    keyword contrast. English stop-words are dropped. `score` is cosine in [0,1].
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [row["text"] for row in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)          # L2-normalized rows by default
    query_vec = vectorizer.transform([query])
    sims = (matrix @ query_vec.T).toarray().ravel()   # dot == cosine (both L2-normed)
    ranked = sorted(zip(corpus, sims), key=lambda pair: pair[1], reverse=True)
    return [{**row, "score": float(score)} for row, score in ranked[:limit]]


def _minmax(scores_by_id):
    """Min-max normalize a {id: score} map to [0, 1] (all-equal -> all 0.0).

    Dense (cosine) and BM25 scores live on different scales, so weighted fusion
    normalizes each list to a common [0, 1] range before combining.
    """
    if not scores_by_id:
        return {}
    values = scores_by_id.values()
    lo, hi = min(values), max(values)
    if hi == lo:
        return {i: 0.0 for i in scores_by_id}
    return {i: (s - lo) / (hi - lo) for i, s in scores_by_id.items()}


def weighted_fusion(dense, lexical, limit, alpha=DEFAULT_ALPHA):
    """Fuse dense + lexical by a weighted sum of min-max-normalized scores.

    combined = alpha * dense_norm + (1 - alpha) * lexical_norm, summed per
    document id over the union of both lists (a doc missing from one list scores
    0 there). Unlike RRF this uses the actual (normalized) scores, so `alpha`
    gives an explicit relevance dial. Returns the top `limit`.
    """
    dense_norm = _minmax({r["id"]: r["score"] for r in dense})
    lexical_norm = _minmax({r["id"]: r["score"] for r in lexical})
    records = {r["id"]: r for r in lexical + dense}   # dedup; same text either way
    fused = [
        {**records[i], "score": alpha * dense_norm.get(i, 0.0)
                                + (1 - alpha) * lexical_norm.get(i, 0.0)}
        for i in records
    ]
    fused.sort(key=lambda r: r["score"], reverse=True)
    return fused[:limit]


def mmr_search(client, model, query, limit, pool, ef, lambda_mult=DEFAULT_LAMBDA,
               collection=DEFAULT_COLLECTION):
    """Dense retrieval re-selected by Maximal Marginal Relevance for diversity.

    Pulls a `pool` of dense candidates (with their vectors), then greedily picks
    `limit` of them maximizing
        lambda * sim(query, doc) - (1 - lambda) * max sim(doc, already-picked)
    so each pick is relevant yet non-redundant. Vectors are L2-normalized, so
    cosine similarity is just a dot product. `score` reported is the query
    relevance (cosine). lambda=1 reduces to plain dense ranking.
    """
    import numpy as np

    candidates = dense_search(client, model, query, pool, ef, collection,
                              with_vectors=True)
    if not candidates:
        return []
    vectors = [np.asarray(c["vector"], dtype="float32") for c in candidates]
    query_vec = np.asarray(embed(model, [query])[0], dtype="float32")
    relevance = [float(query_vec @ v) for v in vectors]

    selected, remaining = [], list(range(len(candidates)))
    while remaining and len(selected) < limit:
        best_i, best_score = remaining[0], -float("inf")
        for i in remaining:
            redundancy = max((float(vectors[i] @ vectors[j]) for j in selected),
                             default=0.0)
            score = lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
    return [{**candidates[i], "score": relevance[i]} for i in selected]


def rerank(query, records, model_id, limit):
    """Re-score `records` with a cross-encoder and return the top `limit`.

    The cross-encoder reads (query, chunk) jointly, so `score` reflects true
    relevance rather than vector distance or term overlap. Expensive per pair,
    so it runs only on the already-retrieved shortlist.
    """
    from sentence_transformers import CrossEncoder

    print(f"Reranking {len(records)} candidates with '{model_id}'...")
    reranker = CrossEncoder(model_id)
    scores = reranker.predict([(query, rec["text"]) for rec in records])
    ranked = sorted(zip(records, scores), key=lambda pair: pair[1], reverse=True)
    return [{**rec, "score": float(score)} for rec, score in ranked[:limit]]


def search(client, query, method, model_name, k, candidates, ef,
           do_rerank, rerank_model, alpha=DEFAULT_ALPHA, lambda_mult=DEFAULT_LAMBDA):
    """Run one retrieval technique (+ optional rerank) and return top-k records.

    Dispatches on `method`; only loads an embedding model for the techniques
    that need one (lexical/tfidf are model-free). When reranking, the base method
    retrieves `candidates` first so the reranker has a shortlist to reorder.
    """
    depth = candidates if do_rerank else k

    def _embedder():
        """Load the query model, validating its dim against the stored vectors."""
        stored = get_stored_dim(client)
        model = get_model(model_name)
        if get_dim(model) != stored:
            raise SystemExit(
                f"Model '{model_name}' is {get_dim(model)}-dim but the stored "
                f"vectors are {stored}-dim. Query with the model they were built "
                f"with, or re-ingest with --model {model_name}.")
        return model

    if method == "dense":
        results = dense_search(client, _embedder(), query, depth, ef)
    elif method == "lexical":
        results = lexical_search(fetch_corpus(client), query, depth)
    elif method == "tfidf":
        results = tfidf_search(fetch_corpus(client), query, depth)
    elif method == "hybrid":
        dense = dense_search(client, _embedder(), query, candidates, ef)
        lexical = lexical_search(fetch_corpus(client), query, candidates)
        results = reciprocal_rank_fusion([dense, lexical], depth)
    elif method == "weighted":
        dense = dense_search(client, _embedder(), query, candidates, ef)
        lexical = lexical_search(fetch_corpus(client), query, candidates)
        results = weighted_fusion(dense, lexical, depth, alpha)
    elif method == "mmr":
        # MMR needs a pool larger than the output to have room to diversify.
        results = mmr_search(client, _embedder(), query, depth, candidates, ef,
                             lambda_mult)
    else:
        raise SystemExit(
            f"Unknown --method {method!r} "
            f"(use dense, lexical, tfidf, hybrid, weighted, or mmr).")

    if do_rerank:
        results = rerank(query, results, rerank_model, k)
    return results


def print_results(query, records, method, reranked, preview=200):
    """Print the ranked results with score, source, and a text preview."""
    label = f"{method}" + (" + rerank" if reranked else "")
    print(f"\nTop {len(records)} for {query!r}  (method: {label})\n" + "-" * 60)
    for rank, rec in enumerate(records, start=1):
        head = " ".join(rec["text"].split())[:preview]
        ellipsis = "..." if len(rec["text"]) > preview else ""
        print(f"\n[{rank}] score={rec['score']:.4f}  source={rec.get('source')}")
        print(f"    {head}{ellipsis}")


def parse_flag_value(argv, flag, default=None):
    """Return the value following `flag` in argv, or `default` if absent."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1]
    return default


def parse_query(argv):
    """Return the first positional (non-flag) argument as the search query."""
    flags_with_values = {"--method", "--model", "--rerank-model", "--k",
                         "--candidates", "--ef", "--alpha", "--lambda"}
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


def main():
    """Parse the query + options, run the chosen technique, print the results."""
    # Chunk text can hold non-cp1252 glyphs (e.g. the sqrt sign in "sqrt(dk)");
    # don't let printing results crash on a Windows console. (Same guard as
    # extractpdf.py / cluster.py.)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    query = parse_query(sys.argv)
    if not query:
        raise SystemExit(
            'Usage: search.py "your query" '
            '[--method dense|lexical|tfidf|hybrid|weighted|mmr] [--model <key>] '
            '[--rerank] [--rerank-model <id>] [--k N] [--candidates N] '
            '[--alpha F] [--lambda F]')

    method = parse_flag_value(sys.argv, "--method", "hybrid")
    model_name = parse_model_arg(sys.argv)          # --model / MILVUS_MODEL / default
    k = int(parse_flag_value(sys.argv, "--k", DEFAULT_K))
    candidates = int(parse_flag_value(sys.argv, "--candidates", DEFAULT_CANDIDATES))
    do_rerank = "--rerank" in sys.argv
    rerank_model = parse_flag_value(sys.argv, "--rerank-model", DEFAULT_RERANK_MODEL)
    # HNSW search width must be >= how many candidates we ask for.
    ef = int(parse_flag_value(sys.argv, "--ef", max(64, candidates)))
    alpha = float(parse_flag_value(sys.argv, "--alpha", DEFAULT_ALPHA))       # weighted
    lambda_mult = float(parse_flag_value(sys.argv, "--lambda", DEFAULT_LAMBDA))  # mmr

    client = connect()
    if not client.has_collection(DEFAULT_COLLECTION):
        raise SystemExit(
            f"Collection '{DEFAULT_COLLECTION}' does not exist. "
            f"Run extractpdf.py --store (or loadmilvus.py) first.")

    results = search(client, query, method, model_name, k, candidates, ef,
                     do_rerank, rerank_model, alpha, lambda_mult)
    print_results(query, results, method, do_rerank)


if __name__ == "__main__":
    main()
