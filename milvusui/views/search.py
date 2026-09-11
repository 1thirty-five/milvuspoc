"""Retrieval: every method in search.py, plus the full hierarchical flag set.

Two dispatch paths, on purpose. Six of the seven methods go through
`search.search`, which is the module's own front door. `hierarchical` goes
straight to `hierarchicalsearch.hierarchical_search` instead, because
`search.search` calls it with five positional arguments and so cannot reach
`--rank-by`, `--floor`, `--hybrid`, `--adaptive`, `--sort` or
`--include-unassigned` -- the flags the README spends most of its hierarchical
section on. Calling the module directly also returns the cluster leaderboard,
which is the most informative thing this technique produces and is otherwise
thrown away.
"""

import time

import streamlit as st

import hierarchicalsearch as hier
import search as searchlib

from ..components import guard, log_panel, model_picker, require, results_list
from ..resources import collection_info, data_version, get_client, get_model
from ..runner import call

METHODS = ["hybrid", "dense", "lexical", "tfidf", "weighted", "mmr", "hierarchical"]

BLURB = {
    "dense": "Embeds the query, ANN search over HNSW. Meaning and paraphrase; weak on exact tokens.",
    "lexical": "BM25 over stored text. No model. Exact terms; blind to paraphrase.",
    "tfidf": "TF-IDF cosine — the classic keyword baseline.",
    "hybrid": "Dense + lexical fused by Reciprocal Rank Fusion. The strong default.",
    "weighted": "Weighted sum of min-max-normalised dense and lexical scores. Explicit dial.",
    "mmr": "Dense, then Maximal Marginal Relevance — avoids near-duplicate results.",
    "hierarchical": "Ranks clusters first, then takes a decreasing quota from each.",
}

# What each method actually computes, behind the ⓘ next to the blurb. The blurb
# says what a method is *for*; this says what it does, which is the part that
# explains why two of them disagree on the same query. Each entry is the scoring
# rule, the constants it is fixed at, and the knobs that move it -- kept short
# enough to read standing up, because a popover nobody finishes is a caption.
FORMULA = {
    "dense": """
**Dense — cosine ANN over HNSW**

```
score(d) = cos(embed(q), embed(d))
```

One approximate-nearest-neighbour sweep of the vector index at width `ef`.
Vectors are L2-normalised, so the cosine is just a dot product. Score is a
similarity in [-1, 1] and is directly comparable across queries.

*Knobs:* HNSW ef, candidate depth.
""",
    "lexical": """
**Lexical — BM25 (Okapi)**

```
score(d) = Σ  IDF(t) · tf·(k₁+1) / (tf + k₁·(1 − b + b·len(d)/avglen))
          t∈q
```

Fixed at `k₁ = 1.5`, `b = 0.75`. Term frequency saturates, so a word repeated
ten times is not ten times the evidence, and long chunks are penalised. No
model: both sides are lowercased into `[a-z0-9]+` tokens. The score is
unbounded and only means anything *within* one result set.

*Knobs:* none.
""",
    "tfidf": """
**TF-IDF — cosine over sparse term vectors**

```
score(d) = cos(tfidf(q), tfidf(d))     tfidf(t) = tf(t) · log(N / df(t))
```

The classic keyword baseline: no TF saturation and no length term, which is
exactly what makes it a useful contrast against BM25. English stop-words are
dropped. Rows are L2-normalised, so the score is a cosine in [0, 1].

*Knobs:* none.
""",
    "hybrid": """
**Hybrid — weighted Reciprocal Rank Fusion**

```
score(d) = Σ  wᵢ / (60 + rankᵢ(d))        w = (alpha, 1 − alpha)
          i∈{dense, lexical}
```

Runs both retrievers, then fuses on **rank position only** — the raw cosine
and BM25 scores never meet, which is what lets two incomparable scales be
combined at all. `60` is the standard RRF damping constant. Weights are
rescaled to average 1, so alpha = 0.5 is exactly the parameter-free RRF.

*Knobs:* alpha (Weights tab), candidate depth.
""",
    "weighted": """
**Weighted — sum of normalised scores**

```
score(d) = alpha · norm(dense(d)) + (1 − alpha) · norm(lexical(d))
```

`norm` is min-max to [0, 1] over each list separately, because a cosine and a
BM25 score are otherwise not addable. A document missing from one list scores
0 on that side rather than being dropped. Unlike hybrid, this dials the
**scores** themselves, so the same alpha gives a different ranking.

*Knobs:* alpha (Weights tab), candidate depth.
""",
    "mmr": """
**MMR — Maximal Marginal Relevance**

```
pick argmax [ lambda · cos(q,d) − (1 − lambda) · max cos(d,s) ]
       d                                        s∈picked
```

Retrieves a dense pool, then fills the k slots greedily: each pick is the
candidate with the best trade-off between matching the query and *not*
resembling what is already chosen. lambda = 1 collapses to plain dense.
The reported score is the query relevance, not the trade-off value.

*Knobs:* lambda (Weights tab), candidate depth (the pool to diversify over).
""",
    "hierarchical": """
**Hierarchical — rank clusters, then take a quota**

```
stage 1   score(c) = mean of c's best 3 hits in one probe sweep
stage 2   take allocation[i] chunks from the i-th ranked cluster
```

One flat sweep `probe` deep scores every cluster by the best material it
actually holds — not by its centroid, which averages a large mixed cluster
toward the corpus mean. The default allocation 5,4,3,2,1 then spreads 15
results over the 5 best regions instead of letting one region take them all.

*Knobs:* allocation, rank-by, floor, adaptive/sharpness (Hierarchical options).
""",
}

# Methods that embed the query and so need a model matching the stored vectors.
NEEDS_MODEL = {"dense", "hybrid", "weighted", "mmr", "hierarchical"}

# Methods that blend a dense signal with a lexical one, and so answer to alpha.
# They differ in where the dial sits -- `hybrid` weights the RRF rank
# contributions, `weighted` weights the min-max-normalised scores -- but it is
# the same number meaning the same thing, so it is the same widget.
DENSE_VS_LEXICAL = {"hybrid", "weighted"}


def _floor_control(key):
    """Build the `--floor` string from a friendlier pair of widgets."""
    mode = st.selectbox(
        "Score floor", ["off", "auto", "auto:M", "absolute cosine"], key=f"{key}_mode",
        help="Makes each quota a cap rather than a mandate: a cluster with "
             "nothing good enough leaves the seat empty instead of padding it.")
    if mode == "off":
        return "off"
    if mode == "auto":
        return "auto"
    if mode == "auto:M":
        multiple = st.slider("Pool multiple (M)", 1.05, 3.0,
                             hier.FLOOR_POOL_MULTIPLE, 0.05, key=f"{key}_mult",
                             help="Cutoff is the (M × budget)-th best chunk in "
                                  "the probe pool. Lower bites harder.")
        return f"auto:{multiple}"
    return str(st.slider("Cosine cutoff", 0.0, 1.0, 0.4, 0.01, key=f"{key}_abs"))


def _hierarchical_options(key):
    """Every flag hierarchicalsearch.py's CLI accepts."""
    options = {}
    left, right = st.columns(2)
    with left:
        options["allocation_text"] = st.text_input(
            "Allocation", ",".join(map(str, hier.DEFAULT_ALLOCATION)), key=f"{key}_alloc",
            help="Chunks taken from the 1st, 2nd, … ranked cluster. Length = how "
                 "many clusters are searched.")
        options["rank_by"] = st.selectbox(
            "Rank clusters by", ["probe", "centroid", "name"], key=f"{key}_rankby",
            help="probe = one flat sweep, score by mean of best 3 hits. "
                 "centroid = mean of member vectors. name = embed the label.")
        options["sort"] = st.selectbox("Result order", ["cluster", "score"],
                                       key=f"{key}_sort",
                                       help="cluster keeps the funnel order.")
        options["probe"] = st.number_input(
            "Probe depth", 10, 5000, hier.DEFAULT_PROBE, 10, key=f"{key}_probe",
            help="`--rank-by probe` only. Depth of the flat sweep.")
    with right:
        options["floor"] = _floor_control(key)
        options["adaptive"] = st.checkbox(
            "Adaptive allocation", key=f"{key}_adaptive",
            help="Derive the funnel's shape from the cluster scores; the "
                 "allocation then sets only the total and the cluster count.")
        options["sharpness"] = st.slider(
            "Sharpness", 0.0, 4.0, hier.DEFAULT_SHARPNESS, 0.1, key=f"{key}_sharp",
            disabled=not options["adaptive"],
            help="Adaptive only. How hard score gaps are amplified. 0 = even split.")
        options["hybrid"] = st.checkbox(
            "BM25 inside each cluster", key=f"{key}_hyb",
            help="Adds a lexical pass per selected cluster, RRF-fused with the "
                 "dense one. For queries hinging on an exact token.")

    extra_left, extra_right = st.columns(2)
    options["include_unassigned"] = extra_left.checkbox(
        "Include `unassigned`", key=f"{key}_unassigned",
        help="Let the leftovers bucket (cluster −1) compete for a slot.")
    options["reuse_pool"] = not extra_right.checkbox(
        "Disable probe-pool reuse", key=f"{key}_nopool",
        help="Diagnostic. Forces one filtered search per cluster instead of "
             "slicing the pool. Measured to return identical results.")
    options["criteria"] = st.text_input(
        "Criteria file", hier.DEFAULT_CRITERIA, key=f"{key}_criteria",
        help="`--rank-by name` only: where the label descriptions are read from.")
    return options


def _current_allocation():
    """The allocation currently typed into the hierarchical options, as a tuple.

    Read from session state rather than from the widget, because `k` is rendered
    above the allocation field and so needs its value one pass early. Falls back
    to the default while the widget does not exist yet (first render) or holds
    something unparseable (mid-edit, e.g. a trailing comma) -- this only decides
    a default, so it must never be the thing that raises.
    """
    text = st.session_state.get("hier_alloc",
                                ",".join(map(str, hier.DEFAULT_ALLOCATION)))
    try:
        quotas = tuple(int(part) for part in str(text).replace(" ", "").split(",")
                       if part)
    except ValueError:
        return tuple(hier.DEFAULT_ALLOCATION)
    return quotas or tuple(hier.DEFAULT_ALLOCATION)


def _sticky_slider(label, key, default, help_text):
    """A 0..1 slider that keeps its value while its method is unselected.

    Streamlit drops a keyed widget's session state on any run that does not
    render it, so a slider drawn only for the method that owns it would reset to
    the default every time you looked at another method and came back. The value
    is mirrored under a second, non-widget key that nothing cleans up, and that
    mirror is what seeds the widget the next time it is drawn.
    """
    remembered = st.session_state.get(f"{key}_kept", default)
    value = st.slider(label, 0.0, 1.0, remembered, 0.05, key=key, help=help_text)
    st.session_state[f"{key}_kept"] = value
    return value


def _weights(method, params):
    """The one weight the chosen method dials, or a note that it has none.

    Both sliders used to sit in the options grid with the irrelevant one
    disabled, which put two greyed-out controls in front of the five methods
    that have no weight at all. On its own tab it can show only the live one, so
    the tab always answers the same question: what is there to tune here?

    The unrendered parameter still has to reach search.search, which takes both
    positionally, so it falls back to the module default. Each is read only by
    the method that owns it, so a default nobody uses cannot change a result.
    """
    params["alpha"] = searchlib.DEFAULT_ALPHA
    params["lambda_mult"] = searchlib.DEFAULT_LAMBDA

    if method in DENSE_VS_LEXICAL:
        params["alpha"] = _sticky_slider(
            "alpha — dense vs lexical", "alpha", searchlib.DEFAULT_ALPHA,
            "Weight on the dense side. 1.0 = all dense, 0 = all lexical.")
        split = (f"**{params['alpha']:.0%}** dense · "
                 f"**{1 - params['alpha']:.0%}** lexical")
        # Same dial, two different things being dialled -- worth saying, because
        # the same alpha on the same query does not give the same ranking under
        # the two methods, and that looks like a bug if you do not know why.
        st.caption(
            f"{split}, applied to each side's rank contribution "
            f"(`1/({searchlib.RRF_K} + rank)`). The dial moves how much a "
            f"retriever's *ordering* counts; the raw cosine and BM25 scores "
            f"never enter the sum. 50/50 is the classic parameter-free RRF."
            if method == "hybrid" else
            f"{split}, each min-max normalised before the sum so the two "
            f"scales are comparable. Here the dial moves the *scores* "
            f"themselves, not the rank positions.")
    elif method == "mmr":
        params["lambda_mult"] = _sticky_slider(
            "lambda — relevance vs diversity", "lam", searchlib.DEFAULT_LAMBDA,
            "1.0 = pure relevance, 0 = pure diversity.")
        st.caption(f"**{params['lambda_mult']:.0%}** relevance to the query · "
                   f"**{1 - params['lambda_mult']:.0%}** penalty for looking "
                   f"like a chunk already picked.")
    else:
        # dense, lexical and tfidf each rank on exactly one signal, so there are
        # no two sides to balance -- a slider here would be a control that
        # cannot change the result.
        st.caption(f"`{method}` ranks on a single signal, so there is nothing "
                   f"to balance. `hybrid` and `weighted` blend dense with "
                   f"lexical; `mmr` blends relevance against diversity.")


def _run(client, query, method, model_name, params):
    """Execute one method. Returns (records, leaderboard, log)."""
    if method != "hierarchical":
        # The model MUST be passed in. search.search otherwise calls
        # loadmilvus.get_model itself, which re-reads the weights from the HF
        # cache on every single search -- ~10s per query for bge-m3, since
        # st.cache_resource wraps our get_model and not the backend's.
        model = get_model(model_name) if method in NEEDS_MODEL else None
        records, log = call(
            searchlib.search, client, query, method, model_name,
            params["k"], params["candidates"], params["ef"],
            params["rerank"], params["rerank_model"],
            params["alpha"], params["lambda_mult"], model=model)
        return records, None, log

    model = get_model(model_name)
    allocation, _ = call(hier.parse_allocation, params["allocation_text"])
    (records, ranked, profiles, quotas), log = call(
        hier.hierarchical_search, client, model, query,
        allocation=allocation, ef=params["ef"], rank_by=params["rank_by"],
        include_unassigned=params["include_unassigned"], sort=params["sort"],
        criteria=params["criteria"], adaptive=params["adaptive"],
        sharpness=params["sharpness"], probe=int(params["probe"]),
        floor=params["floor"], hybrid=params["hybrid"],
        reuse_pool=params["reuse_pool"])

    if params["rerank"]:
        records, rerank_log = call(searchlib.rerank, query, records,
                                   params["rerank_model"], params["k"])
        log += rerank_log
    elif params["k"] < len(records):
        records = records[:params["k"]]

    leaderboard = [
        {"rank": i, "cluster": cid, "name": profiles[cid]["name"],
         "score": score, "quota": quotas[i - 1] if i <= len(quotas) else 0,
         "in pool": profiles[cid].get("hits")}
        for i, (cid, score) in enumerate(ranked, start=1)]
    return records, leaderboard, log


def render():
    st.title("Search")
    info = collection_info(data_version())
    if not require(info.get("exists") and info.get("rows"),
                   "Nothing ingested yet. Use the **Ingest** page first."):
        return

    query = st.text_input("Query", key="q",
                          placeholder="how is risk handled?")

    method = st.segmented_control("Method", METHODS, default="hybrid",
                                  key="method") or "hybrid"

    # The blurb reads as a sentence; the formula does not, and putting it inline
    # would make the line above the query box the densest thing on the page. A
    # tertiary popover is borderless, so it sits next to the caption as an icon
    # rather than as a button competing with Search.
    # Icon first, then the sentence. Streamlit's grid cannot size a column to
    # its text, so a trailing icon lands at the far right edge with a stretch of
    # nothing between it and the caption it belongs to; leading it keeps the two
    # together whatever the blurb's length.
    info_column, blurb_column = st.columns([1, 24], gap="small",
                                           vertical_alignment="center")
    with info_column.popover(":material/info:", type="tertiary",
                             help=f"What `{method}` computes"):
        st.markdown(FORMULA[method])
    blurb_column.caption(BLURB[method])

    if method == "hierarchical" and not info.get("clustered"):
        st.warning("Hierarchical search ranks clusters, and this collection has "
                   "no `cluster` labels. Run a clustering first.")

    # Hierarchical's result count is the sum of its per-cluster quotas, so a
    # default k of 5 would silently discard two thirds of the funnel -- the same
    # trap hierarchicalsearch.py's CLI calls out. The widget also gets a
    # per-family key, so switching methods picks up that family's default
    # instead of inheriting a sticky 5 from the flat methods.
    hierarchical = method == "hierarchical"
    quotas = _current_allocation() if hierarchical else ()
    default_k = sum(quotas) if hierarchical else searchlib.DEFAULT_K

    # `k` sits out here rather than in the options expander below: it is the one
    # retrieval setting people reach for on almost every query, and an expander
    # that defaults to collapsed hides it entirely.
    params = {}
    k_column, _ = st.columns([1, 3])
    params["k"] = k_column.number_input(
        "Results (k)", 1, 500, max(1, default_k),
        key=f"k_{'hier' if hierarchical else 'flat'}",
        help=("Defaults to the allocation's total, so the funnel isn't "
              "truncated. Lower it only to trim a reranked shortlist."
              if hierarchical else "Results returned."))
    if hierarchical:
        st.caption(f"The allocation below totals **{default_k}** chunks across "
                   f"**{len(quotas)}** cluster(s). A lower k trims the funnel.")

    with st.expander("Retrieval options", expanded=False):
        # Two tabs rather than one grid. Depth, model and reranker apply to every
        # method; the weights apply to one method each, so mixing them put
        # controls that do nothing next to controls that always do something.
        retrieval_tab, weights_tab = st.tabs(["Retrieval", "Weights"])

        with retrieval_tab:
            left, right = st.columns(2)
            params["candidates"] = left.number_input(
                "Candidate depth", 1, 2000, searchlib.DEFAULT_CANDIDATES,
                key="cand", help="Shortlist retrieved before fusing/reranking.")
            # The key carries the candidate depth. A widget ignores its `value`
            # argument once its key is in session state, so a fixed key pinned ef
            # to the 64 computed on the first render: raising the depth to 500
            # left the HNSW sweep at 64 and the extra 436 candidates were never
            # really searched for. Varying the key makes the default track the
            # depth, while still letting you override it at any given depth.
            params["ef"] = right.number_input(
                "HNSW ef", 1, 4000, max(64, int(params["candidates"])),
                key=f"ef_{int(params['candidates'])}",
                help="Search width. Higher = better recall, slower. Tracks the "
                     "candidate depth, because ef below it caps recall at ef.")

            model_name = (model_picker(info, "search")
                          if method in NEEDS_MODEL
                          else searchlib.parse_model_arg([]))
            if method not in NEEDS_MODEL:
                st.caption(f"`{method}` needs no embedding model.")

            params["rerank"] = st.checkbox(
                "Cross-encoder rerank", key="rerank",
                help="Re-scores the shortlist by reading query and chunk "
                     "together. More accurate, one model pass per candidate.")
            params["rerank_model"] = st.text_input(
                "Reranker", searchlib.DEFAULT_RERANK_MODEL, key="rrmodel",
                disabled=not params["rerank"])

        with weights_tab:
            _weights(method, params)

    if method == "hierarchical":
        with st.expander("Hierarchical options", expanded=True):
            params.update(_hierarchical_options("hier"))

    if st.button("Search", type="primary", disabled=not query.strip()):
        with guard("the search"), st.spinner("Searching…"):
            started = time.perf_counter()
            records, leaderboard, log = _run(
                get_client(), query.strip(), method, model_name, params)
            st.session_state["results"] = {
                "records": records, "leaderboard": leaderboard, "log": log,
                "query": query.strip(), "method": method,
                "elapsed": time.perf_counter() - started,
                "reranked": params["rerank"],
            }

    _render_results()


def _render_results():
    """Render the last search from session state.

    Kept separate from running it so the display controls below -- preview
    length, highlighting -- re-render instantly instead of re-running a search
    that may have just cost a cross-encoder pass.
    """
    state = st.session_state.get("results")
    if not state:
        return

    st.divider()
    label = state["method"] + (" + rerank" if state["reranked"] else "")
    st.caption(f"**{len(state['records'])}** results for *{state['query']}* "
               f"· `{label}` · {state['elapsed']*1000:.0f} ms")

    if state["leaderboard"]:
        with st.expander("Cluster ranking (stage 1)", expanded=True):
            st.dataframe(state["leaderboard"], hide_index=True,
                         width="stretch",
                         column_config={"score": st.column_config.NumberColumn(
                             format="%.4f")})

    left, right = st.columns([3, 1])
    truncate = left.checkbox("Truncate chunk text", key="trunc")
    preview = (left.slider("Preview characters", 80, 1200, 320, 20, key="prev")
               if truncate else None)
    show_highlight = right.checkbox("Highlight terms", True, key="hl")

    log_panel(state["log"])
    results_list(state["records"], state["query"], preview, show_highlight)
