"""Run several retrieval methods on one query and diff what they return.

The README's case for having seven methods is that they disagree in
characteristic ways -- dense finds paraphrase, lexical finds exact tokens,
hierarchical spreads across regions. That argument is only checkable side by
side, and doing it from the CLI means running the same query five times and
eyeballing five scrollbacks. This does it in one pass and quantifies the
overlap.
"""

import time

import streamlit as st

import hierarchicalsearch as hier
import search as searchlib

from ..components import guard, model_picker, require, result_card
from ..resources import (collection_info, data_version, default_model,
                         get_client, get_model)
from ..runner import BackendError, call

COMPARABLE = ["hybrid", "dense", "lexical", "tfidf", "weighted", "mmr",
              "hierarchical"]
NEEDS_MODEL = {"dense", "hybrid", "weighted", "mmr", "hierarchical"}


def _run_one(client, query, method, model_name, k, candidates, ef, rerank,
             rerank_model):
    """Run one method, returning a record dict for the comparison table."""
    started = time.perf_counter()
    if method == "hierarchical":
        model = get_model(model_name)
        records, _ = call(hier.hierarchical_search, client, model, query,
                          ef=ef)
        records = records[0] if isinstance(records, tuple) else records
        if rerank:
            # Every other method reranks inside search.search. Without this the
            # "Rerank all" checkbox left exactly one column unreranked, so the
            # overlap matrix compared a reranked ordering against a raw one and
            # read as disagreement between the methods rather than between the
            # scorers.
            records, _ = call(searchlib.rerank, query, records, rerank_model, k)
    else:
        # model= is not optional in practice: without it every method in the
        # comparison reloads the weights from disk. See views/search.py.
        records, _ = call(searchlib.search, client, query, method, model_name,
                          k, candidates, ef, rerank, rerank_model,
                          searchlib.DEFAULT_ALPHA, searchlib.DEFAULT_LAMBDA,
                          model=get_model(model_name)
                          if method in NEEDS_MODEL else None)
    return {"records": records[:k], "elapsed": time.perf_counter() - started}


def _overlap(a, b):
    """Jaccard overlap of two result id lists."""
    sa, sb = {r["id"] for r in a}, {r["id"] for r in b}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def render():
    st.title("Compare methods")
    st.caption("One query, several retrievers, and how much they actually agree.")

    info = collection_info(data_version())
    if not require(info.get("exists") and info.get("rows"),
                   "Nothing ingested yet. Use the **Ingest** page first."):
        return

    query = st.text_input("Query", key="cq",
                          placeholder="where are the subsidiaries")
    chosen = st.multiselect("Methods", COMPARABLE,
                            default=["hybrid", "dense", "lexical"], key="cmethods")

    left, right = st.columns(2)
    k = left.number_input("Results each", 1, 50, 5, key="ck")
    candidates = right.number_input("Candidate depth", 1, 2000,
                                    searchlib.DEFAULT_CANDIDATES, key="ccand")
    rerank = left.checkbox("Rerank all", key="crerank")
    rerank_model = right.text_input("Reranker", searchlib.DEFAULT_RERANK_MODEL,
                                    key="crrm", disabled=not rerank)
    # The fallback is the preset matching the stored dimension, not a hardcoded
    # name: nothing selected needs a model in that branch, but a wrong name here
    # becomes a wrong default the moment a vector method is ticked.
    model_name = (model_picker(info, "compare")
                  if any(m in NEEDS_MODEL for m in chosen)
                  else default_model(info.get("dim")))

    if "hierarchical" in chosen and not info.get("clustered"):
        st.warning("`hierarchical` needs cluster labels; it will fail until you "
                   "run a clustering.")

    if st.button("Compare", type="primary",
                 disabled=not (query.strip() and chosen)):
        outcomes = {}
        with st.spinner("Running…"):
            for method in chosen:
                # One method failing (hierarchical with no labels, lexical with
                # rank-bm25 missing) must not lose the others' results.
                try:
                    outcomes[method] = _run_one(
                        get_client(), query.strip(), method, model_name, int(k),
                        int(candidates), max(64, int(candidates)), rerank,
                        rerank_model)
                except BackendError as error:
                    outcomes[method] = {"error": str(error)}
                except Exception as error:     # noqa: BLE001 - reported per method
                    outcomes[method] = {"error": f"{type(error).__name__}: {error}"}
        st.session_state["comparison"] = {"outcomes": outcomes,
                                          "query": query.strip()}

    state = st.session_state.get("comparison")
    if not state:
        return

    st.divider()
    outcomes, cquery = state["outcomes"], state["query"]
    good = {m: o for m, o in outcomes.items() if "error" not in o}

    for method, outcome in outcomes.items():
        if "error" in outcome:
            st.error(f"**{method}** — {outcome['error']}")

    if len(good) > 1:
        st.subheader("Agreement")
        st.caption("Jaccard overlap of the returned chunk sets. Low numbers are "
                   "the point: they are why more than one method exists.")
        names = list(good)
        matrix = [{"method": a, **{b: round(_overlap(good[a]["records"],
                                                     good[b]["records"]), 2)
                                   for b in names}} for a in names]
        st.dataframe(matrix, hide_index=True, width="stretch")

        every = [{r["id"] for r in o["records"]} for o in good.values()]
        shared = set.intersection(*every) if every else set()
        st.caption(f"{len(shared)} chunk(s) returned by *all* {len(good)} methods.")

    if good:
        st.subheader("Results")
        for column, (method, outcome) in zip(st.columns(len(good)), good.items()):
            with column:
                st.markdown(f"**{method}** · {outcome['elapsed']*1000:.0f} ms")
                with guard(f"rendering {method}"):
                    for i, record in enumerate(outcome["records"], start=1):
                        result_card(i, record, cquery, preview=220)
