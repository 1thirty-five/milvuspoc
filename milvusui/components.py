"""Widgets shared across views: error boundary, result cards, status ribbon."""

import contextlib
import html
import re
import traceback

import streamlit as st

from search import tokenize

from .runner import BackendError


@contextlib.contextmanager
def guard(what="this operation"):
    """Render any failure inside the page instead of blanking it.

    Two failure kinds, deliberately shown differently. A BackendError is the
    backend refusing a request it understood -- "run cluster.py first", "that
    model is the wrong dimension" -- and is the user's to fix, so it gets the
    message and the log it printed on the way out. Anything else is a bug in
    this app or the pipeline, so it gets a traceback, because hiding that just
    makes the report harder to act on.
    """
    try:
        yield
    except BackendError as error:
        st.error(str(error))
        log_panel(error.log, "Output before it stopped", expanded=bool(error.log))
    except Exception:                          # noqa: BLE001 - deliberate boundary
        st.error(f"Unexpected error while running {what}.")
        with st.expander("Traceback", expanded=False):
            st.code(traceback.format_exc(), language="text")


def log_panel(log, label="Backend output", expanded=False):
    """Show captured stdout, if there was any.

    The backends print things a UI cannot otherwise show: the cluster
    leaderboard, cache hit counts, the floor cutoff, "5 cluster(s) served from
    the probe pool". Keeping it one expander away means the UI never has to
    reimplement that reporting to stay honest about what happened.
    """
    if not (log or "").strip():
        return
    with st.expander(label, expanded=expanded):
        st.code(log.strip(), language="text")


def highlight(text, query, enable=True):
    """HTML-escape `text` and <mark> the query's terms inside it.

    Escaping first is not optional: chunks are arbitrary text out of a PDF and
    routinely contain characters that would otherwise be parsed as markup.
    """
    escaped = html.escape(text)
    if not enable or not query:
        return escaped
    terms = {t for t in tokenize(query) if len(t) > 2}
    if not terms:
        return escaped
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in
                                           sorted(terms, key=len, reverse=True))
                         + r")\b", re.IGNORECASE)
    return pattern.sub(r"<mark>\1</mark>", escaped)


def result_card(rank, record, query="", preview=None, show_highlight=True):
    """Render one retrieval result: its rank, provenance, scores and text."""
    text = " ".join(str(record.get("text", "")).split())
    truncated = preview is not None and len(text) > preview
    if preview is not None:
        text = text[:preview]

    bits = [f"**{rank}.**", f"`score {record['score']:.4f}`"]
    # Under --hybrid the score becomes an RRF rank score, so the cosine is
    # carried separately and is worth showing beside it -- they mean different
    # things and only one of them is a similarity.
    cosine = record.get("cosine")
    if cosine is not None and abs(cosine - record["score"]) > 1e-9:
        bits.append(f"`cos {cosine:.4f}`")
    if "cluster_rank" in record:
        bits.append(f"`#{record['cluster_rank']} "
                    f"{record.get('cluster_name') or record.get('cluster')}`")
    source = record.get("source")
    if source:
        seq = record.get("seq")
        where = f"{source}" + (f" · chunk {seq}"
                               if seq is not None and int(seq) >= 0 else "")
        bits.append(f"`{where}`")
    bits.append(f"<span style='opacity:.55'>{len(record.get('text',''))} chars</span>")

    with st.container(border=True):
        st.markdown(" ".join(bits), unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.92rem;line-height:1.5'>"
            f"{highlight(text, query, show_highlight)}"
            f"{'…' if truncated else ''}</div>",
            unsafe_allow_html=True)


def results_list(records, query="", preview=None, show_highlight=True):
    """Render a ranked result list, or a hint when it came back empty."""
    if not records:
        st.info("No results. Try a different method, a deeper `--candidates`, "
                "or a lower floor.")
        return
    for i, record in enumerate(records, start=1):
        result_card(i, record, query, preview, show_highlight)


def status_ribbon(info, breakdown=None):
    """A one-line summary of what is currently in Milvus."""
    if not info.get("exists"):
        st.warning("No `documents` collection yet — ingest something first.")
        return
    columns = st.columns(4)
    columns[0].metric("Chunks", f"{info['rows']:,}")
    columns[1].metric("Dimensions", info["dim"] or "—")
    columns[2].metric("Clusters", len(breakdown) if breakdown else
                      ("yes" if info["clustered"] else "none"))
    columns[3].metric("Scheme", info.get("scheme") or "—")


def require(condition, message):
    """Render `message` and return False when `condition` fails.

    Lets a view state its preconditions as a flat guard clause rather than
    nesting the whole page inside an if.
    """
    if condition:
        return True
    st.info(message)
    return False


def model_picker(info, key, label="Embedding model"):
    """Pick a preset or type a raw Hugging Face id, with a dimension warning."""
    from .resources import default_model, dim_mismatch, model_names

    names = model_names()
    preferred = default_model(info.get("dim"))
    choice = st.selectbox(label, names + ["(other…)"], key=f"{key}_preset",
                          index=names.index(preferred) if preferred in names else 0,
                          help="Must match the model the collection was "
                               "ingested with for any vector method.")
    name = (st.text_input("Hugging Face model id", key=f"{key}_custom",
                          placeholder="BAAI/bge-m3")
            if choice == "(other…)" else choice)
    warning = dim_mismatch(name, info.get("dim")) if name else None
    if warning:
        st.warning(warning)
    return name
