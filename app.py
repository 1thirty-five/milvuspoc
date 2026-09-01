"""Streamlit front-end for the Milvus retrieval pipeline.

Run (PowerShell):
    .venv\\Scripts\\streamlit.exe run app.py

Needs Milvus up (`docker compose up -d`). Everything else the app can do for
you: ingest documents, cluster them on a criterion you write, search with any of
the seven retrieval methods, and plot the result.

This file is only wiring — page registration and the global connection check.
The pages live in `milvusui/views/`, and `milvusui/runner.py` explains how the
CLI-shaped backend modules are adapted to a UI.
"""

import streamlit as st

st.set_page_config(page_title="Milvus retrieval", page_icon="🔎",
                   layout="wide", initial_sidebar_state="expanded")

from milvusui.resources import (collection_fingerprint, collection_info,          # noqa: E402
                                connection_error, data_version, invalidate)
from milvusui.views import (clustering, collection, compare, ingest, search,     # noqa: E402
                            visualize)

# `url_path` is given explicitly for every page: Streamlit infers it from the
# callable's name, and every view exposes the same `render`, so without this all
# six collide on one pathname and navigation refuses to build.
PAGES = [
    st.Page(search.render, title="Search", url_path="search",
            icon=":material/search:", default=True),
    st.Page(compare.render, title="Compare", url_path="compare",
            icon=":material/compare_arrows:"),
    st.Page(ingest.render, title="Ingest", url_path="ingest",
            icon=":material/upload_file:"),
    st.Page(clustering.render, title="Clustering", url_path="clustering",
            icon=":material/scatter_plot:"),
    st.Page(visualize.render, title="Visualise", url_path="visualise",
            icon=":material/bubble_chart:"),
    st.Page(collection.render, title="Collection", url_path="collection",
            icon=":material/database:"),
]


# How often the sidebar re-checks Milvus for changes made outside this app.
LIVE_POLL = "5s"


@st.fragment(run_every=LIVE_POLL)
def collection_summary():
    """The collection line, re-checked on a timer so it follows Milvus itself.

    Writes made *through* this app already refresh it, via invalidate(). But the
    collection is changed from a terminal just as often -- extractpdf.py,
    cluster.py, a docker restart -- and nothing tells the app that happened, so
    it looks instead: a fingerprint cheap enough to poll, and a refresh when that
    fingerprint moves.

    Only a real change invalidates. invalidate() also drops the BM25 and
    cluster-text caches that hierarchical search's speed depends on, so a poll
    that cleared them every five seconds would trade a stale number for a slow
    app. See resources.collection_fingerprint for what counts as a change.

    A change reruns the whole app rather than just this fragment, because every
    page renders from the same cached collection state: refreshing the sidebar
    alone would leave the page beside it describing the collection that was.
    """
    fingerprint = collection_fingerprint()
    if st.session_state.setdefault("live_fingerprint", fingerprint) != fingerprint:
        st.session_state["live_fingerprint"] = fingerprint
        invalidate()
        st.rerun(scope="app")

    info = collection_info(data_version())
    if info.get("exists"):
        st.caption(f"**{info['rows']:,}** chunks · **{info['dim']}**-dim"
                   + (f" · `{info['scheme']}`" if info.get("scheme") else "")
                   + ("" if info.get("clustered") else " · unclustered"))
    else:
        st.caption("No collection yet.")


def sidebar():
    """Connection state and collection summary, on every page."""
    with st.sidebar:
        st.markdown("### Milvus")
        error = connection_error()
        if error:
            st.error("Unreachable")
            with st.expander("Details"):
                st.code(error, language="text")
            st.markdown("```\ndocker compose up -d\n```")
            st.caption("Then wait for `healthy` in `docker compose ps`.")
            return False

        st.success("Connected")
        collection_summary()

        st.divider()
        # The poll above covers the collection's own contents. This is for the
        # rest: a criterion file edited on disk, a model swapped in the HF cache,
        # anything the fingerprint cannot see. It is also the way out if the poll
        # itself is ever wrong.
        if st.button("Refresh caches", width="stretch",
                     help=f"Re-read everything now. The collection is already "
                          f"re-checked every {LIVE_POLL}; this also clears the "
                          f"model and criterion caches."):
            invalidate()
            st.rerun()
        return True


def main():
    connected = sidebar()
    page = st.navigation(PAGES)
    if not connected and page.title not in ("Collection",):
        # Every other page needs Milvus for its first read, and would otherwise
        # render a wall of identical errors. The Collection page is the one that
        # explains how to fix it, so let that one through.
        st.title(page.title)
        st.error("Milvus is not reachable — start it and this page will load.")
        st.markdown("```\ndocker compose up -d\n```")
        return
    page.run()


main()
