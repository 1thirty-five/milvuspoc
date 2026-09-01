"""What is actually in Milvus: schema, contents, provenance, and a browser.

The pipeline has enough replace-the-collection behaviour -- clustering drops
`source`, loadmilvus rebuilds wholesale, switching models re-embeds everything --
that "what state is the collection in right now" is a real question with a
non-obvious answer. This page answers it in one place.
"""

import streamlit as st

import loadmilvus
from loadmilvus import DEFAULT_COLLECTION

from ..components import guard, log_panel, require, status_ribbon
from ..resources import (cluster_breakdown, collection_info, connection_error,
                         data_version, get_client, invalidate, source_breakdown)
from ..runner import call


def _schema_table(client):
    described = client.describe_collection(DEFAULT_COLLECTION)
    rows = []
    for field in described["fields"]:
        rows.append({
            "field": field["name"],
            "type": str(field.get("type", "")).split(".")[-1],
            "primary": bool(field.get("is_primary")),
            "params": ", ".join(f"{k}={v}" for k, v in
                                (field.get("params") or {}).items()),
        })
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Dynamic fields (`source`, `seq`, `cluster`, `cluster_name`, "
               "`cluster_scheme`) do not appear in the schema — they ride in the "
               "dynamic field and exist only on rows that carry them.")

    try:
        indexes = client.list_indexes(DEFAULT_COLLECTION)
        for name in indexes:
            detail = client.describe_index(DEFAULT_COLLECTION, name)
            st.caption(f"Index `{name}`: " +
                       ", ".join(f"{k}={v}" for k, v in detail.items()))
    except Exception:                          # noqa: BLE001 - informational only
        pass


def _browser(client, info):
    """Page through stored rows, optionally filtered."""
    st.caption("Milvus has no implicit row order; rows are sorted here by "
               "`(source, seq)`, which is reading order for anything ingested "
               "by extractpdf.")

    filter_expr = st.text_input(
        "Filter expression", "id >= 0", key="br_filter",
        help='A Milvus boolean expression, e.g. `cluster == 3` or '
             '`cluster_name == "pricing"`.')
    limit = st.slider("Rows", 5, 200, 25, key="br_limit")

    if not st.button("Fetch"):
        return

    fields = ["text", "source", "seq"]
    if info.get("clustered"):
        fields.append("cluster")
    if info.get("named"):
        fields.append("cluster_name")

    def _query():
        client.load_collection(DEFAULT_COLLECTION)
        return client.query(collection_name=DEFAULT_COLLECTION,
                            filter=filter_expr or "id >= 0",
                            output_fields=fields, limit=int(limit))

    try:
        rows, _ = call(_query)
    except Exception as error:                 # noqa: BLE001 - user's expression
        # A mistyped filter is the user's, not a bug in the app, so it gets the
        # message rather than guard()'s traceback.
        st.error("Milvus rejected that query — check the filter expression.")
        st.code(str(error), language="text")
        return
    if not rows:
        st.info("No rows matched.")
        return
    # `seq` rides in the dynamic field, so a row ingested by a path that does not
    # set it comes back without one (or with None); int(None) would take the
    # whole page down over a missing sort key.
    rows.sort(key=lambda r: (str(r.get("source") or ""),
                             int(r["seq"]) if r.get("seq") is not None else -1))
    st.dataframe(
        [{**{f: r.get(f) for f in fields if f != "text"},
          "chars": len(r.get("text", "")),
          "text": " ".join(str(r.get("text", "")).split())} for r in rows],
        hide_index=True, width="stretch")


def _benchmark(info):
    """Run benchmark.py and show statistics.md.

    Destructive, and not obviously so from its name: it resets the collection as
    it measures the insert path. That is fine from the CLI where `loadmilvus.py
    --full` re-stores afterwards, but from a UI it would silently delete an
    ingest, so it is gated behind the same typed confirmation as the drop.
    """
    from pathlib import Path

    import benchmark
    from milvusui.components import model_picker

    st.warning("**This resets the collection.** It measures the storing "
               "pipeline, which means rebuilding and re-inserting into it. "
               "Everything currently stored is destroyed; re-ingest afterwards.")
    model_name = model_picker(info, "bench")
    confirm = st.text_input("Type the collection name to enable",
                            key="bench_confirm", placeholder=DEFAULT_COLLECTION)
    if st.button("Run benchmark", disabled=confirm != DEFAULT_COLLECTION):
        with st.spinner("Benchmarking (this resets and refills the collection)…"):
            _, log = call(benchmark.main, model_name)
        invalidate()
        st.success("Wrote statistics.md.")
        log_panel(log, "Benchmark log", expanded=True)

    statistics = Path("statistics.md")
    if statistics.exists():
        with st.expander("statistics.md", expanded=False):
            st.markdown(statistics.read_text(encoding="utf-8"))


def _danger_zone(client, info):
    st.caption("Dropping the collection deletes every stored vector. The "
               "embedding cache lives in the collection, so the next ingest "
               "re-embeds from scratch.")
    confirm = st.text_input(
        "Type the collection name to enable", key="dz_confirm",
        placeholder=DEFAULT_COLLECTION)
    if st.button("Drop collection", type="primary",
                 disabled=confirm != DEFAULT_COLLECTION):
        with guard("dropping the collection"):
            client.drop_collection(DEFAULT_COLLECTION)
            invalidate()
            st.success(f"Dropped '{DEFAULT_COLLECTION}'.")
            st.rerun()


def render():
    st.title("Collection")

    error = connection_error()
    if error:
        st.error("Cannot reach Milvus.")
        st.code(error, language="text")
        st.markdown(
            "Start it with `docker compose up -d`, then wait for `healthy` in "
            "`docker compose ps`. On Windows the data must live in the "
            "`milvus_data` **named volume** — a bind-mount makes the embedded "
            "etcd time out and Milvus panic on boot.")
        return

    info = collection_info(data_version())
    clusters = cluster_breakdown(data_version())
    status_ribbon(info, clusters)

    if not require(info.get("exists"),
                   "No collection yet. Use the **Ingest** page to create one."):
        return

    client = get_client()
    overview, schema, browse, bench, danger = st.tabs(
        ["Overview", "Schema", "Browse", "Benchmark", "Danger zone"])

    with overview:
        with guard("reading the collection"):
            sources = source_breakdown(data_version())
            if sources:
                st.subheader("Sources")
                total = sum(count for _, count in sources)
                st.dataframe(
                    [{"source": name, "chunks": count,
                      "share": 100 * count / total}
                     for name, count in sources],
                    hide_index=True, width="stretch",
                    # See views/clustering.py: ProgressColumn formats the value
                    # it is given without scaling it, so the share is a
                    # percentage and the range says so.
                    column_config={"share": st.column_config.ProgressColumn(
                        format="%.1f%%", min_value=0, max_value=100)})
            if clusters:
                st.subheader("Clusters")
                ctotal = sum(count for _, _, count in clusters)
                st.dataframe(
                    [{"id": cid, "name": name, "chunks": count,
                      "share": 100 * count / ctotal}
                     for cid, name, count in clusters],
                    hide_index=True, width="stretch",
                    column_config={"share": st.column_config.ProgressColumn(
                        format="%.1f%%", min_value=0, max_value=100)})
            else:
                st.info("No cluster labels stored. Hierarchical search and the "
                        "plot both need them.")

    with schema:
        with guard("describing the collection"):
            _schema_table(client)

    with browse:
        with guard("browsing"):
            _browser(client, info)

    with bench:
        _benchmark(info)

    with danger:
        _danger_zone(client, info)
