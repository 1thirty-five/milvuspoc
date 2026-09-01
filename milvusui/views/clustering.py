"""Clustering: criterion-free KMeans, or your own named criterion from criteria.md.

Both paths rebuild the collection to attach labels, so only one clustering lives
in Milvus at a time -- the same constraint the CLI has. The criterion editor is
the point of this page: `customcluster` is driven entirely by a markdown file,
and iterating on that file is the actual work, so it is editable here with the
run button next to it rather than in another window.
"""

import streamlit as st

import cluster as kmeans
import customcluster
import loadmilvus
from pathlib import Path

from ..components import guard, log_panel, model_picker, require
from ..resources import (cluster_breakdown, collection_info, data_version,
                         invalidate)
from ..runner import call_logged


def _show_breakdown():
    """Table of the clustering currently stored."""
    breakdown = cluster_breakdown(data_version())
    if not breakdown:
        return
    st.subheader("Stored clustering")
    total = sum(count for _, _, count in breakdown)
    st.dataframe(
        [{"id": cid, "name": name, "chunks": count,
          "share": 100 * count / total} for cid, name, count in breakdown],
        hide_index=True, width="stretch",
        # ProgressColumn formats the raw value, it does not scale it, so a
        # share held as 0..1 under a "%%" format rendered 30.2%% as "0.3%".
        # The share is stored as a percentage and the range matches it.
        column_config={"share": st.column_config.ProgressColumn(
            format="%.1f%%", min_value=0, max_value=100)})


def _kmeans_tab(info):
    st.caption("Groups by whatever the embedding model thinks dominates. The "
               "only knob is `k`; the clusters get numbers, not names.")

    k = st.number_input("Clusters (k)", 2, max(2, info.get("rows", 2)),
                        min(kmeans.DEFAULT_K, max(2, info.get("rows", 2))),
                        key="km_k")
    sigma = st.slider(
        "Unassigned floor (sigma)", 0.0, 5.0, kmeans.DEFAULT_FLOOR_SIGMA, 0.1,
        key="km_sigma",
        help="Rows further than this many standard deviations below the mean "
             "centroid similarity go to `unassigned` (−1). 0 disables it.")

    if not st.button("Run KMeans", type="primary"):
        return

    def _run():
        client = loadmilvus.connect()
        rows = kmeans.fetch_all(client)
        if not rows:
            raise SystemExit("No vectors stored. Ingest something first.")
        if int(k) > len(rows):
            raise SystemExit(f"Cannot make {k} clusters from {len(rows)} vectors.")
        labels = kmeans.cluster_vectors([r["embedding"] for r in rows],
                                        k=int(k), sigma=float(sigma))
        kmeans.print_clusters(rows, labels, int(k))
        kmeans.store_labels(client, rows, labels)
        kmeans.write_result_labels(rows, labels)
        return len(rows)

    with st.spinner("Clustering…"):
        count, log, error = call_logged(_run)
    invalidate()
    if error:
        st.error(error)
    elif count:
        st.success(f"Labelled {count:,} chunks into {k} clusters.")
    log_panel(log, "Clustering log", expanded=True)


def _criteria_editor(path):
    """Edit and save the criterion file in place."""
    file = Path(path)
    if not file.exists():
        st.warning(f"`{path}` does not exist.")
        if st.button("Create a starter file"):
            file.write_text(
                "# Mode\nanchor\n\n# Labels\n"
                "- pricing: fees, discounts, billing terms\n"
                "- legal risk: liability, indemnity, compliance exposure\n\n"
                "# Settings\nassign = seeded\nfloor  = auto\nmodel  = bge-m3\n",
                encoding="utf-8")
            st.rerun()
        return False

    # The key carries the path. A Streamlit widget ignores its `value` argument
    # once its key exists in session state, so a single fixed key would keep
    # showing the first file's text after you point the box at another file --
    # and "Save criterion" would then write those stale contents over the new
    # path, silently destroying it. Keying on the path makes each file its own
    # widget, so switching loads the real contents and Save writes them back.
    text = st.text_area(str(file), file.read_text(encoding="utf-8"),
                        height=340, key=f"cc_text::{file}",
                        help="Each label is embedded as `name: description`. "
                             "The description carries almost all of the signal.")
    if st.button("Save criterion"):
        file.write_text(text, encoding="utf-8")
        st.success(f"Wrote {path}.")
    return True


def _custom_tab(info):
    st.caption("Say what to cluster on, in words. A criterion stated in words "
               "embeds into the same space as the documents, so it becomes "
               "geometry — no training, no LLM.")

    path = st.text_input("Criterion file", customcluster.DEFAULT_CRITERIA,
                         key="cc_path")
    if not _criteria_editor(path):
        return

    parsed, parse_log, parse_error = call_logged(customcluster.parse_criteria, path)
    if parse_error:
        st.error(parse_error)
        log_panel(parse_log)
        return

    columns = st.columns(4)
    columns[0].metric("Labels", len(parsed["labels"]))
    columns[1].metric("Assign", parsed["assign"])
    columns[2].metric("Floor", str(parsed["floor"]))
    columns[3].metric("Scheme", parsed["scheme"])

    warning = None
    if parsed["model"] and info.get("dim"):
        known = {"bge-m3": 1024, "minilm": 384}.get(parsed["model"])
        if known and known != info["dim"]:
            warning = (f"`model = {parsed['model']}` embeds to {known} dims but "
                       f"the stored vectors are {info['dim']}. The labels would "
                       f"land in a different space and the assignment would be "
                       f"meaningless. Fix `model` in {path}.")
    if warning:
        st.error(warning)

    dry_run = st.checkbox("Dry run (print the groups, write nothing)",
                          key="cc_dry")
    if not st.button("Run clustering", type="primary", disabled=bool(warning)):
        return

    def _run():
        config = customcluster.parse_criteria(path)
        customcluster.describe(config, path)
        client = loadmilvus.connect()
        rows = kmeans.fetch_all(client)
        if not rows:
            raise SystemExit("No vectors stored. Ingest something first.")
        embeddings = [row["embedding"] for row in rows]
        if config["k"] > len(rows):
            raise SystemExit(
                f"Cannot make {config['k']} clusters from {len(rows)} vectors.")

        model = loadmilvus.get_model(config["model"])
        stored_dim, model_dim = len(embeddings[0]), loadmilvus.get_dim(model)
        if stored_dim != model_dim:
            raise SystemExit(
                f"Model '{config['model']}' embeds to {model_dim} dims but the "
                f"stored vectors are {stored_dim}. Set `model` in {path}.")

        anchors = customcluster.embed_anchors(model, config["labels"])
        labels, similarity = customcluster.assign_anchors(
            embeddings, anchors, config["assign"], config["floor"])
        names = [name for name, _ in config["labels"]]
        stats = customcluster.anchor_stats(labels, similarity, config["k"])
        customcluster.print_clusters(rows, labels, names, config["preview"], stats)

        if dry_run:
            print("\nDry run: nothing written to Milvus or result.py.")
            return 0

        row_names = [customcluster.UNASSIGNED_NAME
                     if int(label) == customcluster.UNASSIGNED
                     else names[int(label)] for label in labels]
        kmeans.store_labels(client, rows, labels, names=row_names,
                            scheme=config["scheme"])
        kmeans.write_result_labels(rows, labels)
        return len(rows)

    with st.spinner("Embedding the criterion and assigning…"):
        count, log, error = call_logged(_run)
    if not dry_run:
        invalidate()
    if error:
        st.error(error)
    elif dry_run:
        st.info("Dry run — nothing was written.")
    elif count:
        st.success(f"Labelled {count:,} chunks under scheme "
                   f"'{parsed['scheme']}'.")
    log_panel(log, "Clustering log", expanded=True)


def render():
    st.title("Clustering")
    info = collection_info(data_version())
    if not require(info.get("exists") and info.get("rows"),
                   "Nothing ingested yet. Use the **Ingest** page first."):
        return

    st.caption("Both paths rebuild the collection, so one clustering lives in "
               "Milvus at a time and each run replaces the last.")

    custom, plain = st.tabs(["Custom criterion", "KMeans"])
    with custom:
        with guard("clustering"):
            _custom_tab(info)
    with plain:
        with guard("clustering"):
            _kmeans_tab(info)

    st.divider()
    _show_breakdown()
