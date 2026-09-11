"""UMAP projection of the stored vectors, rendered inline.

visualize.py writes an HTML file and opens a browser tab, which is the right
shape for a CLI and the wrong one here. This reuses its fetch/projection/legend
functions and hands the figure to Streamlit instead, so the plot lives on the
page next to the clustering that produced it.

The projection is the expensive part -- UMAP on a few thousand 1024-dim vectors
is seconds, not milliseconds -- so it is cached against the data version and the
projection parameters, and only recomputed when one of those actually changes.
"""

import numpy as np
import plotly.express as px
import streamlit as st
import umap

import visualize as viz

from ..components import guard, log_panel, require
from ..resources import collection_info, data_version, get_client
from ..runner import call


@st.cache_data(show_spinner=False)
def _project(version, n_neighbors, min_dist, metric):
    """Fetch every row and project it to 2D. Cached: this is the slow step.

    `version` is spelled without a leading underscore on purpose -- see the note
    in resources.py. Underscore-prefixed arguments are left out of the cache key,
    which would pin this plot to whatever was stored the first time it rendered.
    """
    rows, log = call(viz.fetch_labeled, get_client())
    matrix = np.asarray([row["embedding"] for row in rows], dtype="float32")
    reducer = umap.UMAP(
        n_components=2,
        # n_neighbors must stay below the sample count or UMAP raises; a corpus
        # can be smaller than the default 15.
        n_neighbors=min(int(n_neighbors), max(2, len(matrix) - 1)),
        min_dist=float(min_dist),
        metric=metric,
        random_state=42,
    )
    coords = reducer.fit_transform(matrix)
    # The embeddings are the bulk of the payload and nothing downstream needs
    # them; dropping them keeps the cached entry small.
    slim = [{k: v for k, v in row.items() if k != "embedding"} for row in rows]
    return slim, coords, log


def _sources(rows):
    """Each row's source document, with a stand-in for rows that carry none."""
    return [row.get("source") or "(no source)" for row in rows]


def _composition(rows, labels, order):
    """Which documents feed each cluster: [{cluster, document, chunks, share}].

    A cluster is a region of embedding space, not a region of a file, so on a
    multi-document corpus its members routinely come from several -- and whether
    a cluster is one document's private topic or a theme running across the
    whole corpus is the thing the scatter cannot show, because colour is already
    spent on the cluster.
    """
    counts = {}
    for label, source in zip(labels, _sources(rows)):
        counts[(label, source)] = counts.get((label, source), 0) + 1

    totals = {}
    for (label, _), n in counts.items():
        totals[label] = totals.get(label, 0) + n

    rank = {label: i for i, label in enumerate(order)}
    return [{"cluster": label, "document": source, "chunks": n,
             "share of cluster": 100 * n / totals[label]}
            for (label, source), n in sorted(
                counts.items(), key=lambda kv: (rank.get(kv[0][0], 0), -kv[1]))]


def _figure(rows, coords, point_size, opacity):
    labels, order, named = viz.legend_labels(rows)
    scheme = rows[0].get("cluster_scheme") if named else None
    sources = _sources(rows)
    documents = sorted(set(sources))

    # Colour is already carrying the cluster, so the document rides on the
    # marker *shape* -- the one channel left that reads at this density. Only
    # worth spending when there is more than one document: a symbol legend with
    # a single entry is a key to nothing.
    many = len(documents) > 1
    figure = px.scatter(
        x=coords[:, 0], y=coords[:, 1], color=labels,
        symbol=sources if many else None,
        hover_name=[viz.wrap(row["text"]) for row in rows],
        # The tooltip answers it regardless of how many documents there are,
        # since that is where you look once a point is worth asking about.
        hover_data={"document": sources,
                    "chunk": [row.get("seq") if row.get("seq") is not None
                              else -1 for row in rows]},
        category_orders={"color": order, "symbol": documents},
        labels={"color": "cluster", "symbol": "document",
                "x": "UMAP-1", "y": "UMAP-2"},
        title=(f"{len(rows)} chunks"
               + (f" · {len(documents)} documents" if many else
                  f" · {documents[0]}" if documents else "")
               + (f" · criterion '{scheme}'" if scheme else "")),
    )
    figure.update_traces(marker=dict(size=point_size, opacity=opacity,
                                     line=dict(width=0)))
    figure.update_layout(legend_title_text="cluster / document", height=720,
                         margin=dict(l=10, r=10, t=48, b=10))
    return figure


def render():
    st.title("Visualise")
    info = collection_info(data_version())
    if not require(info.get("exists") and info.get("rows"),
                   "Nothing ingested yet. Use the **Ingest** page first."):
        return
    if not require(info.get("clustered"),
                   "Rows have no `cluster` label — the plot would be one colour. "
                   "Run a clustering first."):
        return

    st.caption("UMAP projection with the cosine metric, matching the space the "
               "vectors were embedded and indexed in. Hover a point for its "
               "text, the document it came from and its position in that file.")

    left, middle, right = st.columns(3)
    n_neighbors = left.slider(
        "Neighbours", 2, 100, 15, key="v_nn",
        help="Low values emphasise local structure, high values global shape.")
    min_dist = middle.slider(
        "Minimum distance", 0.0, 0.99, 0.1, 0.01, key="v_md",
        help="How tightly points may pack together.")
    metric = right.selectbox("Metric", ["cosine", "euclidean", "correlation"],
                             key="v_metric")

    size_col, opacity_col = st.columns(2)
    # Defaults follow visualize.build_plot: small translucent points once the
    # corpus is big enough that outlined markers merge into a blob.
    big = info["rows"] > 5000
    point_size = size_col.slider("Point size", 1, 20, 3 if big else 9, key="v_size")
    opacity = opacity_col.slider("Opacity", 0.1, 1.0, 0.5 if big else 0.85, 0.05,
                                 key="v_op")

    with guard("the projection"):
        with st.spinner("Projecting with UMAP…"):
            rows, coords, log = _project(data_version(), n_neighbors, min_dist,
                                         metric)
        st.plotly_chart(_figure(rows, coords, point_size, opacity),
                        width="stretch")

        labels, order, _ = viz.legend_labels(rows)
        with st.expander("Which documents feed each cluster", expanded=False):
            st.dataframe(
                _composition(rows, labels, order), hide_index=True,
                width="stretch",
                # See views/clustering.py: ProgressColumn formats the value it
                # is given without scaling it, so the share is a percentage and
                # the range says so.
                column_config={"share of cluster": st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0, max_value=100)})

        log_panel(log, "Projection log")
