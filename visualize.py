"""
Visualize the Milvus clusters as an interactive 2D scatter plot.

Pulls every row (text + embedding + cluster label) from the Milvus `documents`
collection, projects the high-dim vectors down to 2D with UMAP (cosine metric,
to match the embedding space), and writes an interactive Plotly HTML where each
point is colored by its cluster and hovers to show the document text.

Run `python cluster.py` first so the rows carry a `cluster` label.

Prerequisites:
    pip install umap-learn plotly

Run:
    python visualize.py                 # -> clusters.html
    python visualize.py --out my.html   # choose the output path
"""

import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.express as px
import umap

from loadmilvus import DEFAULT_COLLECTION, INSERT_BATCH, connect

DEFAULT_OUT = "clusters.html"


def fetch_labeled(client, collection=DEFAULT_COLLECTION):
    """Return every row's text, embedding, and cluster label from `collection`.

    Paged, not a single capped query: a plain `query` tops out at 16384 rows, so
    on a book-sized corpus it would quietly plot a fraction of the data and look
    perfectly plausible doing it.
    """
    if not client.has_collection(collection):
        raise SystemExit(
            f"Collection '{collection}' does not exist. Run loadmilvus.py first.")
    client.load_collection(collection)

    iterator = client.query_iterator(
        collection_name=collection,
        filter="id >= 0",
        output_fields=["text", "embedding", "cluster"],
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

    print(f"Fetched {len(rows)} rows from Milvus collection '{collection}'.")
    return rows


def project_2d(embeddings):
    """Reduce high-dim embeddings to 2D with UMAP (cosine metric).

    n_neighbors is kept below the sample count; cosine matches how the vectors
    were embedded/indexed, so the layout reflects semantic similarity.
    """
    matrix = np.asarray(embeddings, dtype="float32")
    n = len(matrix)
    print(f"Projecting {n} vectors ({matrix.shape[1]}-dim) to 2D with UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(matrix)


def wrap(text, width=60):
    """Insert <br> every ~width chars so long docs wrap nicely in the tooltip."""
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    out.append(line)
    return "<br>".join(out)


def build_plot(rows, coords, out=DEFAULT_OUT):
    """Write an interactive Plotly scatter of the 2D points, colored by cluster."""
    clusters = [str(row.get("cluster")) for row in rows]  # str -> discrete colors
    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=clusters,
        hover_name=[wrap(row["text"]) for row in rows],
        category_orders={"color": sorted(set(clusters), key=lambda c: int(c))},
        labels={"color": "cluster", "x": "UMAP-1", "y": "UMAP-2"},
        title=f"Milvus document clusters ({len(rows)} docs, UMAP 2D projection)",
    )
    # Marker size scales down with the corpus: size 10 + a white outline reads
    # well for the ~100 docs of input.md, but at tens of thousands of chunks the
    # outlines merge into an opaque blob and the structure disappears. Small,
    # semi-transparent points let dense regions show through as density instead.
    if len(rows) > 5000:
        fig.update_traces(marker=dict(size=3, opacity=0.5, line=dict(width=0)))
    else:
        fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="white")))
    fig.update_layout(legend_title_text="cluster")

    out_path = Path(out)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Wrote interactive plot to {out_path.resolve()}")
    return out_path


def parse_out_arg(argv, default=DEFAULT_OUT):
    """Read the output path from `--out <path>` in argv, else `default`."""
    if "--out" in argv:
        i = argv.index("--out")
        if i + 1 >= len(argv):
            raise SystemExit("--out needs a value, e.g. --out clusters.html")
        return argv[i + 1]
    return default


def main():
    out = parse_out_arg(sys.argv)

    client = connect()
    rows = fetch_labeled(client)
    if not rows:
        raise SystemExit("No rows found in Milvus. Run loadmilvus.py + cluster.py first.")
    if rows[0].get("cluster") is None:
        raise SystemExit("Rows have no 'cluster' label. Run `python cluster.py` first.")

    coords = project_2d([row["embedding"] for row in rows])
    out_path = build_plot(rows, coords, out)

    # Pop it open in the default browser for convenience.
    webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
