"""Streamlit front-end for the Milvus retrieval pipeline.

Layered so each piece has one job:

    runner      adapts the CLI-shaped backends (print / SystemExit) to a caller
    resources   cached handles on Milvus, models and collection state
    components  widgets shared across views
    views/      one module per page, each exposing render()

Views may import from runner, resources and components. Nothing imports a view.
"""
