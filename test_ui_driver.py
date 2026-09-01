"""Test driver: render one view by name (set MILVUSUI_VIEW). Used by test_ui.py."""
import importlib
import os

import streamlit as st

st.set_page_config(layout="wide")
importlib.import_module(f"milvusui.views.{os.environ['MILVUSUI_VIEW']}").render()
