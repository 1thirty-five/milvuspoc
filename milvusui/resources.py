"""Shared, cached handles on the expensive things: Milvus, models, collection state.

Streamlit re-runs the whole script on every widget interaction. Without caching
that would mean reconnecting to Milvus and reloading a multi-gigabyte embedding
model every time someone drags a slider, so everything costly lives here behind
`st.cache_resource` and is created once per process.

The flip side is staleness. Ingesting or re-clustering changes the collection
under caches that Streamlit has no way to know about -- including the
process-lifetime caches inside `hierarchicalsearch` and `search`, which the
README warns about explicitly. `invalidate()` is the one function that clears
all of them, and every write path in the UI must call it.
"""

import streamlit as st

import hierarchicalsearch
import loadmilvus
import search
from loadmilvus import DEFAULT_COLLECTION, DEFAULT_URI, MODEL_PRESETS

from .runner import call

# Bumped by invalidate(). Passed as an argument into the st.cache_data functions
# below so that changing it produces a cache miss -- Streamlit keys a cached
# value on its arguments, so a version counter is how you say "this depends on
# mutable state you cannot see".
#
# The parameter it arrives on MUST NOT be named with a leading underscore.
# st.cache_data deliberately excludes underscore-prefixed arguments from the
# hash key -- that is how you pass an unhashable handle like a db connection --
# so a `_version` parameter is invisible to the cache and every function below
# would serve its first result for the life of the server: ingesting or
# re-clustering would change nothing on screen until a restart.
_VERSION = 0


def data_version():
    """Current collection-state version; changes whenever the UI writes."""
    return _VERSION


def invalidate():
    """Forget everything derived from the collection's current contents.

    Call after ANY write: ingest, clustering, drop. Clears both the Streamlit
    caches in this module and the per-process caches inside the backend modules
    (cluster texts, BM25 indexes, `cluster_name` probes), which would otherwise
    keep serving pre-ingest data for the life of the server.
    """
    global _VERSION
    _VERSION += 1
    hierarchicalsearch.clear_caches()
    search._BM25_CACHE.clear()


@st.cache_resource(show_spinner=False)
def get_client(uri=DEFAULT_URI):
    """Connect to Milvus once per process.

    Not wrapped in `call()`: a failure here is the app's central precondition,
    and every view checks it through `connection_error()` rather than treating
    it as one operation going wrong.
    """
    return loadmilvus.connect(uri)


def connection_error(uri=DEFAULT_URI):
    """Return a message if Milvus is unreachable, else None.

    Milvus being down is the single most likely reason this app does nothing
    useful, and the raw pymilvus error is a wall of gRPC detail. Views call this
    first and render a short remedy instead.
    """
    try:
        client = get_client(uri)
        client.list_collections()
        return None
    except Exception as error:                # noqa: BLE001 - surfaced verbatim
        # A failed connection must not be cached as if it were a good one, or
        # starting Milvus wouldn't fix the page without a server restart.
        get_client.clear()
        return f"{type(error).__name__}: {error}"


@st.cache_resource(show_spinner=False)
def get_model(name):
    """Load an embedding model once per process (a few GB for bge-m3)."""
    model, _ = call(loadmilvus.get_model, name)
    return model


def collection_fingerprint(collection=DEFAULT_COLLECTION, uri=DEFAULT_URI):
    """A cheap signature of what is in the collection *right now*.

    Deliberately uncached, and deliberately not collection_info(): this is polled
    on a timer, so it has to cost a few round trips and nothing else, and it has
    to see past every cache in the app -- seeing past them is the entire point.

    Row count catches an ingest, a drop, a rebuild from input.md. It does not
    catch a re-clustering, which relabels the same rows and leaves the count
    where it was, so the sample row's scheme and whether it carries a `cluster`
    label ride along too. Both are uniform across rows once a write has
    finished, so which row Milvus happens to hand back does not matter -- and it
    must not, or the fingerprint would flap and invalidate on every poll.

    Every failure collapses to one stable value rather than raising: an
    unreachable Milvus is a state the sidebar renders, not an error, and a
    fingerprint that alternated between values while Milvus was down would clear
    the backend caches over and over for nothing.
    """
    try:
        client = get_client(uri)
        if not client.has_collection(collection):
            return ("absent",)
        rows = int(client.get_collection_stats(collection)["row_count"])
        if not rows:
            return ("empty",)
        try:
            sample = client.query(collection_name=collection, filter="id >= 0",
                                  output_fields=["cluster", "cluster_scheme"],
                                  limit=1)
        except Exception:                     # noqa: BLE001 - unloaded, or no such field
            sample = []
        row = sample[0] if sample else {}
        return (rows, row.get("cluster_scheme"), "cluster" in row)
    except Exception:                         # noqa: BLE001 - reported by connection_error
        return ("unreachable",)


@st.cache_data(show_spinner=False)
def collection_info(version, collection=DEFAULT_COLLECTION, uri=DEFAULT_URI):
    """Describe the collection: rows, dim, and which optional fields it carries.

    Returns a dict with `exists` False rather than raising, because "nothing
    ingested yet" is a normal state for this app to be in and every view needs
    to render a hint for it.
    """
    client = get_client(uri)
    if not client.has_collection(collection):
        return {"exists": False}

    described = client.describe_collection(collection)
    fields = {f["name"]: f for f in described["fields"]}
    dim = fields.get("embedding", {}).get("params", {}).get("dim")
    rows = client.get_collection_stats(collection)["row_count"]

    info = {
        "exists": True,
        "rows": int(rows),
        "dim": dim,
        "fields": sorted(fields),
        "named": False,
        "scheme": None,
        "clustered": False,
    }
    if not rows:
        return info

    client.load_collection(collection)
    info["named"] = hierarchicalsearch.has_cluster_names(client, collection)
    info["scheme"] = hierarchicalsearch.stored_scheme(client, collection)
    # A collection can hold vectors with no cluster labels at all (the state
    # right after an ingest), which is what disables hierarchical search and
    # the plot. Probe one row rather than reading the whole collection.
    try:
        sample = client.query(collection_name=collection, filter="id >= 0",
                              output_fields=["cluster"], limit=1)
        info["clustered"] = bool(sample) and sample[0].get("cluster") is not None
    except Exception:                         # noqa: BLE001 - absent dynamic field
        info["clustered"] = False
    return info


@st.cache_data(show_spinner=False)
def cluster_breakdown(version, collection=DEFAULT_COLLECTION, uri=DEFAULT_URI):
    """Return [(cluster_id, name, count)] sorted by id, or [] if unclustered.

    Reads every row's label (not its vector), so it is cheap in bytes but still
    a full scan -- hence cached against the data version rather than recomputed
    per rerun.
    """
    client = get_client(uri)
    info = collection_info(version, collection, uri)
    if not info.get("exists") or not info.get("clustered"):
        return []

    fields = ["cluster"] + (["cluster_name"] if info["named"] else [])
    iterator = client.query_iterator(
        collection_name=collection, filter="id >= 0",
        output_fields=fields, batch_size=loadmilvus.INSERT_BATCH)
    counts, names = {}, {}
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for row in batch:
                cid = row.get("cluster")
                if cid is None:
                    continue
                cid = int(cid)
                counts[cid] = counts.get(cid, 0) + 1
                names.setdefault(cid, row.get("cluster_name"))
    finally:
        iterator.close()

    return [(cid, names.get(cid) or ("unassigned"
            if cid == hierarchicalsearch.UNASSIGNED else str(cid)), counts[cid])
            for cid in sorted(counts)]


@st.cache_data(show_spinner=False)
def source_breakdown(version, collection=DEFAULT_COLLECTION, uri=DEFAULT_URI):
    """Return [(source filename, chunk count)] for the ingested documents."""
    client = get_client(uri)
    info = collection_info(version, collection, uri)
    if not info.get("exists") or not info.get("rows"):
        return []
    iterator = client.query_iterator(
        collection_name=collection, filter="id >= 0",
        output_fields=["source"], batch_size=loadmilvus.INSERT_BATCH)
    counts = {}
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            for row in batch:
                counts[row.get("source") or "(no source)"] = \
                    counts.get(row.get("source") or "(no source)", 0) + 1
    finally:
        iterator.close()
    return sorted(counts.items(), key=lambda pair: -pair[1])


# Dimensions of the presets, for choosing and validating a model without paying
# to load one. Anything not listed is unknown and simply isn't pre-checked.
PRESET_DIMS = {"bge-m3": 1024, "minilm": 384}


def model_names():
    """Preset keys, for a selectbox. Any HF id also works as a free-text entry."""
    return list(MODEL_PRESETS)


def default_model(stored_dim=None):
    """The preset to preselect: the one that can actually query this collection.

    MODEL_PRESETS is a plain dict and `minilm` happens to be declared first, so
    a selectbox left to its own devices defaults to a 384-dim model against a
    1024-dim collection and every vector method fails on the first click. Match
    the stored dimension where we can, and fall back to the pipeline's own
    default when there is nothing stored to match.
    """
    if stored_dim is not None:
        for name, dim in PRESET_DIMS.items():
            if dim == stored_dim and name in MODEL_PRESETS:
                return name
    return loadmilvus.DEFAULT_MODEL


def dim_mismatch(model_name, stored_dim):
    """Return a warning if `model_name` cannot query vectors of `stored_dim`.

    Checked before running anything, because the backend's own check fires only
    after the model has been loaded -- which for bge-m3 is a multi-second wait
    to be told the model was the wrong one.
    """
    if stored_dim is None:
        return None
    dim = PRESET_DIMS.get(model_name)
    if dim is not None and dim != stored_dim:
        return (f"'{model_name}' embeds to {dim} dimensions but the stored "
                f"vectors are {stored_dim}. Query with the model the collection "
                f"was ingested with, or re-ingest.")
    return None
