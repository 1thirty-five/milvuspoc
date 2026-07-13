"""
Embed text with a sentence transformer and STORE it in Milvus.

Scope: storing only (no search/fetch yet). Each function does one thing and
returns plain objects, so a UI layer can call them independently:

    from loadmilvus import get_model, connect, ensure_collection, store

    model  = get_model()
    client = connect()
    ensure_collection(client, dim=get_dim(model))
    store(client, model, ["some text", "more text"])

Prerequisites (run once in your terminal):
    # 1. Start Milvus standalone via Docker
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
    bash standalone_embed.sh start

    # 2. Install Python deps
    pip install pymilvus sentence-transformers

Then edit input.md with the text to index, and run:
    python loadmilvus.py          # preview each vector, embed + store
    python loadmilvus.py --full    # also write full vectors to result.py and
                                   # regenerate statistics.md (no terminal dump)
"""

import hashlib
import os
import sys
from pathlib import Path

from pymilvus import MilvusClient, DataType

# sentence_transformers (and the torch it drags in) costs ~6s to import, so it is
# imported inside get_model instead of here. Callers that only touch Milvus — a
# cache probe, a query, a delete — then pay nothing for a model they never load.

# Defaults — a UI can override any of these per call.
DEFAULT_URI = "http://localhost:19530"
DEFAULT_COLLECTION = "documents"
DEFAULT_INPUT = "input.md"
DEFAULT_RESULT = "result.py"

# Named model presets — pick one per run with `--model <key>` (see main()).
# Each entry needs an "id" (the Hugging Face model id); these optional keys let a
# preset carry model-specific needs without touching the pipeline:
#   "trust_remote_code": True   -> passed to SentenceTransformer() at load
#   "doc_prefix": "..."         -> prepended to every document before embedding
#                                  (e.g. nomic-embed needs "search_document: ")
# A `--model` value that isn't a key here is treated as a literal HF id with no
# special handling, so you can try any model without editing this dict.
MODEL_PRESETS = {
    "minilm": {"id": "sentence-transformers/all-MiniLM-L6-v2"},
    "bge-m3": {"id": "BAAI/bge-m3"},
}

# Default preset key for a plain run. bge-m3 (1024-dim) is the default embedding
# model; pass `--model minilm` for the lighter 384-dim model (faster cold start).
DEFAULT_MODEL = "bge-m3"


def resolve_model(name=DEFAULT_MODEL):
    """Resolve a preset key (or raw HF id) to (hf_id, load_kwargs, doc_prefix).

    If `name` is a key in MODEL_PRESETS, its config is used; otherwise `name` is
    treated as a literal Hugging Face model id with no special handling.
    """
    preset = MODEL_PRESETS.get(name)
    if preset is None:
        return name, {}, ""
    load_kwargs = {}
    if preset.get("trust_remote_code"):
        load_kwargs["trust_remote_code"] = True
    return preset["id"], load_kwargs, preset.get("doc_prefix", "")


def load_inputs(path=DEFAULT_INPUT):
    """Parse an input.md file into (documents, query).

    Format — two markdown sections keyed by their headings:

        # Documents
        - first thing to index
        - second thing to index

        # Query
        what are you searching for?

    Under "Documents", every non-empty line is one document (a leading `-`,
    `*`, or `1.` bullet marker is stripped). Under "Query", the first non-empty
    line is used as the search query. Both headings are optional.
    """
    documents, query = [], None
    section = None
    in_comment = False

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Skip HTML comment blocks (<!-- ... -->) so notes in the file
        # aren't indexed as documents.
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if line.startswith("<!--"):
            if "-->" not in line:
                in_comment = True
            continue
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
            section = "query" if heading.startswith("quer") else "documents"
            continue
        # Strip a leading bullet / numbered-list marker.
        for marker in ("- ", "* ", "+ "):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break
        else:
            parts = line.split(". ", 1)
            if len(parts) == 2 and parts[0].isdigit():
                line = parts[1].strip()

        if section == "query":
            if query is None:
                query = line
        else:
            documents.append(line)

    return documents, query


def get_model(name=DEFAULT_MODEL):
    """Load and return the sentence-transformer embedding model.

    `name` is a MODEL_PRESETS key (e.g. "bge-m3") or a literal HF model id. Any
    document prefix the preset requires is stashed on the returned model as
    `model.doc_prefix` so embed() can apply it transparently.

    The weights are loaded straight from the local HF cache when they're already
    there. Left to itself, SentenceTransformer round-trips to the Hugging Face
    Hub to re-check file revisions on every single load, which costs ~20s and
    buys nothing once a model is cached (bge-m3: 24s -> 2.4s). Only a genuine
    cache miss is allowed to hit the network, so a first-ever run still downloads.
    """
    from sentence_transformers import SentenceTransformer   # see note at imports

    hf_id, load_kwargs, doc_prefix = resolve_model(name)
    label = f"'{name}' ({hf_id})" if name != hf_id else f"'{hf_id}'"
    print(f"Loading embedding model {label}...")
    try:
        model = SentenceTransformer(hf_id, local_files_only=True, **load_kwargs)
    except Exception:
        # Not in the cache (or only partly): fall back to a normal networked load,
        # which downloads it. Every subsequent run then takes the fast path above.
        print(f"  {hf_id} not in the local cache: downloading (first run only)...")
        model = SentenceTransformer(hf_id, **load_kwargs)
    model.doc_prefix = doc_prefix
    model.hf_id = hf_id          # cache keys hash the resolved id, not the alias
    print(f"Model loaded on {model.device}. Embedding dimension = {get_dim(model)}")
    return model


def get_dim(model):
    """Embedding dimension for a model, across sentence-transformers versions."""
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def connect(uri=DEFAULT_URI):
    """Connect to Milvus and return the client."""
    client = MilvusClient(uri=uri)
    print(f"Connected to Milvus at {uri}.")
    return client


def content_hash(model_name, text):
    """Cache key for one chunk's embedding: a hash of (model, text).

    The model name is part of the key because the same text embeds to a
    different vector under a different model — switching models must miss the
    cache rather than silently reuse vectors from the old one.
    """
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def schema_is_current(client, collection, dim):
    """True if an existing `collection` matches what this pipeline now expects.

    Guards against reusing a collection built before `chunk_hash` existed, or one
    whose vectors came from a model of a different dimension. Either way its rows
    are unusable and it has to be rebuilt.
    """
    fields = {f["name"]: f for f in client.describe_collection(collection)["fields"]}
    if "chunk_hash" not in fields:
        return False
    return fields["embedding"].get("params", {}).get("dim") == dim


def ensure_collection(client, dim, collection=DEFAULT_COLLECTION, reset=False):
    """Make sure `collection` exists with the right schema and index.

    Args:
        client: a connected MilvusClient.
        dim: embedding dimension the vector field should hold.
        collection: collection name.
        reset: if True, drop and recreate for a clean slate.
    """
    if client.has_collection(collection):
        if not reset and schema_is_current(client, collection, dim):
            return collection
        client.drop_collection(collection)

    # Schema: auto ID, the source text, its vector, and the cache key that ties
    # the two together (see content_hash / fetch_cached).
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=2048)
    schema.add_field("chunk_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

    # HNSW: graph-based ANN index, low-latency high-recall lookups for an
    # in-memory collection. COSINE matches the L2-normalized sentence embeddings
    # (HNSW supports COSINE, so retrieval still ranks by cosine similarity).
    # Build params:
    #   M              -> edges per node in the graph (higher = better recall, more RAM)
    #   efConstruction -> candidate list size while building (higher = better graph, slower build)
    # The search-time `ef` param is set per-query when search is added (later phase).
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{collection}' created.")
    return collection


def embed(model, texts):
    """Return normalized embeddings for a list of texts.

    If the model was loaded with a document prefix (model.doc_prefix, set by
    get_model for presets that require one), it is prepended to each text.
    """
    prefix = getattr(model, "doc_prefix", "")
    texts = [prefix + t for t in texts] if prefix else list(texts)
    return model.encode(texts, normalize_embeddings=True)


def fetch_cached(client, collection=DEFAULT_COLLECTION):
    """Return {chunk_hash: row_id} for everything already in `collection`.

    This is the embedding cache. A chunk whose hash is already here was embedded
    on an earlier run, and its vector is sitting in Milvus — so it never has to
    go through the model again. Embedding is by far the slowest stage, so this is
    what makes re-ingesting a mostly-unchanged folder nearly free.
    """
    if not client.has_collection(collection):
        return {}
    fields = {f["name"] for f in client.describe_collection(collection)["fields"]}
    if "chunk_hash" not in fields:
        return {}      # collection predates the cache; ensure_collection rebuilds it
    client.load_collection(collection)
    rows = client.query(
        collection_name=collection,
        filter="id >= 0",
        output_fields=["chunk_hash"],
        limit=16384,
    )
    return {r["chunk_hash"]: r["id"] for r in rows}


def store(client, model, documents, collection=DEFAULT_COLLECTION, embeddings=None):
    """Insert `documents` (and their vectors) into `collection`.

    Pass `embeddings` to reuse vectors you already computed; otherwise they are
    embedded here. Returns the number of rows inserted.
    """
    documents = list(documents)
    if embeddings is None:
        print("Embedding documents...")
        embeddings = embed(model, documents)

    rows = [
        {"text": doc, "embedding": emb.tolist(),
         "chunk_hash": content_hash(model.hf_id, doc)}
        for doc, emb in zip(documents, embeddings)
    ]

    result = client.insert(collection_name=collection, data=rows)
    client.flush(collection)   # make sure data is persisted
    count = result["insert_count"]
    print(f"Inserted {count} documents.")
    return count


def show_vectors(documents, embeddings, preview=8):
    """Print a short preview of each vector (never the full dump — that goes to
    result.py via --full). Shows the first `preview` values and the L2 norm.
    """
    print(f"\nRaw embeddings ({len(documents)} docs, dim={len(embeddings[0])}):")
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        head = ", ".join(f"{v:+.4f}" for v in emb[:preview])
        norm = float((emb ** 2).sum()) ** 0.5
        print(f"\n[{i}] {doc}")
        print(f"    vector[:{preview}] = [{head}, ...]   (||v|| = {norm:.4f})")


def write_results(documents, embeddings, path=DEFAULT_RESULT):
    """Write each sentence + its full embedding vector to a Python file.

    The vector length matches the active model's dimension (bge-m3 1024,
    minilm 384); it is taken from the embeddings themselves, not hardcoded.

    The file defines `results`, a list of {"text": ..., "vector": [...]} dicts,
    so it can be imported: `from result import results`.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write('"""Auto-generated by `loadmilvus.py --full`. Each entry is a\n')
        f.write('sentence and the full embedding vector the model produced for it."""\n\n')
        f.write(f"dim = {len(embeddings[0])}\n\n")
        f.write("results = [\n")
        for doc, emb in zip(documents, embeddings):
            f.write(f"    {{\n        {'text'!r}: {doc!r},\n")
            f.write(f"        {'vector'!r}: {emb.tolist()!r},\n    }},\n")
        f.write("]\n")
    print(f"Wrote {len(documents)} sentence/vector pairs to {path}.")


def parse_model_arg(argv, default=DEFAULT_MODEL):
    """Read the model name from `--model <name>` in argv.

    Falls back to the MILVUS_MODEL env var, then `default`. Accepts a preset key
    or a raw HF id (resolve_model handles either).
    """
    if "--model" in argv:
        i = argv.index("--model")
        if i + 1 >= len(argv):
            raise SystemExit("--model needs a value, e.g. --model bge-m3")
        return argv[i + 1]
    return os.environ.get("MILVUS_MODEL", default)


def list_models():
    """Print the available model presets and exit."""
    print("Available model presets (--model <key>):")
    for key, cfg in MODEL_PRESETS.items():
        default = "  (default)" if key == DEFAULT_MODEL else ""
        print(f"  {key:<10} {cfg['id']}{default}")
    print("Any other --model value is used as a literal Hugging Face model id.")


def main():
    """Storing flow: embed the docs from input.md, show the raw vectors, store them."""
    if "--list-models" in sys.argv:
        list_models()
        return

    full = "--full" in sys.argv
    model_name = parse_model_arg(sys.argv)

    documents, _ = load_inputs(DEFAULT_INPUT)
    if not documents:
        raise SystemExit(f"No documents found in {DEFAULT_INPUT}. Add some under a '# Documents' heading.")
    print(f"Loaded {len(documents)} documents from {DEFAULT_INPUT}.")

    model = get_model(model_name)

    # Embed once; show a short preview (never the full vectors), reuse for storage.
    embeddings = embed(model, documents)
    show_vectors(documents, embeddings)

    if full:
        # --full persists every sentence + full vector to result.py and
        # regenerates the benchmark statistics. (The full vectors are written to
        # result.py, not dumped to the terminal.)
        write_results(documents, embeddings)
        print("\nRegenerating benchmark statistics (statistics.md)...")
        import benchmark   # lazy import to avoid a circular dependency
        benchmark.main(model_name)

    client = connect()
    # reset=True makes input.md the single source of truth: each run rebuilds the
    # collection to match the file exactly, so editing input.md + re-running never
    # creates duplicates. (Call store() yourself with reset=False to append instead.)
    # Stored last so the collection ends populated (the benchmark above resets it).
    ensure_collection(client, dim=get_dim(model), reset=True)
    store(client, model, documents, embeddings=embeddings)

    print("\nDone. Vectors generated and stored in Milvus.")


if __name__ == "__main__":
    main()
