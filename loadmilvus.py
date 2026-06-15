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

import sys
from pathlib import Path

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Defaults — a UI can override any of these per call.
DEFAULT_URI = "http://localhost:19530"
DEFAULT_COLLECTION = "documents"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INPUT = "input.md"
DEFAULT_RESULT = "result.py"


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
    """Load and return the sentence-transformer embedding model."""
    print(f"Loading embedding model '{name}'...")
    model = SentenceTransformer(name)
    print(f"Model loaded. Embedding dimension = {get_dim(model)}")
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


def ensure_collection(client, dim, collection=DEFAULT_COLLECTION, reset=False):
    """Make sure `collection` exists with the right schema and index.

    Args:
        client: a connected MilvusClient.
        dim: embedding dimension the vector field should hold.
        collection: collection name.
        reset: if True, drop and recreate for a clean slate.
    """
    if reset and client.has_collection(collection):
        client.drop_collection(collection)

    if client.has_collection(collection):
        return collection

    # Schema: auto ID, the source text, and its vector.
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=2048)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

    # COSINE works well with normalized sentence embeddings.
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 128},
    )

    client.create_collection(
        collection_name=collection,
        schema=schema,
        index_params=index_params,
    )
    print(f"Collection '{collection}' created.")
    return collection


def embed(model, texts):
    """Return normalized embeddings for a list of texts."""
    return model.encode(list(texts), normalize_embeddings=True)


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
        {"text": doc, "embedding": emb.tolist()}
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
    """Write each sentence + its full 384-dim vector to a Python file.

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


def main():
    """Storing flow: embed the docs from input.md, show the raw vectors, store them."""
    full = "--full" in sys.argv

    documents, _ = load_inputs(DEFAULT_INPUT)
    if not documents:
        raise SystemExit(f"No documents found in {DEFAULT_INPUT}. Add some under a '# Documents' heading.")
    print(f"Loaded {len(documents)} documents from {DEFAULT_INPUT}.")

    model = get_model()

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
        benchmark.main()

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
