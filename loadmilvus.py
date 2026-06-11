"""
Load text into Milvus and search it, as small modular pieces.

Each function does one thing and returns plain objects, so a UI layer can call
them independently:

    from loadmilvus import get_model, connect, ensure_collection, store, search

    model  = get_model()
    client = connect()
    ensure_collection(client, dim=model.get_sentence_embedding_dimension())
    store(client, model, ["some text", "more text"])
    results = search(client, model, "a query")

Prerequisites (run once in your terminal):
    # 1. Start Milvus standalone via Docker
    curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
    bash standalone_embed.sh start

    # 2. Install Python deps
    pip install pymilvus sentence-transformers

Then edit input.md with the text to index + your query, and run:
    python loadmilvus.py
"""

from pathlib import Path

from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# Defaults — a UI can override any of these per call.
DEFAULT_URI = "http://localhost:19530"
DEFAULT_COLLECTION = "documents"
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_INPUT = "input.md"


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


def store(client, model, documents, collection=DEFAULT_COLLECTION):
    """Embed `documents` and insert them into `collection`.

    Returns the number of rows inserted.
    """
    documents = list(documents)
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


def search(client, model, query, collection=DEFAULT_COLLECTION, limit=3):
    """Run a semantic search and return a list of {text, score} dicts."""
    query_vec = embed(model, [query])
    hits = client.search(
        collection_name=collection,
        data=query_vec.tolist(),
        limit=limit,
        output_fields=["text"],
    )
    return [
        {"text": hit["entity"]["text"], "score": hit["distance"]}
        for hit in hits[0]
    ]


def main():
    """End-to-end demo: index the docs from input.md, then run its query."""
    documents, query = load_inputs(DEFAULT_INPUT)
    if not documents:
        raise SystemExit(f"No documents found in {DEFAULT_INPUT}. Add some under a '# Documents' heading.")
    query = query or "How do I run a vector DB on my machine?"
    print(f"Loaded {len(documents)} documents from {DEFAULT_INPUT}.")

    model = get_model()
    client = connect()
    # reset=True makes input.md the single source of truth: each run rebuilds the
    # collection to match the file exactly, so editing input.md + re-running never
    # creates duplicates. (Call store() yourself with reset=False to append instead.)
    ensure_collection(client, dim=get_dim(model), reset=True)
    store(client, model, documents)

    print(f"\nSearch query: {query!r}")
    results = search(client, model, query)

    print("\nTop matches:")
    for rank, hit in enumerate(results, start=1):
        print(f"  {rank}. (score {hit['score']:.3f})  {hit['text']}")

    print("\nDone. Data is stored in Milvus and searchable.")


if __name__ == "__main__":
    main()
