# Milvus vector store POC

Stores text in a Milvus vector database using a sentence-transformer model.
Embeds documents, inserts them into Milvus, and runs a semantic search to verify.

## Prerequisites

- Python 3.9+
- Docker (Docker Desktop running)
- Internet access on first run (to download the embedding model)

## Setup

Clone the repo, then from the project folder:

### 1. Start Milvus (Docker)

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Milvus now runs on `localhost:19530`. Check it's up with `docker ps`.
This only needs to be done once per machine; the containers keep running.

### 2. Create a virtual environment and install deps

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` -> "Python: Select Interpreter" -> pick the `.venv` one.

### 3. Run

```bash
python store_in_milvus.py
```

You should see the model load, 5 documents inserted, and a ranked search result.

## Notes

- The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run and is cached locally.
- The script drops and recreates the collection on each run so it's safe to re-run. Remove the `drop_collection` block once you want data to persist.
- To use your own data, replace the `documents` list in `store_in_milvus.py`.

## What does NOT live in git

The `.venv` folder, Milvus data (`volumes/`), and the model cache are gitignored.
Each device recreates these locally via the steps above.