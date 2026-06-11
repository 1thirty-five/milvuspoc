# Milvus vector db POC

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

### 3. Run

```bash
python loadmilvus.py
```


## Notes

- The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run and is cached locally.

