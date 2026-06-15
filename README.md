# Milvus vector db POC

Embeds text with a sentence-transformer model (`all-MiniLM-L6-v2`) and **stores**
it in a Milvus vector database. Scope is the storing pipeline only — embed
documents and insert their vectors into Milvus (semantic search is a later phase).

## Prerequisites

- Python 3.9+
- Docker (Docker Desktop running)
- Internet access on first run (to download the embedding model)

## Setup

Clone the repo, then from the project folder:

### 1. Start Milvus (Docker)

With Docker Desktop running, from the project folder:

```bash
docker compose up -d
```

Milvus now runs on `localhost:19530`. Check it's up with `docker compose ps`
(wait for `healthy` — ~30–90s on first boot, which also pulls the image).

Everyday commands:

```bash
docker compose up -d        # start
docker compose ps           # status (look for "healthy")
docker compose logs -f      # follow logs
docker compose down         # stop & remove the container (data is kept)
docker compose down -v      # also wipe the data volume (clean slate)

use   -restart
docker compose down -v
docker compose up -d
docker compose ps 

The container and its data persist, so after the first run you just
`docker compose up -d` again whenever you want Milvus back.

> **Why compose / a named volume?** Data is stored in the `milvus_data` named
> volume (defined in `docker-compose.yml`), not a Windows host bind-mount. On
> Windows the embedded etcd needs fast `fsync`; bind-mounting the data dir to a
> Windows path makes etcd time out and Milvus panics on boot (`etcdserver:
> leader changed`, exit 134). The named volume lives inside the Docker VM and
> avoids this. The old `standalone_embed.sh` script also assumes Linux `sudo`
> and fails on Windows.

### 2. Create a virtual environment and install deps

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate
# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate   # or .venv/bin/activate on macOS / Linux

pip install -r requirements.txt
```

### 3. Add your text and run

Edit `input.md` — every bullet line under `# Documents` is indexed as one
document:

```markdown
# Documents

- The forecast predicts heavy rain across the region tomorrow.
- Storm clouds gathered quickly over the coastal town.
```

Then run:

```bash
python loadmilvus.py          # shows an 8-value preview of each vector
python loadmilvus.py --full   # prints the full 384-dim vectors
```

This embeds the documents, prints the raw vectors the model produced, and stores
them in the `documents` collection. The collection is rebuilt each run
(`reset=True`), so `input.md` is the single source of truth — editing and
re-running never creates duplicates.

## Benchmarks

`benchmark.py` measures the storing pipeline and writes `statistics.md`:

```bash
python benchmark.py
```

Captured metrics: cold-start costs (model load, connect, collection+index create,
flush), per-op latency (embed / insert, with best / p95 / p99 / max), throughput
(docs/sec), batch-size scaling, and vector storage size projected to 1K / 1M
vectors.

## Files

| File | Purpose |
|------|---------|
| `loadmilvus.py` | Embed text from `input.md` and store it in Milvus (modular functions: `get_model`, `connect`, `ensure_collection`, `embed`, `store`). |
| `input.md` | The documents to index (bullets under `# Documents`). |
| `benchmark.py` | Measure storing-pipeline latency/throughput → `statistics.md`. |
| `statistics.md` | Generated benchmark results. |
| `requirements.txt` | Python dependencies (`pymilvus`, `sentence-transformers`). |

## Notes

- The embedding model (`all-MiniLM-L6-v2`, ~80MB) downloads automatically on first run and is cached locally. It produces 384-dimensional vectors.
