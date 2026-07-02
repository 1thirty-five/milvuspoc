# Milvus vector db POC

Embeds text with a sentence-transformer model and **stores** it in a Milvus
vector database. Scope is the storing pipeline only — embed documents and insert
their vectors into Milvus (semantic search is a later phase).

The embedding model is chosen per run with `--model` (see [Choosing a model](#choosing-a-model)),
so the same pipeline can demo different sentence-transformers. The default is
`minilm` (`all-MiniLM-L6-v2`, 384-dim).

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

# full restart with a clean slate
docker compose down -v && docker compose up -d && docker compose ps
```

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
python loadmilvus.py --full   # also writes full vectors to result.py + refreshes statistics.md
```

This embeds the documents, prints the raw vectors the model produced, and stores
them in the `documents` collection. The collection is rebuilt each run
(`reset=True`), so `input.md` is the single source of truth — editing and
re-running never creates duplicates.

### Ingesting PDFs (and other documents)

Instead of typing documents into `input.md`, drop files into the `fileinput/`
folder and ingest the whole folder with PyMuPDF:

```bash
python extractpdf.py                  # preview the chunks from every file in fileinput/
python extractpdf.py --store          # embed + store the whole folder in Milvus
python extractpdf.py --store --model bge-m3
python extractpdf.py file.pdf --out input.md   # one file -> input.md bullets instead
```

Each file's text is split into chunks that fit Milvus's `VARCHAR(2048)` `text`
field (`--max-chars`, default 1500; `--overlap`, default 200, repeats a little
text across chunk boundaries). Every chunk is stored with its source filename in
the collection's dynamic `source` field, so you can tell which document a result
came from. PyMuPDF also opens XPS/EPUB/MOBI/FB2/CBZ/TXT. Scanned/image-only PDFs
have no text layer and extract empty — that needs OCR (see `pdf-extractors.md`).
Like `loadmilvus.py`, `--store` rebuilds the collection each run, so the folder
is the single source of truth (no duplicates). The folder's contents are
git-ignored; only a `.gitkeep` is committed.

### Choosing a model

Pick the embedding model for a run with `--model`; one model is used for the
whole run. List the built-in presets with `--list-models`:

```bash
python loadmilvus.py --list-models
python loadmilvus.py                  # default preset: minilm (384-dim)
python loadmilvus.py --model bge-m3   # BAAI/bge-m3 (1024-dim)
```

| Preset | Hugging Face id | Dim |
|--------|-----------------|-----|
| `minilm` (default) | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `bge-m3` | `BAAI/bge-m3` | 1024 |

Any other `--model` value is used as a literal Hugging Face model id, so you can
try a model without adding a preset. You can also set `MILVUS_MODEL` as the
default instead of passing the flag. Presets live in `MODEL_PRESETS` in
`loadmilvus.py` — add an entry to register a new one (an optional `doc_prefix` /
`trust_remote_code` per preset covers models that need them, e.g. nomic-embed).

There's a single `documents` collection, rebuilt each run at the chosen model's
dimension, so **switching models replaces the previously stored data** (only one
model's vectors live in Milvus at a time). Larger models (bge-m3) download a few
GB on first use and, on a CPU-only PyTorch install, embed noticeably slower than
minilm.

## Benchmarks

`benchmark.py` measures the storing pipeline and writes `statistics.md`:

```bash
python benchmark.py                 # default model (minilm)
python benchmark.py --model bge-m3  # benchmark a specific preset / HF id
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
| `extractpdf.py` | Extract text from documents in `fileinput/` with PyMuPDF, chunk it, and store it in Milvus (`get_model`, `extract_chunks`, `extract_records`, `store_records`). |
| `fileinput/` | Drop PDFs (and other PyMuPDF formats) here for `extractpdf.py` to ingest; contents are git-ignored. |
| `benchmark.py` | Measure storing-pipeline latency/throughput → `statistics.md`. |
| `statistics.md` | Generated benchmark results. |
| `requirements.txt` | Python dependencies (`pymilvus`, `sentence-transformers`). |

## Notes

- The default embedding model (`all-MiniLM-L6-v2`, ~80MB, 384-dim) downloads automatically on first run and is cached locally. Other models selected with `--model` download on first use — see [Choosing a model](#choosing-a-model).
