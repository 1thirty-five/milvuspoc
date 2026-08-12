# Milvus vector DB POC

An end-to-end semantic search pipeline on [Milvus](https://milvus.io/): ingest
documents, embed them with a sentence-transformer, store the vectors, then
**search**, **cluster**, and **visualize** them.

```
 PDFs (fileinput/)  ──extractpdf.py──┐                                   ┌──> search.py        (6 retrieval methods + rerank)
                                     ├──> embed ──> Milvus `documents` ──┼──> cluster.py       (KMeans -> `cluster` label)
 text  (input.md)   ──loadmilvus.py──┘        (HNSW / COSINE)            ├──> customcluster.py (criteria.md -> named clusters)
                                                                         └──> visualize.py     (UMAP -> clusters.html)
```

Every stage is a small module of single-purpose functions returning plain dicts,
so a UI layer can call any stage on its own. The embedding model is chosen per
run with `--model`; the default is `bge-m3` (1024-dim).

## Prerequisites

- Python 3.9+
- Docker (Docker Desktop running)
- Internet access on first run, to download the embedding model. After that the
  weights are loaded straight from the local Hugging Face cache and no run touches
  the network again. (Left to itself sentence-transformers re-checks the Hub on
  *every* load, which costs ~20s and buys nothing; `get_model` skips it.)
- An NVIDIA GPU is optional but makes embedding ~14x faster — see Setup step 3.

## Setup

### 1. Start Milvus

```bash
docker compose up -d
```

Milvus listens on `localhost:19530`. Wait for `healthy` in `docker compose ps`
(~30–90s on first boot, which also pulls the image).

```bash
docker compose up -d        # start
docker compose ps           # status (look for "healthy")
docker compose logs -f      # follow logs
docker compose down         # stop & remove container (data kept)
docker compose down -v      # also wipe the data volume (clean slate)
```

> **Why compose / a named volume?** Data lives in the `milvus_data` named volume
> (see `docker-compose.yml`), not a Windows bind-mount. On Windows the embedded
> etcd needs fast `fsync`; bind-mounting the data dir to a Windows path makes
> etcd time out and Milvus panic on boot (`etcdserver: leader changed`, exit
> 134). The named volume lives inside the Docker VM and avoids this. The old
> `standalone_embed.sh` assumes Linux `sudo` and fails on Windows.

### 2. Install dependencies

```bash
python -m venv .venv

.venv\Scripts\activate            # Windows (PowerShell)
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt
```

### 3. Enable the GPU (optional, but a 14x speedup)

`requirements.txt` gives you the **CPU** build of torch — that's what PyPI serves.
Embedding is the slowest stage of the pipeline by a wide margin, and it's the one
stage a GPU transforms. If you have an NVIDIA card, install the CUDA build over
the top (same torch version, just compiled against CUDA 12.6):

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.12.0+cu126
python -c "import torch; print(torch.cuda.is_available())"      # -> True
```

Embedding all 1072 chunks of `fileinput/` with the default `bge-m3`:

| | CPU | GPU (RTX 3050) |
|---|---|---|
| bge-m3 (1024-dim, default) | ~370s | **~26s** |
| minilm (384-dim) | ~18s | ~1.4s |

Nothing needs configuring: sentence-transformers finds the GPU on its own, and
every script prints the device it loaded onto (`Model loaded on cuda:0`).

## Quickstart

```bash
# 1. ingest — either source works, both build the same `documents` collection
python loadmilvus.py                   # from input.md
python extractpdf.py --store           # from every file in fileinput/

# 2. search
python search.py "how does the model avoid recurrence?"

# 3. cluster + visualize (optional)
python cluster.py
python visualize.py                    # writes + opens clusters.html
```

## Ingesting documents

### From `input.md`

Every bullet line under `# Documents` becomes one document:

```markdown
# Documents

- The forecast predicts heavy rain across the region tomorrow.
- Storm clouds gathered quickly over the coastal town.
```

```bash
python loadmilvus.py            # embed + store, prints an 8-value vector preview
python loadmilvus.py --full     # also write full vectors to result.py + refresh statistics.md
```

### From PDFs and other files

Drop files into `fileinput/` and ingest the folder:

```bash
python extractpdf.py                            # preview chunks (no writes)
python extractpdf.py --store                    # embed + store the whole folder
python extractpdf.py --store --model minilm
python extractpdf.py file.pdf --out input.md    # one file -> input.md bullets
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--store` | off | Embed the chunks and store them in Milvus. |
| `--out <path>` | — | Write the chunks to a markdown file instead. |
| `--max-chars N` | `600` | Target chunk size. Chunks never cut a sentence in half. |
| `--overlap N` | `100` | Characters of trailing sentences repeated into the next chunk. |
| `--reset` | off | Ignore the embedding cache and rebuild the collection from scratch. |

Text is split on sentence boundaries and packed into chunks that fit Milvus's
`VARCHAR(2048)` `text` field. Each chunk carries its source filename in the
dynamic `source` field, so results stay traceable to their document. PyMuPDF
also opens XPS/EPUB/MOBI/FB2/CBZ/TXT. Scanned/image-only PDFs have no text layer
and extract empty — that needs OCR (see `pdf-extractors.md`).

### Re-ingesting is cheap: the collection is the cache

`extractpdf.py --store` **only embeds what it hasn't seen before.** Every chunk is
keyed by a hash of (model id + text), stored beside its vector in the `chunk_hash`
field. On each run the collection is asked which hashes it already holds, and only
the misses go through the model. Chunks that disappeared from `fileinput/` are
deleted, so the collection still mirrors the folder exactly.

This means Milvus itself is the embedding cache — there's no second store to keep
in sync, and it persists in the existing `milvus_data` volume. Adding one PDF costs
only that PDF:

```
first ingest         1072 chunks embedded          ~46s
re-run, no changes   1072/1072 cache hits           ~1.7s   (model never even loads)
add one file         1072/1073 cache hits           ~17s    (1 chunk embedded)
delete that file     1 stale chunk dropped          ~2s     (nothing embedded)
```

Switching `--model` changes the hash, so it correctly misses the whole cache and
re-embeds (a different model's vectors aren't interchangeable). Use `--reset` to
force a rebuild.

> `loadmilvus.py` still rebuilds the collection wholesale (`reset=True`), so the
> two ingest paths **replace** each other's data. `extractpdf.py` is the incremental
> one; prefer it.

## Searching

```bash
python search.py "multi-head attention"                          # hybrid (default)
python search.py "multi-head attention" --method lexical
python search.py "scaled dot-product" --method hybrid --rerank
python search.py "attention" --method weighted --alpha 0.7
python search.py "attention" --method mmr --lambda 0.5 --k 5
```

Six techniques, chosen with `--method`:

| Method | How it ranks | Good at |
|--------|--------------|---------|
| `dense` | Embeds the query, ANN search over the HNSW index (cosine). | Meaning, paraphrase. Weak on exact tokens. |
| `lexical` | BM25 over the stored text. No model, no vectors. | Exact terms (codes, names, "WMT 2014"). Blind to paraphrase. |
| `tfidf` | TF-IDF cosine — the classic lexical baseline. | A simpler keyword contrast to BM25. |
| `hybrid` **(default)** | Runs dense + lexical, fuses by Reciprocal Rank Fusion. | The strong default: semantic recall + keyword precision. |
| `weighted` | Weighted sum of min-max-normalized dense & lexical scores. | When you want an explicit dial (`--alpha`) instead of RRF. |
| `mmr` | Dense, then Maximal Marginal Relevance re-selection. | Avoiding near-duplicate results (`--lambda`). |

Add `--rerank` to any method: a cross-encoder re-scores the shortlist by reading
the query and each chunk **together**, then keeps the top `--k`. More accurate
ordering, at the cost of a model pass per candidate.

| Flag | Default | Meaning |
|------|---------|---------|
| `--k N` | `5` | Results returned. |
| `--candidates N` | `50` | Shortlist depth retrieved before fusing/reranking. |
| `--ef N` | `max(64, candidates)` | HNSW search width. Higher = better recall, slower. |
| `--alpha F` | `0.5` | `weighted` only. `1.0` = all dense, `0` = all lexical. |
| `--lambda F` | `0.5` | `mmr` only. `1.0` = pure relevance, `0` = pure diversity. |
| `--rerank-model <id>` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder to rerank with. |

> Dense/hybrid/weighted/mmr embed the query, so `--model` must match the model
> the stored vectors were built with. `search.py` checks the dimension and exits
> with a clear error on mismatch. `lexical` and `tfidf` need no model at all.

## Clustering and visualizing

```bash
python cluster.py               # k = 10
python cluster.py --k 6
python visualize.py             # -> clusters.html, opens in your browser
```

`cluster.py` reads every stored vector, runs KMeans, prints each cluster with its
member documents, and writes each row's `cluster` label back into Milvus. The
label rides in the dynamic field, so you can filter on it:

```python
client.query(collection_name="documents", filter="cluster == 3",
             output_fields=["text", "cluster"])
```

`visualize.py` projects the vectors to 2D with UMAP (cosine metric, matching the
embedding space) and writes an interactive Plotly scatter, colored by cluster,
hovering to show the document text.

> **Gotcha:** `cluster.py` rebuilds the collection with only `text`, `embedding`,
> and `cluster` — it **drops the `source` field**. After clustering a PDF ingest,
> `search.py` will print `source=None`. Re-run `extractpdf.py --store` to get
> provenance back (which in turn clears the `cluster` labels).

## Clustering on your own criterion

`cluster.py` is criterion-free — KMeans groups by whatever the embedding model
thinks dominates, and the only knob is `k`. `customcluster.py` lets you say
*what to cluster on*, in words, in `criteria.md`:

```bash
python customcluster.py                     # read criteria.md
python customcluster.py --criteria mine.md  # a different criterion file
python customcluster.py --dry-run           # print the groups, write nothing
```

It works because **a criterion stated in words embeds into the same space as the
documents** — so it becomes geometry, with no training and no LLM.

### The file

```markdown
# Mode
anchor

# Labels
- pricing: fees, discounts, billing terms
- legal risk: liability, indemnity, compliance exposure

# Settings
assign = seeded
floor  = auto
model  = bge-m3
```

Comment it with `<!-- -->` only. The parser treats a leading `#` as a section
heading, so a `#` comment inside `# Settings` silently discards every setting
below it.

### Labels: the description is what does the work

Each label is embedded as *name plus description*, and the description carries
almost all of the signal. `Tier 1` embeds to noise; `Tier 1: capital adequacy
requirements under Basel III` embeds to something a corpus can actually match.
Write it the way you would explain the bucket to a colleague. A bare name works
and matches far worse.

Two labels in the shipped `criteria.md` exist to **absorb** text rather than to
be read. `front matter` catches table-of-contents dot leaders, the translation
disclaimer and the IR contact page; navigation furniture carries no subject
matter, so without a bucket of its own it spreads evenly across the real ones
and quietly dirties every cluster. `financial statements` does the same job for
the balance-sheet and note tables. Give the junk a home and the prose buckets
stay clean.

Splitting matters too. `history and group structure` is separate from strategy
because the 1947-onward founding narrative is about the company's past, not its
plans — merged, it made `business and strategy` the largest bucket in the report
by absorbing seventy years of karuta manufacturing.

### Settings

| key | default | what it does |
|---|---|---|
| `assign` | `seeded` | `seeded` uses your labels as KMeans init, so clusters settle onto real density while keeping your names. `hard` takes the nearest label and stops — fully predictable, and what you want if the buckets must mean exactly what you wrote. |
| `floor` | `auto` | Cutoff below which an assignment is called uncorrelated and sent to `unassigned` (cluster `-1`). `auto` = two sigma below this run's own mean. `0` leaves the bucket empty — it is still always reported. |
| `model` | `bge-m3` | **Must** be the model that embedded the corpus, or the labels land in a different space and the assignment is garbage. |
| `scheme` | `default` | A name for this clustering, stored per row as `cluster_scheme` so you can tell runs apart. |
| `preview` | `10` | Member chunks printed per cluster. |

**Don't set `floor` to an absolute cosine without measuring.** Modern embedding
models compress cosine into a narrow high band: on this corpus bge-m3 puts
*every* chunk between 0.64 and 0.88 of its centroid, so a floor of 0.45 — which
sounds permissive — catches exactly nothing. `auto` reads the cutoff off the run
itself and means the same thing on any model or corpus.

### Why `seeded` is the default

Measured, not assumed. The intuition says `hard` — buckets that mean exactly
what you wrote — but nearest-anchor performs badly on a single-document corpus:
every chunk sits at 0.55–0.63 cosine to *every* label, because the shared
"Nintendo annual securities report" component dominates each vector and the
discriminating margin is only what is left over. At that margin, incidental word
overlap decides the assignment.

Contiguity in document order makes it concrete. Chunks are stored in reading
order, so a clustering that has really found the sections should produce long
unbroken runs of one label:

| | `hard` | `seeded` |
|---|---|---|
| isolated flips | 16% | **8%** |
| corporate governance | span 7–397 (density 0.21) | **span 237–397 (0.55)** |
| employees | span 21–462 (density 0.05) | **span 398–462 (0.23)** |

Under `hard`, `corporate governance` was smeared across the whole document.
Under `seeded` it is the contiguous block it is in the printed report. The
centroids do drift off the literal labels — but they drift *onto the document's
own section boundaries*, which is what was wanted.

### What anchor mode cannot do

It suits **topical** criteria: which regulatory regime, which product area. A
criterion that cuts *across* topic — "how formal is this", "how urgent" —
nearest-label assignment cannot express. In raw embedding space topic dominates,
so such a criterion sends every chunk to whichever label shares its subject
matter and returns something plausible and wrong. An `axes` mode that handled
those (projecting onto `normalize(embed(left) - embed(right))`) was removed on
2026-08-12.

### Output

Labels are stored three ways per row — `cluster` (int, same as before),
`cluster_name`, and `cluster_scheme` — so `search.py` and `visualize.py` keep
working untouched:

```python
client.query(collection_name="documents", filter='cluster_name == "pricing"',
             output_fields=["text", "cluster_name"])
```

> **Gotcha:** same rebuild as `cluster.py`, so only one clustering lives in
> Milvus at a time and each run replaces the last. `cluster_scheme` tells you
> which one is currently stored.

## Choosing a model

One model is used for a whole run. List the presets with `--list-models`:

| Preset | Hugging Face id | Dim |
|--------|-----------------|-----|
| `bge-m3` **(default)** | `BAAI/bge-m3` | 1024 |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 |

```bash
python loadmilvus.py --list-models
python loadmilvus.py                  # default: bge-m3
python loadmilvus.py --model minilm   # lighter, faster cold start
```

Any other `--model` value is used as a literal Hugging Face id, so you can try a
model without adding a preset. `MILVUS_MODEL` sets the default instead of the
flag. Presets live in `MODEL_PRESETS` in `loadmilvus.py`; an optional
`doc_prefix` / `trust_remote_code` per preset covers models that need them (e.g.
nomic-embed).

There is a single `documents` collection, rebuilt each run at the chosen model's
dimension, so **switching models replaces the stored data** — only one model's
vectors live in Milvus at a time. `bge-m3` downloads a few GB on first use and,
on a CPU-only PyTorch install, embeds noticeably slower than `minilm`.

## Benchmarks

```bash
python benchmark.py                 # default model
python benchmark.py --model minilm
```

Writes `statistics.md`: cold-start costs (model load, connect, collection+index
create, flush), per-op latency (embed / insert, with best / bo10 / p95 / p99 /
max), throughput (docs/sec), batch-size scaling at 1/10/50/100 docs, and vector
storage projected to 1K / 1M vectors. Scope is the storing pipeline only; search
is not measured.

> `benchmark.py` resets the collection as it runs. `loadmilvus.py --full` calls
> it, then re-stores, so the collection ends populated.

## The collection

`documents` — created by `ensure_collection` in `loadmilvus.py`:

| Field | Type | Notes |
|-------|------|-------|
| `id` | `INT64` | Primary key, auto-generated. |
| `text` | `VARCHAR(2048)` | The document or chunk. |
| `embedding` | `FLOAT_VECTOR(dim)` | L2-normalized. `dim` follows the model. |
| `source` | dynamic | Filename, set by `extractpdf.py`. |
| `cluster` | dynamic | KMeans label, set by `cluster.py` / `customcluster.py`. |
| `cluster_name` | dynamic | Human name for the cluster, set by `customcluster.py`. |
| `cluster_scheme` | dynamic | Which criterion produced the labels, set by `customcluster.py`. |

Indexed with **HNSW** (`M=16`, `efConstruction=200`) on **COSINE**, matching the
normalized embeddings — so scores are cosine similarity and `ef` tunes recall at
query time.

## Files

| File | Purpose |
|------|---------|
| `loadmilvus.py` | Embed `input.md` and store it. Core helpers (`get_model`, `connect`, `ensure_collection`, `embed`, `store`) reused by every other script. |
| `extractpdf.py` | Extract + chunk documents from `fileinput/` with PyMuPDF, store them with a `source` field. |
| `search.py` | Retrieval front-end: dense / lexical / tfidf / hybrid / weighted / mmr, plus cross-encoder reranking. |
| `cluster.py` | KMeans over the stored vectors; writes a `cluster` label back into Milvus. |
| `customcluster.py` | Clustering on a user-defined criterion read from `criteria.md`. |
| `criteria.md` | What to cluster on: mode, labels, settings. Documented in this README. |
| `visualize.py` | UMAP projection of the vectors to an interactive Plotly scatter. |
| `benchmark.py` | Measure the storing pipeline → `statistics.md`. |
| `input.md` | Documents to index (bullets under `# Documents`). |
| `fileinput/` | Drop PDFs here for `extractpdf.py`. Contents git-ignored. |
| `docker-compose.yml` | Milvus standalone + its named data volume. |
| `pdf-extractors.md` | Why PyMuPDF, and the OCR story for scanned PDFs. |
| `requirements.txt` | Python dependencies. |

Generated, git-ignored: `result.py` (full vectors), `statistics.md` (benchmarks),
`clusters.html` (the plot).

## Notes

- The embedding model downloads on first use and is cached locally (`bge-m3` is
  a few GB; `minilm` is ~80MB).
- `lexical` / `hybrid` / `weighted` pull the whole corpus client-side for BM25 —
  fine at POC scale, not how you'd do it in production (Milvus has a native
  sparse/BM25 field for that).
