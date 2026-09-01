# Milvus vector DB POC

An end-to-end semantic search pipeline on [Milvus](https://milvus.io/): ingest
documents, embed them with a sentence-transformer, store the vectors, then
**search**, **cluster**, and **visualize** them.

```
 PDFs (fileinput/)  ──extractpdf.py──┐                                   ┌──> search.py        (7 retrieval methods + rerank)
                                     ├──> embed ──> Milvus `documents` ──┼──> cluster.py       (KMeans -> `cluster` label)
 text  (input.md)   ──loadmilvus.py──┘        (HNSW / COSINE)            ├──> customcluster.py (criteria.md -> named clusters)
                                                                         ├──> hierarchicalsearch.py (rank clusters -> 5/4/3/2/1)
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

## The app

Everything below is also a Streamlit front-end, which is the easier way to use
any of it:

```bash
streamlit run app.py
```

Six pages: **Search** (all seven methods, every flag, including the hierarchical
ones `search.py` can't reach), **Compare** (one query across several methods,
with a Jaccard agreement matrix), **Ingest** (folder or upload, chunking, the
full OCR flag set), **Clustering** (KMeans, or edit `criteria.md` in place and
run it), **Visualise** (UMAP inline), and **Collection** (schema, sources,
cluster breakdown, row browser, benchmark, drop).

> Streamlit binds `0.0.0.0` by default and prints an external URL. Add
> `--server.address localhost` if you don't want it reachable from your network.

The UI is a thin layer over the same functions the CLIs call — see
`milvusui/runner.py` for how modules that `print()` and `raise SystemExit` are
adapted to it. One thing worth knowing: the backend caches are process-lifetime,
so if you change the collection from a terminal while the server is running, hit
**Refresh caches** in the sidebar.

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
dynamic `source` field, so results stay traceable to their document.

Extraction is **pypdfium2**, so the supported set is `.pdf`, `.txt` and images
(`.png .jpg .jpeg .tif .tiff .bmp .webp`) — the XPS/EPUB/MOBI/FB2/CBZ formats
MuPDF handled are gone; images arrive in exchange, via OCR.

### OCR

A scanned PDF has no text layer and extracts empty, so `extractpdf.py` OCRs it.
Images have no text layer by definition and are always OCR'd whole.

| Flag | Default | Meaning |
|------|---------|---------|
| `--ocr auto\|always\|never` | `auto` | `auto` OCRs only pages whose text layer is too thin; `never` is text-layer only (and skips images). |
| `--ocr-min-chars N` | `32` | `auto` only: fewer characters than this on a page means "no text layer". |
| `--ocr-backend <name>` | `unlimited` | OCR engine. The default is a local VLM (`baidu/Unlimited-OCR`). |
| `--ocr-dpi N` | `300` | Render resolution for the page image. |
| `--ocr-quant auto\|none\|4bit\|8bit` | `auto` | Model quantisation. |
| `--ocr-prompt <text>` | `document parsing.` | Prompt given to the VLM. |
| `--ocr-url <url>` | `http://127.0.0.1:10000` | For server-style backends. |
| `--ocr-figures` | off | Also parse detected figure regions. |

> OCR output is part of the chunk hash, so changing any OCR setting produces
> different text and re-embeds the affected chunks. That is correct — it *is*
> different text — but it means OCR settings are not free to change. VLM decoding
> is not bit-deterministic across driver or quantisation changes either, so
> expect some churn. The default backend needs a GPU; without one use
> `--ocr never` or a server backend.

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
python search.py "how is risk handled?" --method hierarchical
```

Seven techniques, chosen with `--method`:

| Method | How it ranks | Good at |
|--------|--------------|---------|
| `dense` | Embeds the query, ANN search over the HNSW index (cosine). | Meaning, paraphrase. Weak on exact tokens. |
| `lexical` | BM25 over the stored text. No model, no vectors. | Exact terms (codes, names, "WMT 2014"). Blind to paraphrase. |
| `tfidf` | TF-IDF cosine — the classic lexical baseline. | A simpler keyword contrast to BM25. |
| `hybrid` **(default)** | Runs dense + lexical, fuses by Reciprocal Rank Fusion. | The strong default: semantic recall + keyword precision. |
| `weighted` | Weighted sum of min-max-normalized dense & lexical scores. | When you want an explicit dial (`--alpha`) instead of RRF. |
| `mmr` | Dense, then Maximal Marginal Relevance re-selection. | Avoiding near-duplicate results (`--lambda`). |
| `hierarchical` | Ranks the **clusters** against the query, then takes 5/4/3/2/1 chunks from the top five. | Broad or exploratory queries: guarantees coverage of five regions instead of 15 hits from one. Needs cluster labels. |

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
| `--allocation a,b,c` | `5,4,3,2,1` | `hierarchical` only. Chunks taken from the 1st, 2nd, … ranked cluster. |
| `--rerank-model <id>` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder to rerank with. |

> Dense/hybrid/weighted/mmr/hierarchical embed the query, so `--model` must match
> the model the stored vectors were built with. `search.py` checks the dimension
> and exits with a clear error on mismatch. `lexical` and `tfidf` need no model
> at all.

### Hierarchical search

Every other technique searches the whole pot at once. `hierarchical` does it in
two stages: score each **cluster** against the query (cosine to the cluster
centroid), then run a cluster-restricted dense search against the winners with a
decreasing quota — 5 chunks from the best cluster, 4 from the 2nd, 3 from the
3rd, 2 from the 4th, 1 from the 5th. Fifteen results that span five regions of
the corpus by construction, weighted toward the best one but never owned by it.
It's diversification like `mmr`, but at cluster granularity and reserved up
front rather than penalised per result.

The trade: if the answer lives entirely in cluster 1, six of the fifteen seats
are spent elsewhere. Use it for survey questions, not pinpoint lookups.

Run `cluster.py` or `customcluster.py` first — with no `cluster` labels there is
nothing to rank, and it says so. `hierarchicalsearch.py` also has its own CLI,
which prints the cluster leaderboard (which cluster won, by what cosine, and how
many chunks it contributed) before the results:

```bash
python hierarchicalsearch.py "how is risk handled?"
python hierarchicalsearch.py "how is risk handled?" --adaptive  # shape from the scores
python hierarchicalsearch.py "attention" --allocation 8,4,2,1   # steeper funnel
python hierarchicalsearch.py "pricing" --rank-by name           # rank by cluster label
python hierarchicalsearch.py "BLEU" --sort score --rerank
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--allocation a,b,c` | `5,4,3,2,1` | Per-rank quotas. Length = how many clusters get searched. |
| `--adaptive` | off | Derive the funnel's shape from the cluster scores; `--allocation` then sets only the total and the cluster count. |
| `--sharpness F` | `1.0` | `--adaptive` only. How hard score gaps are amplified. `0` = even split. |
| `--floor auto\|off\|F` | `off` | Make each quota a cap: drop picks below the cutoff instead of padding the seat. `auto:M` tunes the multiple (default 1.5). |
| `--hybrid` | off | Add a BM25 pass inside each selected cluster, RRF-fused with the dense one. |
| `--rank-by` | `probe` | `probe` = one flat sweep, score each cluster by the mean of its best 3 hits. `centroid` = mean of member vectors. `name` = embed the label as `name: description`. |
| `--probe N` | `300` | `--rank-by probe` only. Depth of the flat sweep. |
| `--criteria <path>` | `criteria.md` | `--rank-by name` only. Where to read the label descriptions from. |
| `--sort` | `cluster` | `cluster` keeps the funnel order; `score` re-sorts everything by cosine. |
| `--include-unassigned` | off | Let the `unassigned` bucket (`cluster = -1`) compete for a slot. |
| `--no-pool-reuse` | off | Go back to one filtered search per cluster instead of taking chunks from the probe pool. Diagnostic; see below. |

A cluster smaller than its quota simply contributes fewer chunks — nothing is
backfilled from another cluster, since that would undo the guarantee the
allocation exists to make.

**`--floor auto` makes the quota a cap.** A cluster told to produce 5 chunks
when it has 3 worth having will otherwise pad the rest with whatever it had
left. The cutoff is the score of the (1.5 × budget)-th best chunk in the probe
pool — read off this query's own distribution, not an absolute cosine, for the
reason `cluster.py` documents at `DEFAULT_FLOOR_SIGMA`. It's a precision-vs-
coverage dial: measured over three queries on the Nintendo corpus it keeps
15/12/12 of 15 seats at the default 1.5, 14/11/12 at `auto:1.2`, and 15/13/15
at `auto:2.0` (nearly inert). Empty seats are reported, never backfilled.

**`--hybrid` adds BM25 inside each selected cluster**, RRF-fused with the dense
pass, for queries that hinge on an exact token. `score` then becomes the RRF
score and the cosine is printed beside it. Combined with `--floor`, BM25 ranks
only over chunks that clear the floor — otherwise a lexical hit sharing one
token with the query walks into a seat the floor just emptied (measured: 9 of
15 seats, on "Nintendo Switch 2 hardware sales"). Restricting BM25's corpus does
change its IDF and average-document-length statistics, but measurably for the
better here: in cluster 2 the unfiltered BM25 top 5 is chunks that merely repeat
the token (cosines 0.24–0.33), and the chunk listing the actual subsidiaries
climbs from BM25 rank 11 to rank 3 once the corpus is restricted.

**Why `probe` and not `centroid`?** A centroid is the mean of *every* member, so
a large heterogeneous cluster averages out toward the corpus mean. Measured on
the Nintendo corpus with the query *"where are the subsidiaries"*, centroid
ranking spread all 12 clusters across a 0.13 cosine band — consecutive ranks
differed by ~0.003, which is noise — and ranked `corporate governance` **8th of
12** even though it held the 2nd-best chunk in the whole corpus. Probing scores
a cluster by the best material it actually has for *this* query, which is what
stage 2 is about to retrieve anyway. It is also cheaper: one ANN search instead
of reading every stored vector.

| Ranking | Top-5 clusters for "where are the subsidiaries" |
|---|---|
| `centroid` | front matter, shares and dividends, business and strategy, employees, financial results |
| `probe` | front matter, business and strategy, financial results, corporate governance, shares and dividends |

The second list is the one holding the consolidation notes, the subsidiary
table, the subsidiaries' year-ends and the governance-of-subsidiaries text.

### Stage 2 takes its chunks from the probe pool

The probe is already a cosine-ranked sweep with every hit's cluster attached —
that is how stage 1 ranks anything. So when the pool covers a cluster's quota,
re-asking Milvus for the same chunks under a `cluster == n` filter buys nothing
but a round trip. Stage 2 slices the pool, falling back to the filtered search
only for a cluster the pool didn't cover. `--no-pool-reuse` forces the old path.

Safe because the pool is the *higher*-recall read: the flat sweep sees each
cluster at `ef = --probe`, the filtered search at `ef = 64`. Over 25 queries × 3
configurations (default, `--floor auto`, `--allocation 8,4,2,1`) the two paths
returned **byte-identical results in 75 of 75 cases**.

| Path | calls/query | median ms |
|---|---|---|
| one filtered search per cluster | 7.0 | 40.9 |
| pool reuse | 2.5 | 32.1 |
| pool reuse, warm process | 1.5 | 27.7 |
| `--hybrid --floor auto`, cold | 12.0 | 999.3 |
| `--hybrid --floor auto`, warm | 7.7 | 280.0 |

"Warm" is the second and later query in one process — what a UI or sweep does.
The `--hybrid` rows matter most: that path pages whole clusters out of Milvus and
builds a BM25 index over each, none of which depends on the query.

> **Long-lived processes cache the collection.** `has_cluster_names`,
> `cluster_scheme`, each cluster's text and its BM25 index are cached per process
> (`hierarchicalsearch.clear_caches()`); `search.py` caches its BM25 index keyed
> by row ids. A CLI run is unaffected. **Re-ingesting inside a running process
> needs `clear_caches()`**, or those reads are stale.

### Thin clusters are scored on what they have — on purpose

`probe` scores a cluster by the mean of its best three hits; a cluster with fewer
than three is scored on what it has, not penalised for the shortfall. That reads
like a bug — one chunk at 0.65 outranks three at 0.64/0.63/0.62, the "one lucky
chunk carries the cluster" case the mean exists to prevent.

Penalising it was tried and measured **wrong**. Hit count conflates relevance
with cluster *size*: a small cluster cannot place many chunks in the pool however
well it matches. Padding the missing slots with the pool's weakest score, over 25
queries, changed the top-5 on 3 and the top-3 on 2 — and the clusters it demoted
out of rank 1 were `financial exchange rate and disaster risks` for "foreign
exchange rate risk" and `climate and environmental initiatives` for "greenhouse
gas emissions", both on 2 pool hits. Both are exactly the right cluster, and both
are thin because they are small.

Any penalty proportional to hit count therefore demotes the narrow,
precisely-matching cluster — the case the technique most needs to get right. The
real cost (a 2-hit cluster taking a quota of 5 and padding from chunks that never
made the pool) belongs to allocation, not ranking, and `--floor auto` already
handles it by making the quota a cap.

> **`--rank-by name` reads `criteria.md`.** Milvus stores only the label's
> *name*; the description that `customcluster.py` actually embedded to place the
> rows isn't stored anywhere. Three words rank far worse than the sentence, so
> name mode reads the descriptions back out of the criterion file and ranks on
> the same `name: description` anchor text the clustering used. If the file is
> missing it falls back to bare names; if the file's `scheme` no longer matches
> the collection's `cluster_scheme` it says so and ignores the descriptions.
>
> Name mode is only as good as the agreement between your labels and their
> members. Under `assign = seeded` the centroids move off your anchors, and the
> names can end up describing something other than what's in the bucket — at
> which point `centroid` is the honest ranking and `name` is ranking a fiction.
> Use `assign = hard` if you want name mode to be trustworthy.

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

> **It rebuilds the collection**, dropping and recreating it from the rows it
> read. `store_labels` therefore carries every field back through — `source`,
> `seq` and `chunk_hash` as well as the label — because anything left out is
> destroyed for good: `source` is what results are attributed to, and losing
> `chunk_hash` would force a full re-embed on the next ingest.

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
| `extractpdf.py` | Extract + chunk documents from `fileinput/` with pypdfium2 (plus OCR for scans and images), store them with a `source` field. |
| `search.py` | Retrieval front-end: dense / lexical / tfidf / hybrid / weighted / mmr / hierarchical, plus cross-encoder reranking. |
| `hierarchicalsearch.py` | Two-stage search: rank the clusters, then take 5/4/3/2/1 chunks from the top five. Own CLI; also `search.py --method hierarchical`. |
| `cluster.py` | KMeans over the stored vectors; writes a `cluster` label back into Milvus. |
| `customcluster.py` | Clustering on a user-defined criterion read from `criteria.md`. |
| `criteria.md` | What to cluster on: mode, labels, settings. Documented in this README. |
| `visualize.py` | UMAP projection of the vectors to an interactive Plotly scatter. |
| `benchmark.py` | Measure the storing pipeline → `statistics.md`. |
| `app.py` | Streamlit front-end. Page registration and the connection check only. |
| `milvusui/` | The UI: `runner.py` (adapts print/SystemExit), `resources.py` (cached client, models, collection state), `components.py` (shared widgets), `views/` (one module per page). |
| `test_ui.py` | Headless render + interaction test for every page (`python test_ui.py`). |
| `input.md` | Documents to index (bullets under `# Documents`). |
| `fileinput/` | Drop PDFs here for `extractpdf.py`. Contents git-ignored. |
| `docker-compose.yml` | Milvus standalone + its named data volume. |
| `requirements.txt` | Python dependencies. |

Generated, git-ignored: `result.py` (full vectors), `statistics.md` (benchmarks),
`clusters.html` (the plot).

## Notes

- The embedding model downloads on first use and is cached locally (`bge-m3` is
  a few GB; `minilm` is ~80MB).
- **Searching in a loop? Pass `model=`.** `search.search` otherwise calls
  `get_model` per query, which re-reads the weights from the local cache every
  time — ~10s for bge-m3. Invisible to a CLI run that searches once and exits;
  crippling to a UI, sweep or eval loop. `hierarchical_search` already takes the
  model as an argument. Measured: 9.8s → 0.27s per repeated query.
- `lexical` / `hybrid` / `weighted` pull the whole corpus client-side for BM25 —
  fine at POC scale, not how you'd do it in production (Milvus has a native
  sparse/BM25 field for that).
