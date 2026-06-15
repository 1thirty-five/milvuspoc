"""
Measure storing-pipeline metrics and write them to statistics.md.

Scope matches loadmilvus.py: storing only (embed text -> insert into Milvus).
No search/fetch is measured.

What it captures:
  - One-time / cold-start costs : model load, connect, collection+index, flush
  - Per-op latency             : embed (1 doc & batch), insert  (best/mean/p95/p99/max)
  - Throughput                 : docs/sec for embedding and insert
  - Batch-size scaling         : latency + throughput at 1/10/50/100 docs
  - Vector size + storage       : bytes per vector, projected to 1K / 1M vectors

For per-op latency we also report:
  - best : single fastest run (warm)
  - bo10 : best-of-10 (min over a 10-run batch)

Run:
    python benchmark.py
"""

import time
from datetime import datetime
from math import ceil
from statistics import mean

from loadmilvus import (
    DEFAULT_COLLECTION,
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    connect,
    embed,
    ensure_collection,
    get_dim,
    get_model,
    load_inputs,
)

RUNS = 30            # timed runs per steady-state metric (>= 10 so bo10 is meaningful)
SCALE_SIZES = [1, 10, 50, 100]   # batch sizes for the scaling table
SCALE_RUNS = 20      # runs per scaling cell; median rejects background-flush stalls
BYTES_PER_FLOAT = 4              # FLOAT_VECTOR stores float32


def pct(sorted_times, p):
    """p-th percentile (nearest-rank) of an already-sorted list."""
    idx = max(0, ceil(p / 100 * len(sorted_times)) - 1)
    return sorted_times[idx]


def timed(fn):
    """Run fn once, return elapsed milliseconds."""
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000.0


def bench(fn, runs=RUNS, warmup=3):
    """Time fn `runs` times (ms) and return a stats dict."""
    for _ in range(warmup):
        fn()
    times = [timed(fn) for _ in range(runs)]
    s = sorted(times)
    return {
        "best": s[0],
        "bo10": min(times[:10]),
        "mean": mean(times),
        "p50": pct(s, 50),
        "p95": pct(s, 95),
        "p99": pct(s, 99),
        "max": s[-1],
    }


def bench_insert_clean(client, dim, rows, runs=8):
    """Best insert latency (ms) with a fresh empty collection before each run.

    Resetting per run removes segment-accumulation effects, so the result is a
    clean function of batch size (N rows into an empty collection).
    """
    times = []
    for _ in range(runs):
        ensure_collection(client, dim=dim, reset=True)
        times.append(timed(
            lambda: client.insert(collection_name=DEFAULT_COLLECTION, data=rows)))
    return min(times)


def make_docs(base, n):
    """Cycle the base documents to produce exactly n of them."""
    return [base[i % len(base)] for i in range(n)]


def rows_for(documents, embeddings):
    return [
        {"text": doc, "embedding": emb.tolist()}
        for doc, emb in zip(documents, embeddings)
    ]


def lat_row(metric, s):
    return (f"| {metric} | {s['best']:.3f} | {s['bo10']:.3f} | {s['mean']:.3f} "
            f"| {s['p95']:.3f} | {s['p99']:.3f} | {s['max']:.3f} |")


def main():
    base, _ = load_inputs(DEFAULT_INPUT)
    if not base:
        raise SystemExit(f"No documents found in {DEFAULT_INPUT}.")
    n = len(base)
    one = base[0]

    # ---- One-time / cold-start costs -------------------------------------
    print("Measuring cold-start costs...")
    t = time.perf_counter()
    model = get_model()
    load_ms = (time.perf_counter() - t) * 1000.0
    dim = get_dim(model)

    connect_ms = timed(connect)
    client = connect()

    create_stats = bench(lambda: ensure_collection(client, dim=dim, reset=True), runs=5, warmup=1)

    embeddings = embed(model, base)
    rows = rows_for(base, embeddings)
    flush_stats = bench(
        lambda: (client.insert(collection_name=DEFAULT_COLLECTION, data=rows),
                 client.flush(DEFAULT_COLLECTION)),
        runs=5, warmup=1,
    )

    # ---- Steady-state latency --------------------------------------------
    print("Measuring embedding (input) latency...")
    in1 = bench(lambda: embed(model, [one]))
    inb = bench(lambda: embed(model, base))

    print("Measuring insert (output) latency...")
    ensure_collection(client, dim=dim, reset=True)
    ins = bench(lambda: client.insert(collection_name=DEFAULT_COLLECTION, data=rows))

    # Throughput from mean batch latency (docs/sec).
    embed_tput = n / (inb["mean"] / 1000.0)
    insert_tput = n / (ins["mean"] / 1000.0)

    # ---- Batch-size scaling ----------------------------------------------
    print("Measuring batch-size scaling...")
    scaling = []
    for size in SCALE_SIZES:
        docs = make_docs(base, size)
        embs = embed(model, docs)
        srows = rows_for(docs, embs)
        # Start each size from a clean collection, then take the median over many
        # runs so a one-off background-flush stall doesn't skew the result.
        e = bench(lambda d=docs: embed(model, d), runs=SCALE_RUNS, warmup=3)
        # Embed noise is one-sided (GC pauses), so its min is the clean baseline.
        # Insert uses a fresh empty collection per run to avoid accumulation skew.
        embed_best = e["best"]
        insert_best = bench_insert_clean(client, dim, srows)
        scaling.append((
            size,
            embed_best, size / (embed_best / 1000.0),
            insert_best, size / (insert_best / 1000.0),
        ))

    # Reset so repeated benchmark inserts don't leave duplicate data behind.
    ensure_collection(client, dim=dim, reset=True)

    # ---- Vector size + storage projection --------------------------------
    bpv = dim * BYTES_PER_FLOAT

    def storage(count):
        b = bpv * count
        if b < 1024 ** 2:
            return f"{b:,} B ({b / 1024:.2f} KiB)"
        if b < 1024 ** 3:
            return f"{b:,} B ({b / 1024 ** 2:.2f} MiB)"
        return f"{b:,} B ({b / 1024 ** 3:.2f} GiB)"

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scale_rows = "\n".join(
        f"| {size} | {em:.3f} | {et:,.0f} | {im:.3f} | {it:,.0f} |"
        for (size, em, et, im, it) in scaling
    )

    md = f"""# Storing Pipeline Statistics

_Generated by `benchmark.py` on {generated}._

**Setup:** model `{DEFAULT_MODEL}`, {n} base documents from `{DEFAULT_INPUT}`,
Milvus collection `{DEFAULT_COLLECTION}`, float32 vectors of dim {dim}.
Steady-state metrics use {RUNS} warm runs. All latencies in **milliseconds**.
`bo10` = best-of-10 (min over a 10-run batch).

## One-time / cold-start costs

| Step | Latency |
|------|---------|
| Model load (from cache) | {load_ms:.1f} ms |
| Milvus connect | {connect_ms:.3f} ms |
| Collection + index create (best of 5) | {create_stats['best']:.3f} ms |
| Insert + flush, {n} vectors (best of 5) | {flush_stats['best']:.3f} ms |

## Per-operation latency (warm)

| Metric | Best | bo10 | Mean | p95 | p99 | Max |
|--------|------|------|------|-----|-----|-----|
{lat_row("Input - embed 1 doc", in1)}
{lat_row(f"Input - embed batch ({n} docs)", inb)}
{lat_row(f"Output - insert {n} vectors", ins)}

## Throughput

| Operation | Throughput |
|-----------|-----------|
| Embedding (batch of {n}) | {embed_tput:,.0f} docs/sec |
| Insert (batch of {n}) | {insert_tput:,.0f} docs/sec |

## Batch-size scaling

Embed = best of {SCALE_RUNS} runs (min rejects one-sided GC noise). Insert = best
with a fresh empty collection per run, so it reflects batch size without
segment-accumulation skew. Note: at these sizes insert is **overhead-bound** -
even 100 vectors is ~150 KB, so per-call latency is dominated by fixed Milvus RPC
cost, not row count. Watch the throughput column (docs/sec), which rises cleanly
with batch size as that fixed overhead amortizes.

| Batch size | Embed best (ms) | Embed docs/sec | Insert best (ms) | Insert docs/sec |
|-----------:|----------------:|---------------:|-----------------:|----------------:|
{scale_rows}

## Vector size & storage projection

| Property | Value |
|----------|-------|
| Dimensions | {dim} |
| Dtype | float32 ({BYTES_PER_FLOAT} bytes) |
| Bytes per vector | {bpv:,} ({bpv / 1024:.2f} KiB) |
| Raw vectors @ 1K | {storage(1_000)} |
| Raw vectors @ 1M | {storage(1_000_000)} |

_Storage figures are raw vector bytes only (excludes index overhead, the stored
`text` field, and Milvus metadata)._
"""

    with open("statistics.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\nWrote statistics.md")
    print(md)


if __name__ == "__main__":
    main()
