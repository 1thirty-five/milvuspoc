# Axis search — steering retrieval along a named contrast

**Status: proposal, not implemented.** Nothing in this document is in the code yet.
It is written so the decision can be made on the geometry and the evidence rather
than on intuition, and so that if the idea is rejected a second time it is
rejected for a recorded reason.

**Not to be confused with the removed `axes` clustering mode.** That mode
(`customcluster.py`, built 2026-08-04, removed 2026-08-12) used axis projections
as the *only* coordinates and ran KMeans on them. This is a query-time modifier
on retrieval. Section 3 explains why the distinction is not cosmetic.

---

## 1. The problem

All seven retrieval methods rank by topical proximity, in one form or another:

| Method | What it ranks by |
|---|---|
| `dense` | cosine of the query to the chunk |
| `lexical`, `tfidf` | term overlap |
| `hybrid`, `weighted` | a fusion of the two above |
| `mmr` | cosine, re-selected to be mutually dissimilar |
| `hierarchical` | cosine to a cluster, then cosine within it |

None of them can express **"same topic, different kind of passage."** In the
current corpus (a Nintendo annual securities report) the queries this blocks are
real ones:

- *the numbers on operating profit*, not the narrative discussion of it
- *what could go wrong* with a risk, not the mitigation measures for it
- *forward-looking statements* about a segment, not its historical results

Each pair shares its topic almost exactly, so every method above returns them
interleaved. Filtering by `cluster_name` gets close when the distinction happens
to align with a section of the document — but that is the criterion being
*topical* again, which is the only kind `customcluster.py` handles well.

## 2. The method in one line

A contrast stated in words becomes a **direction** in the embedding space, and a
direction can be added to a ranking score.

This is the same principle `customcluster.py` already runs on — "a criterion
stated in words embeds into the same space as the documents, so it becomes
geometry with no training and no LLM" — applied to the query instead of to the
corpus.

## 3. Why this is not the mode that was removed

The removed mode's failure was specific and it is worth restating precisely,
because it is the strongest argument *against* this proposal and it does not
quite land.

`axes` clustering projected every chunk onto one to three axis directions and
ran KMeans **on those projections alone**, discarding the other ~1021 dimensions.
The axis therefore had to carry the entire signal. In raw embedding space topic
dominates, so what the projections mostly encoded was residual topic, and the
resulting clusters were plausible and wrong.

At query time the arithmetic is different:

```
score(doc) = cos(query, doc)  +  beta * (axis . doc)
             ^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
             topic, supplied      the modifier: only has to
             by the query         break ties among chunks
                                  already on-topic
```

The axis never has to *find* the subject matter; the query has already narrowed
the pool to it. The axis only orders what survives. That is a genuinely weaker
demand on a weak estimator, which is the entire case for revisiting the idea.

It is not a proof. Topic still dominates the space, and if `beta` is large enough
the second term overwhelms the first and you are back to sorting the corpus by
whatever the axis actually encodes — which may not be what you named. Section 7
is about finding that out cheaply.

## 4. Constructing the direction

All stored vectors are L2-normalized (`normalize_embeddings=True` in
`loadmilvus.embed`), so **a dot product is exactly cosine similarity** and no
separate normalization step is needed at scoring time. This is the same property
`assign_anchors` and `mmr_search` already rely on.

### 4.1 One pair (what the removed mode did)

```
d = normalize(embed(left) - embed(right))
```

Cheap — two embeddings, cached per axis — and weak. Two problems:

1. **Residual topic.** The poles share subject matter. "formal academic prose"
   and "casual explanation" both contain *prose*-adjacent and *explanation*-
   adjacent content, and the difference vector keeps whatever of that fails to
   cancel. What you get is partly the contrast you named and partly the
   idiosyncrasy of the two sentences you happened to write.
2. **No way to tell.** A single difference always produces *a* unit vector. It
   never fails loudly, so a meaningless axis looks exactly like a good one until
   you inspect the results.

A minimal guard: if `cos(embed(left), embed(right)) > 0.95` the poles embed to
nearly the same point and their difference is numerical noise. Refuse the axis.
Necessary, not sufficient.

### 4.2 Several pairs, top principal component (recommended)

**This is the construction to use**, and it comes from Bolukbasi et al. (2016)
[[1]](#references), who needed a "gender direction" and did not trust a single
`he − she`. Their method:

1. Write **several** pairs that express the same contrast in different words.
2. Take each difference and **normalize it to unit length**.
3. Run **PCA** over that set of difference vectors.
4. The axis is the **top principal component**.

Two things this buys over a plain mean of the differences. First, PCA finds the
direction of maximal *variance shared across the pairs*, so wording specific to
any one pair — the residual topic of problem 1 — lands in the lower components
and is discarded rather than averaged in. Second, and more valuable here:

> **The variance explained by the top component is a diagnostic you get before
> retrieving anything.** Bolukbasi et al. report that for a real direction the
> top component explains substantially more variance than any other. If your
> pairs produce a flat spectrum — no dominant component — then the contrast you
> named is not a single coherent direction in this embedding space, and no
> choice of `beta` will rescue it.

That is a falsification test that costs `2 x len(pairs)` embeddings and one PCA
on a matrix of maybe 8 x 1024. It should run *before* any retrieval code is
written.

### 4.3 Classifier normal, from corpus examples (the fallback)

If phrases prove too weak, the stronger construction is TCAV's, from Kim et al.
(2018) [[2]](#references): a **Concept Activation Vector** is the normal of a
linear classifier trained to separate examples of a concept from random ones,
and a thing's alignment with the concept is the directional derivative along
that normal.

Translated to this pipeline: label ~20 chunks per pole *from the corpus itself*,
fit a linear separator on their stored vectors, and use its normal as `d`. This
costs labelling effort and stops being zero-configuration, but it estimates the
direction from text that actually exists in the corpus rather than from your
phrasing of it. Keep it in reserve: it is the answer to "the axis is real but my
phrases can't find it," not to "the axis isn't real."

### 4.4 A note on anisotropy

Sentence embeddings occupy a narrow cone rather than the whole sphere — there is
a dominant mean direction shared by nearly every vector. A *difference* cancels
that common component by construction, which is a quiet advantage of this whole
family of methods: subtracting two embeddings mean-centers them for free. It is
also why the raw projection `d . doc` is roughly centered on zero across a corpus
and can be read as a signed position on the contrast, negative toward `right`
and positive toward `left`.

## 5. Applying it at query time

Three options. They differ in cost, in what they can express, and in how much of
the existing system they disturb.

| | Mechanism | Cost | Disturbs |
|---|---|---|---|
| **A. Query shift** | `q' = normalize(q + beta*d)`, then one ordinary HNSW search | ~free; uses the index | nothing |
| **B. Pool re-rank** | fetch `--candidates` with vectors, re-score, re-sort | ≈ `mmr`'s cost | nothing |
| **C. Stored projection** | write `axis_x` per row at ingest, filter server-side | a re-ingest per axis | the schema |

### A. Query shift — the Rocchio move

This is not a new idea; it is **Rocchio (1971)** [[3]](#references), the
relevance-feedback algorithm from Salton's SMART system, which builds a modified
query

```
q' = alpha*q0 + beta*(centroid of relevant docs) - gamma*(centroid of non-relevant docs)
```

— move the query toward one pile and away from another. Axis search is that
formula with **two written phrases substituted for two sets of judged
documents.** Much cheaper, and strictly weaker, and that substitution is the
whole bet of this proposal.

Worth taking from Rocchio directly: its weights "need to be set empirically."
Fifty-five years of use have not produced a principled setting for them. Expect
none for `beta` either (§8).

The practical drawback is that shifting the query moves it **off the data
manifold**. A large `beta` puts `q'` somewhere no document lives, and the nearest
neighbours of a point in empty space are arbitrary. The failure is silent: you
get fifteen confident results that are simply wrong.

### B. Pool re-rank — build this one

Retrieve a pool by ordinary dense search, then re-score every candidate:

```
score(doc) = cos(query, doc) + beta * (d . doc)
```

The query stays where it belongs, the axis only reorders chunks the corpus
actually contains, and `beta` degrades gracefully — at worst you get the pool
back in a bad order, never fifteen results from empty space.

It also needs **no new machinery**. `dense_search(..., with_vectors=True)`
(`search.py:102`) already returns each chunk's stored `embedding`; it was written
for `mmr_search` (`search.py:267`), which already does exactly this shape of
work — pull a pool with vectors, score with numpy dot products, re-select. Axis
search is a simpler version of a function that has been in the file since 7 July.

### C. Stored projection — not now

Computing `d . doc` for every row at ingest and storing it as a dynamic field
would let Milvus filter server-side (`filter="axis_formality > 0.1"`), which is
the only version that scales past a client-side pool. It is also the only version
that requires a **re-ingest per axis**, and it collides head-on with the standing
problem that both clustering paths drop and recreate the collection, so only one
labelling scheme fits at a time. Revisit only after that is fixed.

## 6. Concrete design for this repo

### 6.1 A modifier, not a method

`--axis` should stack on any `--method`, the way `--rerank` does:

```powershell
.venv\Scripts\python.exe search.py "operating profit" --axis "a table of figures vs a narrative discussion" --axis-beta 0.3
.venv\Scripts\python.exe search.py "risk" --method hybrid --axis "what could go wrong vs how it is mitigated"
.venv\Scripts\python.exe search.py "segment results" --method hierarchical --axis "forward-looking vs historical" --rerank
```

An eighth `--method` value would inherit the problem the seventh already has:
`search.py` calls `hierarchical_search` with five positional arguments, so
`--rank-by`, `--floor`, `--hybrid` and `--adaptive` are unreachable from the main
CLI. A modifier composes with all seven for free and adds no branch to the GUI's
method switch.

| Flag | Default | Meaning |
|---|---|---|
| `--axis "<left> vs <right>"` | — | The contrast. Repeatable for multi-pair (§4.2). |
| `--axis-beta F` | `0.3` | Weight of the axis term. `0` = no steering. |
| `--axis-report` | off | Print the diagnostics of §7 and exit without searching. |
| `--axis-pairs <file>` | — | Read several pairs, one `left vs right` per line. |

`--axis-beta`, not `--beta`: `--alpha` and `--lambda` are already taken by
`weighted` and `mmr`, and a bare `--beta` beside them tells the reader nothing
about which method it belongs to.

### 6.2 Sketch

Grounded in the existing helpers — `embed` and `get_model` come from
`loadmilvus`, as they already do at `search.py:59`.

```python
# Axis directions are per (model, pairs) and never change within a run; embedding
# them twice per query would double the cost of the cheapest part of the method.
_AXIS_CACHE = {}


def axis_direction(model, pairs):
    """A named contrast as a unit direction, plus how much to trust it.

    Returns (d, explained) where `explained` is the share of variance taken by
    the top principal component of the normalized pair differences. Bolukbasi et
    al. (2016) construct a concept direction this way rather than from a single
    difference: wording peculiar to any one pair varies across pairs and lands in
    the lower components, so PCA keeps only what the pairs agree on.

    `explained` is the honest part. A real direction dominates its spectrum; a
    contrast that is not a single direction in this space produces a flat one,
    and no choice of beta fixes that. With one pair there is no spectrum to read
    and it returns 1.0, which means "unknown", not "certain".
    """
    import numpy as np

    key = (id(model), tuple(pairs))
    if key in _AXIS_CACHE:
        return _AXIS_CACHE[key]

    diffs = []
    for left, right in pairs:
        a, b = np.asarray(embed(model, [left, right]), dtype="float32")
        if float(a @ b) > 0.95:               # the poles embed to the same point;
            continue                          # their difference is noise
        d = a - b
        diffs.append(d / np.linalg.norm(d))   # unit, so no pair dominates by scale

    if not diffs:
        raise SystemExit(
            "The two sides of every axis embed to nearly the same vector, so "
            "there is no contrast to steer along. Rephrase the poles so they "
            "differ in more than a word.")

    matrix = np.stack(diffs)
    if len(diffs) == 1:
        result = (matrix[0], 1.0)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(len(diffs), 8)).fit(matrix)
        top = pca.components_[0]
        result = (top / np.linalg.norm(top), float(pca.explained_variance_ratio_[0]))

    _AXIS_CACHE[key] = result
    return result


def axis_steer(candidates, d, beta, limit):
    """Re-rank a dense pool by cosine plus a weighted projection onto the axis.

    Vectors are L2-normalized, so `d @ v` is the signed cosine of the chunk to
    the axis: positive toward the left pole, negative toward the right. Both
    terms are therefore on the same [-1, 1] scale, which is what makes a single
    beta meaningful across queries -- unlike `weighted`, which has to min-max
    normalize BM25 against cosine before it can add them.

    `axis` is reported alongside `score` so the caller can show why a chunk moved.
    """
    import numpy as np

    scored = []
    for c in candidates:
        projection = float(d @ np.asarray(c["vector"], dtype="float32"))
        scored.append({**c, "axis": projection,
                       "score": c["score"] + beta * projection})
    scored.sort(key=lambda r: -r["score"])
    return scored[:limit]
```

Wiring: in `search()`, after the chosen method returns, apply `axis_steer` when
an axis was given. Methods that already carry vectors (`dense`, `mmr`) need no
extra fetch; `lexical` and `tfidf` return no vectors and would need one query to
get them, which is the one place the modifier is not free.

### 6.3 Where the axis definition lives

**Not in `criteria.md`.** That file was deliberately stripped of comments twice
and is *clustering input*, read by `customcluster.py` at run time. An axis is
query-time state that changes per question. CLI flags, with `--axis-pairs`
pointing at a scratch file for a set worth keeping.

## 7. Validation — do this before building the retriever

The project's documented standard is that claims in the docs are measured, not
asserted, and there is a precedent for why: on 2026-08-20 a plausible fix to the
thin-cluster probe score was implemented, measured over 25 queries, found to
demote the *correct* small cluster, and reverted. An unmeasured axis feature is
exactly that shape of mistake.

Three tests, cheapest first. Each can kill the idea on its own.

**1. Spectrum test (no retrieval, seconds).** Write 5–8 pairs for one contrast,
build the direction, read `explained`. A dominant top component means the
contrast is a direction in this space. A flat spectrum means it is not — stop.

**2. Extremes test (one corpus read, no ranking).** Project every stored chunk
onto `d` and print the top 5 and bottom 5:

```powershell
.venv\Scripts\python.exe search.py --axis "a table of figures vs a narrative discussion" --axis-report
```

Read them. If the extremes are recognisably the two poles you named, the axis is
real. If they read as two *topics*, the axis has found subject matter again and
this is the removed clustering mode wearing a hat.

**3. Retrieval test (the 25-query protocol).** Only if 1 and 2 pass. Fix a query
set, run each query with `--axis-beta 0` and with a swept beta, and record
whether the intended kind of passage rises. Sweep beta over `{0.1, 0.2, 0.3,
0.5, 1.0}` and look for a *plateau* rather than a peak — a setting that only
works in a narrow band is a setting that will not survive a different corpus.

**Candidate axes for the current corpus**, in descending order of plausibility:

- `a table of figures vs a narrative discussion` — the report is full of both,
  and the OCR table path (§ chunk_text) makes table rows textually distinctive,
  so this one has the best chance
- `what could go wrong vs how it is mitigated` — the risk section states both
- `forward-looking statements vs historical results` — the report separates them
  explicitly, and the distinction is grammatical as much as topical

**Do not test "formal vs casual" on this corpus.** A statutory annual securities
report is uniformly formal; there is no variance for the axis to find, and a
negative result would say nothing about the method.

## 8. Known weaknesses

- **`beta` has no principled setting.** Both terms are cosines on `[-1, 1]`,
  which is better than `weighted`'s incommensurable BM25-vs-cosine problem, but
  the right *ratio* is still empirical, per axis and per corpus. Rocchio's
  weights have been empirical since 1971; ActAdd's coefficient needs per-case
  tuning. This is the state of the art, not an oversight in this design.
- **Two phrases are a weak estimator.** §4.2 mitigates it; §4.3 replaces it.
- **Pooling may average the contrast away.** See §9.
- **Silent degradation.** A meaningless axis produces confidently reordered
  results, not an error. Only the §7 tests catch that, and they must be re-run
  per axis, not once for the feature.
- **The corpus is close to a worst case.** One document, strongly sectioned,
  uniform register — precisely where topical methods already work. A negative
  result here should be recorded as *inconclusive for the method*, not as proof
  against it.
- **Opportunity cost.** The CLIP text+image objective — the original stated goal
  of the POC — is still unstarted.

## 9. What the literature does and does not support

The honest gap, stated plainly so it does not have to be rediscovered:

**Bolukbasi et al.** and the vector-arithmetic tradition behind it (Mikolov et
al., 2013 [[5]](#references), the `king - man + woman` offset) concern **word**
embeddings. **TCAV** and **ActAdd** [[4]](#references) concern **internal
activations** of a neural network. None of the four concerns sentence-embedding
retrieval, which is what bge-m3 produces.

Applying them to 1024-dim pooled sentence vectors is an argument **by analogy,
not a transferred result.** The specific risk the analogy hides: these methods
operate on representations of a word or a token position, while a chunk
embedding is *pooled over 600 characters of text*. A contrast that is a clean
direction for a word may be averaged into insignificance across a whole
paragraph — a failure mode none of these papers would warn about, because none
of them pools.

What the literature does give, concretely:

| Source | What it contributes here |
|---|---|
| Rocchio (1971) | The query-shift form, and 55 years of evidence that the weights are empirical |
| Mikolov et al. (2013) | That a difference of embeddings is a usable direction at all |
| Bolukbasi et al. (2016) | The multi-pair PCA construction **and** the variance diagnostic of §7.1 |
| Kim et al. (2018) | The stronger example-based construction to fall back on |
| Turner et al. (2023) | That a phrase-pair contrast plus a scaling coefficient works with no training or labelled data |

ActAdd is the closest analogue in spirit: it takes the difference of activations
on a contrast prompt pair ("Love" vs "Hate"), adds it back scaled by a
coefficient, and requires no labelled data, no backward pass, and no learned
encoder or classifier. That is this proposal's shape almost exactly — at a
different layer of a different kind of model.

## 10. Recommendation

Build §7.1 and §7.2 **first**, as a standalone `--axis-report` path, and test the
three candidate axes of §7. Only build the re-ranker if an axis survives both.

If none survives, write the result into the report as a second, now *measured*
dead end alongside the 2026-08-12 removal. That is a good outcome: it converts a
rejected-by-intuition decision into a rejected-by-measurement one, and it closes
the question rather than leaving it to be reopened a third time.

---

## References

1. **Bolukbasi, T., Chang, K.-W., Zou, J., Saligrama, V., & Kalai, A. (2016).**
   *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word
   Embeddings.* Advances in Neural Information Processing Systems 29 (NeurIPS
   2016).
   <http://papers.neurips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf>
   — Source of the multi-pair PCA construction (§4.2) and of the
   variance-explained diagnostic (§7.1). The paper identifies a "gender
   direction" from the top principal component of ten normalized pair
   differences, and reports that for a real direction the top component explains
   substantially more variance than any other.

2. **Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viégas, F., &
   Sayres, R. (2018).** *Interpretability Beyond Feature Attribution:
   Quantitative Testing with Concept Activation Vectors (TCAV).* Proceedings of
   the 35th International Conference on Machine Learning (ICML 2018), PMLR 80.
   <https://proceedings.mlr.press/v80/kim18d.html>
   — Source of the classifier-normal construction (§4.3). A CAV is the normal of
   a linear classifier separating examples of a concept from random ones;
   alignment is the directional derivative along it.

3. **Rocchio, J. J. (1971).** *Relevance Feedback in Information Retrieval.* In
   G. Salton (ed.), *The SMART Retrieval System: Experiments in Automatic
   Document Processing*, Prentice-Hall, pp. 313–323.
   Summary: <https://link.springer.com/rwe/10.1007/978-0-387-39940-9_932>
   — The IR ancestor of the query-shift variant (§5A):
   `q' = alpha*q0 + beta*(relevant centroid) - gamma*(non-relevant centroid)`,
   with weights that "need to be set empirically."

4. **Turner, A. M., Thiergart, L., Leech, G., Udell, D., Mini, U., &
   MacDiarmid, M. (2023).** *Activation Addition: Steering Language Models
   Without Optimization.* arXiv:2308.10248. Since retitled *Steering Language
   Models With Activation Engineering*.
   <https://arxiv.org/abs/2308.10248>
   — The closest modern statement of the formulation: a steering vector from the
   difference of activations on a contrast prompt pair, added back scaled by a
   coefficient, with no labelled data and no backward pass.

5. **Mikolov, T., Yih, W.-t., & Zweig, G. (2013).** *Linguistic Regularities in
   Continuous Space Word Representations.* NAACL-HLT 2013, pp. 746–751.
   — The origin of the offset/analogy property that everything above rests on.

Behind methods already implemented here, for context:

6. **Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).** *Reciprocal Rank
   Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR
   2009. — The source of `RRF_K = 60`, and the one paper `search.py` already
   cites in a comment (`search.py:77`).

7. **Carbonell, J., & Goldstein, J. (1998).** *The Use of MMR, Diversity-Based
   Reranking for Reordering Documents and Producing Summaries.* SIGIR 1998. —
   Behind `mmr_search`, which is not currently attributed in the code. Worth a
   one-line comment there, matching the RRF one.
