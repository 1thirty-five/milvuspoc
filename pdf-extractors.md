# PDF Text/Content Extractors — Comparison

For a RAG ingestion pipeline: PDF → text (+ tables/images) → chunk → embed (bge-m3) → Milvus.
None are currently installed in `.venv`.

## Quick matrix

| Library | Text quality | Tables | Images | OCR (scanned) | Output | Speed | Deps | License |
|---|---|---|---|---|---|---|---|---|
| PyMuPDF | Excellent | Basic | Yes | No (pair w/ OCR) | text/dict/markdown | Very fast | 1 wheel | AGPL-3.0 |
| pdfplumber | Very good | Excellent | Coords only | No | text + table objs | Slow | pure-py | MIT |
| pypdf | OK | No | Basic | No | text | Fast | pure-py | BSD-3 |
| pdfminer.six | Good | Manual | No | No | layout tree | Slow | pure-py | MIT |
| pdftotext (poppler) | Very good | Layout-ish | No | No | text | Fast | poppler bin | GPL-2 |
| Unstructured | Good | Yes | Yes | Yes (Tesseract) | typed elements | Slow | Heavy | Apache-2 |
| Docling (IBM) | Excellent | Excellent | Yes | Yes | markdown/JSON | Slow (ML) | Heavy | MIT |
| Marker | Excellent | Very good | Yes | Yes | markdown | Slow (GPU) | Heavy | GPL/commercial |
| Camelot | — | Excellent | No | No | DataFrames | Medium | Ghostscript | MIT |
| LlamaParse | Excellent | Excellent | Yes | Yes | markdown/JSON | API latency | Cloud SDK | SaaS (paid) |

---

## PyMuPDF (`pymupdf`, imported as `fitz`)
- **What:** C-backed (MuPDF) parser. The fast all-rounder.
- **Strengths:** Excellent text extraction; pulls embedded **images** (raster + vector render); `page.get_text("dict")` gives blocks/spans/coords; can emit markdown; per-page rendering to PNG (useful for CLIP and OCR fallback).
- **Weaknesses:** Table extraction is basic (has `find_tables()` but not best-in-class); **AGPL-3.0** license — fine internally, a concern for redistribution/commercial.
- **Output:** plain text, structured dict, markdown, page images.
- **Install:** `pip install pymupdf`
- **Best for:** the default workhorse, especially since it covers the future text **+ image** goal in one dependency.

```python
import fitz
doc = fitz.open("file.pdf")
text = "\n".join(p.get_text() for p in doc)
images = [doc.extract_image(x[0]) for p in doc for x in p.get_images()]
```

## pdfplumber
- **What:** Layout-aware extractor built on pdfminer.six.
- **Strengths:** Best-in-class **table** detection with precise cell/word bounding boxes; great when column structure must survive.
- **Weaknesses:** Slow on big docs; no real image extraction; no OCR.
- **Output:** text, `page.extract_tables()`, word/char objects with coordinates.
- **Install:** `pip install pdfplumber`
- **Best for:** table-heavy financial/scientific PDFs where cells must stay intact.

```python
import pdfplumber
with pdfplumber.open("file.pdf") as pdf:
    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    tables = [t for p in pdf.pages for t in p.extract_tables()]
```

## pypdf
- **What:** Pure-Python, the lightweight standard (successor to PyPDF2).
- **Strengths:** Tiny, BSD license, no native deps, fine for simple born-digital PDFs; also merge/split/encrypt.
- **Weaknesses:** Weak on complex multi-column layouts; no tables; minimal image handling; no OCR.
- **Output:** plain text per page.
- **Install:** `pip install pypdf`
- **Best for:** clean license + minimal footprint when PDFs are simple.

```python
from pypdf import PdfReader
text = "\n".join(pg.extract_text() or "" for pg in PdfReader("file.pdf").pages)
```

## pdfminer.six
- **What:** Low-level pure-Python layout engine (pdfplumber sits on it).
- **Strengths:** Fine-grained control of layout analysis (LTTextBox/LTChar), good text fidelity.
- **Weaknesses:** Verbose API; slow; tables/images are manual; no OCR.
- **Output:** layout object tree, or `extract_text()`.
- **Install:** `pip install pdfminer.six`
- **Best for:** custom layout logic when you need raw positions and MIT license.

## pdftotext (Poppler)
- **What:** Python binding over the Poppler `pdftotext` CLI.
- **Strengths:** Fast, strong text with `-layout` preserving columns.
- **Weaknesses:** Needs the **poppler** binary installed (extra system dep on Windows); GPL; no images/tables/OCR.
- **Install:** `pip install pdftotext` + poppler binaries.
- **Best for:** high-throughput plain-text extraction when you can ship poppler.

## Unstructured (`unstructured`)
- **What:** High-level RAG ingestion toolkit; partitions docs into typed **elements**.
- **Strengths:** Returns Title / NarrativeText / ListItem / **Table** / Image elements — ideal for structure-aware chunking; built-in OCR (Tesseract) and image extraction; handles many file types beyond PDF.
- **Weaknesses:** Heavy dependency tree; slower; quality varies by "strategy" (`fast` vs `hi_res`).
- **Output:** list of element objects (→ easy chunk boundaries).
- **Install:** `pip install "unstructured[pdf]"` (+ Tesseract/poppler for hi_res).
- **Best for:** turnkey RAG ingestion where clean chunk boundaries matter more than footprint.

```python
from unstructured.partition.pdf import partition_pdf
els = partition_pdf("file.pdf", strategy="hi_res")  # OCR + tables + images
```

## Docling (IBM)
- **What:** ML document-understanding pipeline → structured markdown/JSON.
- **Strengths:** Excellent layout + **table** structure recovery; reading-order aware; markdown output chunks beautifully; OCR support; MIT license.
- **Weaknesses:** Downloads ML models (big, slow cold start); heavier compute.
- **Output:** markdown, JSON, document tree.
- **Install:** `pip install docling`
- **Best for:** best retrieval quality via clean structured markdown, when deps/latency are acceptable.

## Marker
- **What:** Deep-learning PDF → **markdown** converter.
- **Strengths:** High-fidelity markdown incl. equations, tables, images; great for academic PDFs.
- **Weaknesses:** Large models, **GPU** recommended; licensing has commercial restrictions; heaviest setup.
- **Install:** `pip install marker-pdf`
- **Best for:** highest-fidelity markdown when you have GPU and license fits.

## Camelot (tables only)
- **What:** Dedicated table extractor (lattice/stream modes).
- **Strengths:** Very accurate tables → pandas DataFrames.
- **Weaknesses:** Tables only (no prose); needs Ghostscript; struggles on borderless tables.
- **Install:** `pip install camelot-py[cv]` + Ghostscript.
- **Best for:** pairing with a text extractor when tables are the priority. (Alt: **Tabula**, Java-based.)

## LlamaParse (cloud, paid)
- **What:** Hosted RAG-optimized parser (LlamaIndex).
- **Strengths:** Excellent markdown/JSON, tables, images, OCR — minimal local setup.
- **Weaknesses:** **Cloud SaaS** (data leaves the machine), API key, per-page cost, network latency.
- **Install:** `pip install llama-parse` (needs API key).
- **Best for:** quick high-quality results without local ML, if cloud is acceptable.

---

## OCR add-on (for scanned/image-only PDFs)
Pure extractors return empty text on scans — add OCR:
- **Tesseract** via `pytesseract` (+ render pages with PyMuPDF) — free, offline, needs the Tesseract binary.
- **OCRmyPDF** — adds a text layer to scanned PDFs in place.
- **Cloud OCR** — AWS Textract / Azure Document Intelligence / Google Document AI — best accuracy + tables on scans, but paid + cloud.

## Recommendation for this POC
- **Default:** PyMuPDF — fast, great text, extracts images (covers the future CLIP/mixed-data goal) in one light dep. *Mind the AGPL note.*
- **If tables matter:** add pdfplumber (or Camelot).
- **If chunk quality matters most:** Docling for structure-aware markdown.
- **License-clean alternative:** pypdf (BSD) + Tesseract for images/scans.
- **Scanned docs:** add Tesseract as a fallback when a page has no text layer.

