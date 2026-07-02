"""
Extract text from documents with PyMuPDF and feed it into the Milvus pipeline.

PyMuPDF (imported as `fitz`) is the C-backed MuPDF parser — fast, great text
extraction, one light wheel. See pdf-extractors.md for why it's the default.

This is the front of the pipeline: files -> text -> chunks -> (input.md | Milvus).
The common case is to drop documents into the `fileinput/` folder and ingest the
whole folder in one run:

    python extractpdf.py --store        # extract every file in fileinput/ -> Milvus

Each function does one thing and returns plain objects, so a UI layer can call
them independently:

    from extractpdf import extract_text, chunk_text
    text   = extract_text("file.pdf")
    chunks = chunk_text(text)          # each chunk fits the VARCHAR(2048) field

Milvus stores `text` as VARCHAR(2048) (see loadmilvus.ensure_collection), so
extracted text is chunked to stay under that limit before it's embedded/stored.
Every chunk also carries its source filename, stored alongside the vector in the
collection's dynamic `source` field so you can tell which document it came from.

PyMuPDF opens more than PDF (XPS/EPUB/MOBI/FB2/CBZ/TXT), so those are ingested
too. Scanned/image-only pages have no text layer and extract empty — that needs
OCR, which is out of scope here (see the OCR add-on note in pdf-extractors.md).

Prerequisites:
    pip install pymupdf            # or: pip install -r requirements.txt

Run:
    python extractpdf.py                          # preview chunks from fileinput/
    python extractpdf.py --store                  # embed + store the whole folder
    python extractpdf.py --store --model bge-m3
    python extractpdf.py file.pdf                 # a single file instead of the folder
    python extractpdf.py file.pdf --out input.md  # write chunks as documents
"""

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# Sentence boundary: end punctuation (. ! ?) followed by whitespace and the start
# of a new sentence (optional opening quote/bracket, then a capital or digit).
# Requiring a capital/digit after the space avoids splitting on decimals ("2.5")
# and most lowercase abbreviations ("e.g. the ..."); it's not perfect around
# "Fig. 3" / "et al." but keeps chunks from cutting mid-sentence, which matters
# far more for embedding/clustering quality than the odd false split.
_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=["\'(\[]?[A-Z0-9])')

# Default folder ingested when no path is given. Drop documents here.
DEFAULT_INPUT_DIR = "fileinput"

# File types MuPDF can open as documents. Anything else in the folder is skipped.
SUPPORTED_EXTS = {".pdf", ".xps", ".epub", ".mobi", ".fb2", ".cbz", ".txt"}

# Chunk sizing. Smaller chunks make each one a tighter semantic unit (roughly a
# few sentences about one idea), so its embedding is topically pure and clusters
# come out intuitive / single-topic. Milvus stores `text` as VARCHAR(2048), so
# these stay well under that. OVERLAP repeats a sentence or two between
# consecutive chunks so context spanning a boundary still embeds together.
DEFAULT_MAX_CHARS = 600
DEFAULT_OVERLAP = 100


def extract_pages(path):
    """Return [(page_number, text), ...] for every page (page numbers 1-based)."""
    doc = fitz.open(path)
    try:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()


def extract_text(path):
    """Return the whole PDF's plain text, pages joined by blank lines."""
    return "\n\n".join(text for _, text in extract_pages(path))


def split_sentences(text):
    """Collapse PDF line-wraps and split text into a list of sentence strings.

    PyMuPDF returns text with hard line breaks wherever a line wrapped in the
    PDF, so a single sentence is sprinkled with `\\n`. Collapsing all whitespace
    to single spaces rejoins those, then we split on sentence-ending punctuation
    (`_SENTENCE_BOUNDARY`). The result is the unit we pack into chunks, so chunk
    edges land at sentence ends instead of cutting mid-sentence.
    """
    text = " ".join(text.split())                 # rejoin wrapped lines -> one stream
    if not text:
        return []
    return _SENTENCE_BOUNDARY.split(text)


def _overlap_sentences(sentences, overlap):
    """Return the trailing whole sentences whose length is ~<= overlap chars.

    Carries context from the end of one chunk into the start of the next without
    ever beginning on a mid-word fragment. Always keeps at least the last
    sentence so consecutive chunks share a real boundary sentence.
    """
    if overlap <= 0:
        return []
    tail, total = [], 0
    for s in reversed(sentences):
        if tail and total + len(s) > overlap:
            break
        tail.insert(0, s)
        total += len(s) + 1
    return tail


def _split_oversized(piece, max_chars):
    """Break a single sentence that exceeds max_chars into <=max_chars pieces.

    Splits on whitespace so words stay intact; a lone word longer than max_chars
    (rare — URLs, hashes) is hard-cut as a last resort. Only hit by the
    occasional monster "sentence" (a table row or equation with no punctuation).
    """
    out, cur = [], ""
    for word in piece.split():
        if len(word) > max_chars:                 # word alone too big: hard-cut it
            if cur:
                out.append(cur)
                cur = ""
            for j in range(0, len(word), max_chars):
                out.append(word[j:j + max_chars])
            continue
        candidate = f"{cur} {word}".strip()
        if len(candidate) > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        out.append(cur)
    return out


def chunk_text(text, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    """Split text into <=~max_chars chunks that never cut a sentence in half.

    Whole sentences are packed together until the next one would overflow
    max_chars; the chunk is then flushed and the next one seeded with the
    previous chunk's trailing sentence(s) for `overlap` chars of context. A lone
    sentence longer than max_chars is word-split as a fallback. Keeping edges at
    sentence boundaries means each chunk is a coherent semantic unit, so its
    embedding (and the clusters built from it) reflect one topic, not a fragment.
    """
    # Split into sentence units, word-splitting any single one that's too long.
    sentences = []
    for s in split_sentences(text):
        if len(s) > max_chars:
            sentences.extend(_split_oversized(s, max_chars))
        else:
            sentences.append(s)

    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur))
            cur = _overlap_sentences(cur, overlap)   # carry whole sentences over
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def extract_chunks(path, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    """File path -> list of chunk strings ready to embed/store."""
    return chunk_text(extract_text(path), max_chars=max_chars, overlap=overlap)


def find_documents(folder):
    """Return the supported document files directly in `folder`, sorted by name."""
    base = Path(folder)
    if not base.is_dir():
        raise SystemExit(f"Input folder not found: {folder}")
    return sorted(
        p for p in base.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def extract_records(path, max_chars=DEFAULT_MAX_CHARS, overlap=DEFAULT_OVERLAP):
    """Extract chunks from a file OR every supported file in a folder.

    Returns a list of {"text": chunk, "source": filename} records. A folder is
    walked one level deep over SUPPORTED_EXTS; files that yield no text (e.g.
    scanned/image-only PDFs) are reported and skipped. `source` is the file name
    so each chunk stays traceable to its origin once stored.
    """
    target = Path(path)
    files = [target] if target.is_file() else find_documents(target)
    if not files:
        raise SystemExit(
            f"No supported documents in {path} "
            f"(looked for: {', '.join(sorted(SUPPORTED_EXTS))}).")

    records = []
    for f in files:
        try:
            chunks = extract_chunks(f, max_chars=max_chars, overlap=overlap)
        except Exception as e:                    # one bad file shouldn't sink the batch
            print(f"  ! skipped {f.name}: {e}")
            continue
        if not chunks:
            print(f"  ! {f.name}: no text extracted (scanned/image-only? needs OCR)")
            continue
        print(f"  + {f.name}: {len(chunks)} chunks")
        records.extend({"text": c, "source": f.name} for c in chunks)
    return records


def write_input_md(records, path="input.md"):
    """Write chunk records as bullet documents under a '# Documents' heading.

    Produces a file loadmilvus.py can read directly. Newlines inside a chunk are
    flattened to spaces so each chunk stays one bullet line (load_inputs treats
    every non-empty line as a separate document). The `source` is dropped here —
    input.md is a plain text-only format — but is kept when storing to Milvus.
    """
    lines = ["# Documents", ""]
    lines += [f"- {' '.join(r['text'].split())}" for r in records]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} document chunks to {path}.")


def show_chunks(records, preview=160):
    """Print a short preview of each chunk (index, source, length, leading text)."""
    print(f"\nExtracted {len(records)} chunks total:")
    for i, r in enumerate(records):
        head = " ".join(r["text"].split())[:preview]
        ellipsis = "..." if len(r["text"]) > preview else ""
        print(f"\n[{i}] {r['source']} ({len(r['text'])} chars)")
        print(f"    {head}{ellipsis}")


def parse_value_arg(argv, flag):
    """Read the value following `flag` in argv, or None if the flag is absent."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 >= len(argv):
            raise SystemExit(f"{flag} needs a value")
        return argv[i + 1]
    return None


def parse_path_arg(argv, default):
    """Return the first positional (non-flag) argument, or `default` if none."""
    flags_with_values = {"--out", "--max-chars", "--overlap", "--model"}
    skip = False
    for arg in argv[1:]:
        if skip:
            skip = False
            continue
        if arg in flags_with_values:
            skip = True
            continue
        if arg.startswith("--"):
            continue
        return arg
    return default


def store_records(records, model_name):
    """Embed every record's text and store it (with its source) in Milvus.

    Mirrors loadmilvus's storing flow but adds a per-row `source` field (the
    collection has dynamic fields enabled, so it's stored alongside text +
    vector). reset=True rebuilds the collection from this batch, matching
    loadmilvus's "input is the single source of truth" behavior (no duplicates).
    """
    # Imported lazily so plain extraction/preview doesn't pull in torch/pymilvus.
    from loadmilvus import connect, embed, ensure_collection, get_dim, get_model

    texts = [r["text"] for r in records]
    model = get_model(model_name)
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embed(model, texts)

    client = connect()
    ensure_collection(client, dim=get_dim(model), reset=True)
    rows = [
        {"text": r["text"], "embedding": emb.tolist(), "source": r["source"]}
        for r, emb in zip(records, embeddings)
    ]
    result = client.insert(collection_name="documents", data=rows)
    client.flush("documents")
    print(f"Inserted {result['insert_count']} chunks "
          f"from {len({r['source'] for r in records})} files.")


def main():
    """Extract files into chunks, then preview / write to input.md / store."""
    # Extracted text can hold characters outside the terminal's legacy code page
    # (e.g. math symbols like pi from a PDF). Don't let a preview print crash the
    # run on Windows consoles (cp1252); replace anything unencodable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    path = parse_path_arg(sys.argv, default=DEFAULT_INPUT_DIR)
    if not Path(path).exists():
        raise SystemExit(f"Path not found: {path}")

    max_chars = int(parse_value_arg(sys.argv, "--max-chars") or DEFAULT_MAX_CHARS)
    overlap = int(parse_value_arg(sys.argv, "--overlap") or DEFAULT_OVERLAP)

    where = f"folder {path}/" if Path(path).is_dir() else path
    print(f"Extracting text from {where} with PyMuPDF...")
    records = extract_records(path, max_chars=max_chars, overlap=overlap)
    if not records:
        raise SystemExit("No text extracted from any file.")
    show_chunks(records)

    out = parse_value_arg(sys.argv, "--out")
    if out:
        write_input_md(records, out)

    if "--store" in sys.argv:
        from loadmilvus import parse_model_arg
        store_records(records, parse_model_arg(sys.argv))
        print("\nDone. Chunks embedded and stored in Milvus.")
    elif not out:
        print("\nPreview only. Re-run with --out input.md to save, or "
              "--store to embed into Milvus.")


if __name__ == "__main__":
    main()
