"""Ingestion: extract, chunk and store documents, with the full OCR flag set.

Both ingest paths the repo has. `extractpdf` is the incremental one -- the
collection is its own embedding cache, so re-running costs only what changed --
and `loadmilvus` rebuilds wholesale from input.md. They overwrite each other,
which is a property of the pipeline rather than of this page, so the page says
so where you choose.
"""

import time
from pathlib import Path

import streamlit as st

import extractpdf
import loadmilvus

from ..components import guard, log_panel, model_picker, result_card
from ..resources import collection_info, data_version, invalidate
from ..runner import call_logged


def _ocr_options():
    """The `--ocr*` flags extract_records forwards to extract_pages."""
    mode = st.selectbox(
        "OCR", extractpdf.OCR_MODES,
        index=extractpdf.OCR_MODES.index(extractpdf.DEFAULT_OCR_MODE),
        help="auto = OCR only pages with too little text (scanned PDFs). "
             "never = text layer only. always = OCR every page.")
    options = {"ocr": mode}
    if mode == "never":
        return options

    left, right = st.columns(2)
    options["backend"] = left.text_input("Backend", extractpdf.DEFAULT_OCR_BACKEND)
    options["dpi"] = right.number_input("Render DPI", 72, 900,
                                        extractpdf.DEFAULT_OCR_DPI, 25)
    options["min_chars"] = left.number_input(
        "Min chars before OCR", 0, 10_000, extractpdf.DEFAULT_OCR_MIN_CHARS, 8,
        help="`auto` only: a page with fewer characters than this is treated as "
             "having no text layer.")
    options["quant"] = right.selectbox("Quantisation",
                                       ["auto", "none", "4bit", "8bit"])
    options["prompt"] = st.text_input("Prompt", extractpdf.DEFAULT_OCR_PROMPT)
    options["url"] = left.text_input("Server URL", extractpdf.DEFAULT_OCR_URL,
                                     help="Used by server-style backends.")
    options["figures"] = right.checkbox(
        "Parse figures", extractpdf.DEFAULT_OCR_FIGURES,
        help="Also run the model over detected figure regions.")
    return options


def _save_uploads(uploads, folder):
    """Write uploaded files into `folder` so extraction sees them as normal files.

    They are written to the real input folder rather than a temp dir on purpose:
    the collection mirrors that folder, so a file that only ever existed in a
    temp dir would be deleted from the collection by the next ingest as 'no
    longer present in the input'.
    """
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for upload in uploads:
        path = target / upload.name
        path.write_bytes(upload.getbuffer())
        written.append(path.name)
    return written


def _render_outcome(log, error, records=None, stored=None):
    """Show the log and the result of a write, whichever way it went."""
    if error:
        st.error(error)
    elif stored is not None:
        st.success(f"Stored. {stored} new chunk(s) embedded and inserted.")
    log_panel(log, "Ingest log", expanded=True)
    if records:
        st.caption(f"{len(records)} chunk(s) extracted from "
                   f"{len({r['source'] for r in records})} file(s).")
        for i, record in enumerate(records[:25], start=1):
            result_card(i, {**record, "score": 0.0}, preview=300)
        if len(records) > 25:
            st.caption(f"…and {len(records) - 25} more.")


def _is_corpus_file(path):
    """Whether `path` is a document the input folder exists to hold.

    A dotfile is not. The one that matters is `.gitkeep`: the folder's contents
    are git-ignored but the folder itself is tracked, and that empty file is the
    only reason the folder survives a clone. It is not part of the corpus, it is
    what holds the corpus's home open.

    This is one predicate on purpose, asked by both the listing and the delete,
    so the two cannot drift apart. `.gitkeep` is not merely hidden from the
    checklist -- it is undeletable through this page even if a caller assembles
    its own list and hands it straight over.
    """
    return path.is_file() and not path.name.startswith(".")


def _files_in(folder):
    """Every corpus file directly in `folder`, ingestable or not.

    Top level only, which is also all extractpdf walks -- a subfolder's contents
    are not shown because nothing in this pipeline reads them, so deleting them
    from here would change nothing about an ingest.
    """
    base = Path(folder)
    if not base.is_dir():
        return []
    return sorted((path for path in base.iterdir() if _is_corpus_file(path)),
                  key=lambda path: path.name.lower())


def _file_label(path):
    """`name - size`, flagging the files an ingest would pass over.

    Worth saying per row: the list is every file in the folder, but only some of
    them are why the folder exists, and `not ingested` is the difference between
    deleting a document and deleting a stray note.
    """
    megabytes = path.stat().st_size / 1e6
    ingested = path.suffix.lower() in extractpdf.SUPPORTED_EXTS
    return f"{path.name} · {megabytes:.1f} MB" + ("" if ingested else " · not ingested")


def _delete_files(paths, generation):
    """Delete `paths`, hand the outcome to the next run, and rerun.

    The outcome goes through session state rather than being rendered here: the
    checklist above was drawn before the deletion and would otherwise carry on
    offering files that no longer exist.
    """
    removed, failed = [], []
    for path in paths:
        # Asked again here, not just in _files_in. Nothing in this page can pass
        # a dotfile in, so this never fires -- it is the guarantee rather than
        # the mechanism, and it is what makes "`.gitkeep` is never deleted" a
        # property of the delete itself instead of of one filter upstream of it.
        if not _is_corpus_file(path):
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError as error:      # open elsewhere, read-only, gone already
            failed.append(f"{path.name}: {error}")
    st.session_state["dropped_files"] = (removed, failed)
    # A new generation for the checkbox keys. They are keyed by filename, so
    # without this a tick would survive the file: re-adding a deleted name later
    # would bring it back already selected for deletion.
    st.session_state["drop_generation"] = generation + 1
    st.session_state["drop_all_armed"] = False
    st.rerun()


def _drop_documents(folder):
    """Tick files out of the input folder and delete them from disk.

    Ticking is the confirmation for the selected delete: nothing goes that you
    did not name, and the button says how many. `Drop all` is the exception and
    takes two clicks, because it is the one button here whose target is
    everything and whose cost is unrecoverable.

    Unrecoverable is meant literally. These are ordinary deletes, not a move to
    the recycle bin, and files uploaded through this page were written straight
    into this folder and exist nowhere else.

    It reaches Milvus too, one step later. The folder is the source of truth for
    an incremental ingest, so the next "Extract and store" reads whatever you
    deleted as no longer present in the input and drops its chunks. Nothing
    leaves the collection here, which is why this invalidates no cache: Milvus
    is untouched until you ingest again.
    """
    outcome = st.session_state.pop("dropped_files", None)
    if outcome:
        removed, failed = outcome
        if removed:
            st.success(f"Deleted {len(removed)} file(s): {', '.join(removed)}")
        for message in failed:
            st.error(message)

    files = _files_in(folder)
    if not files:
        st.caption(f"`{folder}` has no files in it.")
        return

    st.caption("Deletes from disk, not from Milvus. Chunks already stored stay "
               "until the next **Extract and store**, which reads this folder "
               "as the source of truth and drops what is no longer in it. "
               "There is no undo.")

    generation = st.session_state.get("drop_generation", 0)
    # A fixed height makes the list scroll instead of pushing the buttons off
    # the page: the folder is a corpus, and a corpus can be hundreds of files.
    with st.container(height=240, border=True):
        picked = [path for path in files
                  if st.checkbox(_file_label(path),
                                 key=f"drop::{generation}::{path.name}")]

    left, right = st.columns(2)
    if left.button(f"Delete {len(picked)} selected", type="primary",
                   disabled=not picked, width="stretch"):
        _delete_files(picked, generation)
    if right.button(f"Drop all {len(files)}", width="stretch"):
        st.session_state["drop_all_armed"] = True
        st.rerun()

    if st.session_state.get("drop_all_armed"):
        st.warning(f"Delete **all {len(files)} file(s)** in `{folder}`, "
                   f"ticked or not?")
        yes, no = st.columns(2)
        if yes.button("Yes, delete all", type="primary", width="stretch"):
            _delete_files(files, generation)
        if no.button("Cancel", width="stretch"):
            st.session_state["drop_all_armed"] = False
            st.rerun()


def _extract(folder, max_chars, overlap, ocr):
    """Run the extraction behind a progress bar. Returns (records, log, error).

    A spinner was the wrong shape for this step. extractpdf reports progress the
    way a CLI should -- an in-place page counter, a line per file -- and
    runner.call redirects stdout into a buffer, so all of it landed in the log
    and none of it on the screen. On a scanned PDF, where one page is seconds of
    GPU, that is a spinner and nothing else for minutes.

    The callbacks run on the script thread, because extract_records is called
    synchronously from here, so they can write to the widget directly with no
    queue or worker in between.
    """
    bar = st.progress(0.0, text="Reading the folder…")
    at = {"index": 0, "total": 1, "name": ""}
    drawn = [0.0]

    def on_file(index, total, path):
        at.update(index=index, total=total, name=path.name)
        bar.progress((index - 1) / total, text=f"{index}/{total} · {path.name}")

    def on_page(page, pages):
        # Files set the coarse position and pages refine it, so a single long
        # document still shows movement instead of parking the bar at 1/2.
        # Throttled because a born-digital PDF walks its text layer faster than
        # anything can usefully be read, and every call is a browser update.
        now = time.perf_counter()
        if page < pages and now - drawn[0] < 0.1:
            return
        drawn[0] = now
        done = (at["index"] - 1) / at["total"]
        bar.progress(min(done + (page / max(pages, 1)) / at["total"], 1.0),
                     text=f"{at['index']}/{at['total']} · {at['name']} · "
                          f"page {page}/{pages}")

    try:
        return call_logged(extractpdf.extract_records, folder,
                           max_chars=max_chars, overlap=overlap,
                           on_file=on_file, on_page=on_page, **ocr)
    finally:
        bar.empty()


def _documents_tab(info):
    st.caption("Extract and chunk every supported file in a folder, then store "
               "only what Milvus does not already have.")

    folder = st.text_input("Input folder", extractpdf.DEFAULT_INPUT_DIR,
                           key="in_folder")
    uploads = st.file_uploader(
        "Add files", accept_multiple_files=True, key="in_uploads",
        type=[e.lstrip(".") for e in sorted(extractpdf.SUPPORTED_EXTS)],
        help=f"Saved into the input folder. Supported: "
             f"{', '.join(sorted(extractpdf.SUPPORTED_EXTS))}")
    if uploads and st.button("Save uploads to folder"):
        with guard("saving uploads"):
            names = _save_uploads(uploads, folder)
            st.success(f"Saved {len(names)} file(s): {', '.join(names)}")

    # Opened when there is an outcome to show: the panel reruns after deleting,
    # and a collapsed expander would swallow the confirmation it just produced.
    pending = st.session_state.get("dropped_files")
    with st.expander(f"Drop files from `{folder}`", expanded=bool(pending)):
        _drop_documents(folder)

    left, right = st.columns(2)
    max_chars = left.number_input("Target chunk size", 100, 2000,
                                  extractpdf.DEFAULT_MAX_CHARS, 50,
                                  help="Chunks never cut a sentence in half.")
    overlap = right.number_input("Overlap", 0, 1000, extractpdf.DEFAULT_OVERLAP, 10,
                                 help="Trailing characters repeated into the "
                                      "next chunk.")
    model_name = model_picker(info, "ingest")
    reset = st.checkbox("Rebuild from scratch (`--reset`)", key="in_reset",
                        help="Ignore the embedding cache and rebuild the "
                             "collection. Re-embeds everything.")

    with st.expander("OCR", expanded=False):
        ocr = _ocr_options()

    preview_col, store_col = st.columns(2)
    do_preview = preview_col.button("Preview chunks", width="stretch")
    do_store = store_col.button("Extract and store", type="primary",
                                width="stretch")

    if not (do_preview or do_store):
        return

    if not Path(folder).exists():
        st.error(f"`{folder}` does not exist.")
        return

    records, log, error = _extract(folder, int(max_chars), int(overlap), ocr)
    if error or not records:
        _render_outcome(log, error or f"No chunks extracted from `{folder}`.")
        return

    if do_preview:
        _render_outcome(log, None, records=records)
        return

    with st.spinner(f"Embedding and storing {len(records)} chunks…"):
        stored, store_log, store_error = call_logged(
            extractpdf.store_records, records, model_name, reset)
    invalidate()
    _render_outcome(log + store_log, store_error, stored=stored)


def _markdown_tab(info):
    st.caption("Every bullet under `# Documents` becomes one document. This path "
               "**rebuilds** the collection, replacing anything ingested from "
               "the documents folder.")

    path = st.text_input("Markdown file", loadmilvus.DEFAULT_INPUT, key="md_path")
    file = Path(path)
    if file.exists():
        # Keyed on the path for the same reason as the criterion editor in
        # views/clustering.py: a fixed key would keep the previous file's text
        # on screen after the path changes, and "Save file" would write it over
        # the new path.
        text = st.text_area("Contents", file.read_text(encoding="utf-8"),
                            height=260, key=f"md_text::{file}")
        if st.button("Save file"):
            file.write_text(text, encoding="utf-8")
            st.success(f"Wrote {path}.")
    else:
        st.warning(f"`{path}` does not exist.")
        return

    model_name = model_picker(info, "md")
    if not st.button("Embed and store", type="primary"):
        return

    def _store():
        documents, _ = loadmilvus.load_inputs(path)
        if not documents:
            raise SystemExit(f"No documents found in {path}. Add some under a "
                             f"'# Documents' heading.")
        model = loadmilvus.get_model(model_name)
        client = loadmilvus.connect()
        # reset=True mirrors loadmilvus.main: the file is the source of truth,
        # so each run rebuilds rather than appending duplicates.
        loadmilvus.ensure_collection(client, dim=loadmilvus.get_dim(model),
                                     reset=True)
        return loadmilvus.store(client, model, documents)

    with st.spinner("Embedding and storing…"):
        stored, log, error = call_logged(_store)
    invalidate()
    _render_outcome(log, error, stored=stored)


def render():
    st.title("Ingest")
    info = collection_info(data_version())
    if info.get("exists"):
        st.caption(f"Collection currently holds **{info['rows']:,}** chunks at "
                   f"**{info['dim']}** dimensions."
                   + (" Storing will clear existing cluster labels."
                      if info.get("clustered") else ""))

    documents, markdown = st.tabs(["Documents folder", "input.md"])
    with documents:
        with guard("ingestion"):
            _documents_tab(info)
    with markdown:
        with guard("ingestion"):
            _markdown_tab(info)
