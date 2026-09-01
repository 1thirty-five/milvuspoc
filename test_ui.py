"""Headless render + interaction test for the Streamlit UI.

Views are exercised through AppTest.from_function rather than by navigating
app.py: st.navigation builds its pages from callables, and AppTest.switch_page
only resolves file-based pages, so driving the real nav is not possible from the
harness. Rendering each view directly tests the same code with less indirection.

The `regressions` section at the end pins bugs that rendered and searched
perfectly well while being wrong, so the render and interaction passes above
could not see them. Each check names the behaviour, not the line it was fixed on.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

TIMEOUT = 300


def run_view(name):
    """Render one view through the driver script.

    from_function would run the view's source without its module globals
    (`st is not defined`), so the driver imports it properly instead.
    """
    import os
    os.environ["MILVUSUI_VIEW"] = name
    at = AppTest.from_file("test_ui_driver.py", default_timeout=TIMEOUT)
    at.run()
    return at


def count(at):
    return (len(at.button) + len(at.text_input) + len(at.selectbox)
            + len(at.checkbox) + len(at.slider) + len(at.number_input)
            + len(at.multiselect) + len(at.tabs))


def report(label, at, expect_title=None):
    excs = [f"{e.type}: {e.message}" for e in at.exception]
    errs = [str(e.value) for e in at.error]
    title = at.title[0].value if at.title else "(none)"
    bad = bool(excs) or (expect_title and title != expect_title)
    print(f"[{'FAIL' if bad else '  ok'}] {label:<14} title={title!r:<14} "
          f"widgets={count(at):<3} errors={len(errs)} exceptions={len(excs)}")
    for e in excs:
        print(f"          EXC {e[:500]}")
    for e in errs:
        print(f"          err {e[:220]}")
    return not bad


def check(label, condition, detail=""):
    print(f"[{'  ok' if condition else 'FAIL'}] {label:<22} {detail}")
    return bool(condition)


def check_cache_invalidation():
    """invalidate() must actually reach the st.cache_data caches.

    They are keyed on a version counter, and st.cache_data leaves
    underscore-prefixed arguments out of its hash key -- so naming that
    parameter `_version` silently pins every page to the collection as it was
    when the server booted, through any number of ingests and re-clusterings.
    Counting recomputes is the only way to see it: the pages still render.
    """
    import milvusui.resources as resources

    calls = []
    original = resources.hierarchicalsearch.has_cluster_names

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    resources.hierarchicalsearch.has_cluster_names = counting
    try:
        # Start on a version this process has not cached, or the first call is
        # served from the render pass above and proves nothing.
        resources.invalidate()
        resources.collection_info(resources.data_version())
        first = len(calls)
        resources.collection_info(resources.data_version())
        cached = len(calls)
        resources.invalidate()
        resources.collection_info(resources.data_version())
        after = len(calls)
    finally:
        resources.hierarchicalsearch.has_cluster_names = original

    return (check("cache hits", cached == first,
                  f"recomputes {first} -> {cached}")
            & check("invalidate recomputes", after == first + 1,
                    f"recomputes {cached} -> {after}"))


def check_criterion_editor_follows_path():
    """Pointing the criterion editor at another file must load that file.

    A Streamlit widget ignores its `value` argument once its key is in session
    state, so a fixed key kept the first file's text on screen -- and `Save
    criterion` then wrote those contents over whatever the path now pointed at.
    """
    other = Path(tempfile.gettempdir()) / "milvusui_criteria_probe.md"
    expected = "# Mode\nanchor\n\n# Labels\n- a: alpha\n- b: beta\n"
    other.write_text(expected, encoding="utf-8")
    try:
        at = run_view("clustering")
        before = at.text_area[0].value
        at.text_input(key="cc_path").set_value(str(other))
        at.run()
        shown = at.text_area[0].value
        loaded = check(
            "editor follows path", shown == expected,
            "loads the file the path names" if shown != before
            else "still showing the previous file")

        for button in at.button:
            if button.label == "Save criterion":
                button.click()
                at.run()
                break
        # Compared against the file's own contents, not against what the
        # box displayed: the bug was that Save wrote the *previous* file's
        # text here, which a `== shown` comparison would happily agree with.
        saved = check("save writes that file",
                      other.read_text(encoding="utf-8") == expected,
                      "no cross-file overwrite")
        return loaded & saved
    finally:
        other.unlink(missing_ok=True)


def check_ef_tracks_candidates():
    """HNSW ef must follow the candidate depth, not stick at its first value.

    ef below the depth caps recall at ef, so an ef frozen at 64 quietly makes a
    candidate depth of 500 a lie -- and nothing errors.
    """
    at = run_view("search")
    at.number_input(key="cand").set_value(500)
    at.run()
    widgets = [n for n in at.number_input if n.label == "HNSW ef"]
    if not widgets:
        return check("ef tracks candidates", False, "no HNSW ef widget")
    return check("ef tracks candidates", widgets[0].value >= 500,
                 f"ef={widgets[0].value} at candidates=500")


def check_fingerprint_is_stable():
    """The live poll must not see change where there is none.

    collection_fingerprint samples one arbitrary row and Milvus guarantees no
    row order, so putting a per-row field in it would make the poll "change" on
    every tick -- clearing the BM25 and cluster-text caches every few seconds in
    the name of freshness, and reloading every page under the user.
    """
    from milvusui.resources import collection_fingerprint

    seen = [collection_fingerprint() for _ in range(5)]
    return check("fingerprint stable", all(f == seen[0] for f in seen),
                 f"{seen[0]} across {len(seen)} polls")


def check_drop_panel():
    """The checklist deletes what is ticked; Drop all needs a second click.

    Run against a sandbox folder, never the real one: the whole point of the
    control is that it deletes files with no undo, so a test of it must not be
    able to reach anything the user cares about.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="milvusui_drop_"))
    try:
        for name in ("a.pdf", "b.pdf"):
            (sandbox / name).write_bytes(b"%PDF-1.4 fake")
        (sandbox / "keep.txt").write_text("supported, not ticked", encoding="utf-8")
        (sandbox / "notes.md").write_text("not ingestable, still listed",
                                          encoding="utf-8")
        (sandbox / ".gitkeep").write_text("", encoding="utf-8")

        at = run_view("ingest")
        at.text_input(key="in_folder").set_value(str(sandbox))
        at.run()

        listed = sorted(box.label.split(" · ")[0] for box in at.checkbox
                        if box.key and box.key.startswith("drop::"))
        every = check("drop lists every file",
                      listed == ["a.pdf", "b.pdf", "keep.txt", "notes.md"],
                      f"lists {listed}")

        buttons = {b.label.split(" ")[0]: b for b in at.button}
        gated = check("delete needs a tick",
                      "Delete" in buttons and buttons["Delete"].disabled,
                      "disabled while nothing is ticked")

        at.checkbox(key="drop::0::a.pdf").check()
        at.run()
        [b for b in at.button if b.label.startswith("Delete ")][0].click()
        at.run()
        left = sorted(path.name for path in sandbox.iterdir())
        selected = check("delete takes only ticked",
                         left == [".gitkeep", "b.pdf", "keep.txt", "notes.md"],
                         f"left behind {left}")

        # Drop all: the first click only arms it, and must delete nothing.
        [b for b in at.button if b.label.startswith("Drop all")][0].click()
        at.run()
        armed = check("drop all arms first",
                      sorted(p.name for p in sandbox.iterdir()) == left,
                      "nothing deleted on the first click")

        [b for b in at.button if b.label == "Yes, delete all"][0].click()
        at.run()
        # .gitkeep is what keeps the folder in the repo, so it must survive the
        # one button whose whole point is that it takes everything.
        remaining = sorted(path.name for path in sandbox.iterdir())
        emptied = check("drop all spares .gitkeep", remaining == [".gitkeep"],
                        f"left behind {remaining}")
        return every & gated & selected & armed & emptied
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def check_gitkeep_is_undeletable():
    """`.gitkeep` must be invisible to the checklist AND refused by the delete.

    The folder's contents are git-ignored while the folder itself is tracked, so
    that empty file is the only reason fileinput/ survives a clone. Hiding it
    from the list is not enough on its own: this hands it straight to the
    deleter, bypassing the list entirely, which is what "never deleted" has to
    mean if it is to survive someone later assembling their own list.
    """
    from milvusui.views import ingest

    sandbox = Path(tempfile.mkdtemp(prefix="milvusui_keep_"))
    try:
        (sandbox / ".gitkeep").write_text("", encoding="utf-8")
        (sandbox / "doc.pdf").write_bytes(b"%PDF-1.4 fake")

        listed = [path.name for path in ingest._files_in(sandbox)]
        hidden = check("gitkeep not listed", listed == ["doc.pdf"],
                       f"lists {listed}")
        try:
            ingest._delete_files([sandbox / ".gitkeep", sandbox / "doc.pdf"], 0)
        except Exception:
            pass          # st.rerun() has nothing to rerun outside a script run
        left = sorted(path.name for path in sandbox.iterdir())
        return hidden & check("gitkeep refused by delete", left == [".gitkeep"],
                              f"left behind {left}")
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def check_extract_reports_progress():
    """extract_records must report progress, not only print it.

    The Ingest page draws its bar from these callbacks, and they are the only
    structured account of a long extraction: extractpdf's own reporting is an
    in-place counter on stdout, which runner.call captures into a buffer nobody
    sees until the job it was describing is already over.
    """
    import contextlib
    import io

    import extractpdf

    sandbox = Path(tempfile.mkdtemp(prefix="milvusui_prog_"))
    try:
        for name in ("one.txt", "two.txt"):
            (sandbox / name).write_text("A sentence. And another one here.",
                                        encoding="utf-8")
        files, pages = [], []
        with contextlib.redirect_stdout(io.StringIO()):
            extractpdf.extract_records(
                str(sandbox), ocr="never",
                on_file=lambda i, n, path: files.append((i, n, path.name)),
                on_page=lambda page, n: pages.append((page, n)))
        # The total arrives with the first callback, so a bar can be sized
        # before any work is done rather than growing as it goes.
        return (check("extract reports files",
                      files == [(1, 2, "one.txt"), (2, 2, "two.txt")], f"{files}")
                & check("extract reports pages", pages == [(1, 1), (1, 1)],
                        f"{pages}"))
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def regressions():
    ok = True
    for name, function in [("caches", check_cache_invalidation),
                           ("editor", check_criterion_editor_follows_path),
                           ("ef", check_ef_tracks_candidates),
                           ("fingerprint", check_fingerprint_is_stable),
                           ("drop", check_drop_panel),
                           ("gitkeep", check_gitkeep_is_undeletable),
                           ("progress", check_extract_reports_progress)]:
        try:
            ok &= function()
        except Exception as error:
            print(f"[FAIL] {name:<22} harness: {type(error).__name__}: {error}")
            ok = False
    return ok


def main():
    ok = True
    print("-- render --")
    for name, title in [("search", "Search"), ("compare", "Compare methods"),
                        ("ingest", "Ingest"), ("clustering", "Clustering"),
                        ("visualize", "Visualise"), ("collection", "Collection")]:
        try:
            ok &= report(name, run_view(name), title)
        except Exception as e:
            print(f"[FAIL] {name:<14} harness: {type(e).__name__}: {e}")
            ok = False

    print("\n-- app shell --")
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    at.run()
    ok &= report("app.py", at, "Search")

    print("\n-- search interaction --")
    for method in ["hybrid", "dense", "lexical", "tfidf", "mmr", "weighted",
                   "hierarchical"]:
        try:
            at = run_view("search")
            at.text_input(key="q").set_value("where are the subsidiaries")
            at.session_state["method"] = method
            at.run()
            buttons = [b for b in at.button if b.label == "Search"]
            if not buttons:
                print(f"[FAIL] {method:<14} no Search button")
                ok = False
                continue
            buttons[0].click()
            at.run()
            state = at.session_state["results"] if "results" in at.session_state else None
            excs = [f"{e.type}: {e.message}" for e in at.exception]
            errs = [str(e.value) for e in at.error]
            n = len(state["records"]) if state else 0
            bad = bool(excs) or bool(errs) or n == 0
            ms = f"{state['elapsed']*1000:.0f}ms" if state else "-"
            print(f"[{'FAIL' if bad else '  ok'}] {method:<14} results={n:<3} {ms}")
            for e in excs:
                print(f"          EXC {e[:400]}")
            for e in errs:
                print(f"          err {e[:220]}")
            ok &= not bad
        except Exception as e:
            print(f"[FAIL] {method:<14} harness: {type(e).__name__}: {e}")
            ok = False

    print("\n-- regressions --")
    ok &= regressions()

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
