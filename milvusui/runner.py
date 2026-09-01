"""Adapt the CLI-shaped backend modules to a UI caller.

Every module in this repo is written for a terminal: it reports progress with
`print` and refuses bad input with `raise SystemExit("...")`. Both are exactly
right for a CLI and both are hostile to a UI -- the progress vanishes into the
server's console, and SystemExit is a *BaseException*, so an ordinary
`except Exception` around a call site does not catch it and Streamlit tears the
script down mid-render instead of showing the message.

This module is the single place that translates. `call()` runs a backend
function with its stdout captured and turns SystemExit into a BackendError
carrying both the message and whatever was printed before the failure -- which
is usually the more informative half, since these modules print their diagnosis
and then exit.

Nothing else in the UI should import io/contextlib or catch SystemExit.
"""

import contextlib
import io
import threading

# redirect_stdout swaps a process-global, but Streamlit runs one script thread
# per browser session. Two sessions searching at once would otherwise interleave
# their output into whichever buffer was installed last -- or worse, one session
# restores `sys.stdout` while the other is still writing to its buffer. The lock
# makes capture a critical section: concurrent sessions queue rather than
# corrupt each other's logs.
#
# It serializes backend calls across sessions, which is the correct trade here.
# The work is Milvus round trips and GPU embedding, neither of which parallelises
# usefully in-process anyway, and a POC UI has a handful of users at most.
_CAPTURE_LOCK = threading.Lock()


class BackendError(RuntimeError):
    """A backend refused the request.

    `log` holds whatever it printed before giving up. The backends diagnose in
    print() and then exit with a short message, so the log is often where the
    actionable detail is; views should offer both.
    """

    def __init__(self, message, log=""):
        super().__init__(message)
        self.log = log


def call(fn, *args, **kwargs):
    """Run `fn(*args, **kwargs)`, returning (value, printed_output).

    Raises BackendError if the backend exited, with the message it exited on.
    Any other exception propagates untouched -- a genuine bug should reach the
    view's error boundary with its traceback intact, not be flattened into the
    same box as "you need to run cluster.py first".
    """
    buffer = io.StringIO()
    with _CAPTURE_LOCK:
        try:
            with contextlib.redirect_stdout(buffer):
                value = fn(*args, **kwargs)
        except SystemExit as exit_:
            # SystemExit(str) -> the message; SystemExit(0)/SystemExit() -> a
            # backend that returned early rather than failed. Treat a bare or
            # zero exit as success with no value, since that is what it means.
            code = exit_.code
            if code is None or code == 0:
                return None, buffer.getvalue()
            raise BackendError(str(code), buffer.getvalue()) from None
    return value, buffer.getvalue()


def call_logged(fn, *args, **kwargs):
    """Like `call`, but returns a (value, log) pair even when the backend fails.

    Returns (None, log, error_message) on failure instead of raising. For the
    long-running write paths -- ingest, clustering -- where the partial log
    matters as much as the outcome and the view wants to render both in one
    place rather than in an except block.
    """
    try:
        value, log = call(fn, *args, **kwargs)
        return value, log, None
    except BackendError as error:
        return None, error.log, str(error)
