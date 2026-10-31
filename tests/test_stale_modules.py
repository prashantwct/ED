"""A deploy that changes core/ is picked up without rebooting the app.

Streamlit re-reads app.py on every run but keeps imported modules in
memory. On the hosted app that paired a new call in app.py with an old
function in core/, and the page died with a TypeError until someone
rebooted it. AppTest reproduces the condition exactly: each run reads
the script afresh while sys.modules persists across runs in this process.
"""

import os
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


def _core_modules():
    return {n: m for n, m in sys.modules.items() if n == "core" or n.startswith("core.")}


def test_a_changed_core_module_is_reloaded_on_the_next_run():
    saved = _core_modules()
    try:
        first = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        assert not first.exception

        # The module in memory no longer matches what app.py calls: the
        # old function, as after a deploy that added an argument.
        stale = sys.modules["core.landing"]

        def hero():
            raise TypeError("hero() got an unexpected keyword argument 'new'")

        stale.hero = hero

        # Nothing on disk has changed, so nothing is reloaded and the
        # stale function runs: the failure the hosted app showed.
        still = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        assert still.exception

        # The deploy: the file on disk is now newer than the module.
        path = stale.__file__
        before = os.stat(path)
        os.utime(path, (before.st_atime, time.time() + 5))
        try:
            fresh = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
        finally:
            os.utime(path, (before.st_atime, before.st_mtime))

        assert not fresh.exception
        assert sys.modules["core.landing"] is not stale
    finally:
        # Other tests hold and patch the module objects they imported;
        # hand those back rather than the ones loaded here.
        for name in list(_core_modules()):
            del sys.modules[name]
        sys.modules.update(saved)
