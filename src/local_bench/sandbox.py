"""Run model-written Python the way HumanEval and MBPP do.

This is the standard functional-correctness check: a short-lived subprocess
with a timeout. It is not a security sandbox. Generated code can still read
files or make network calls. Run these benches only for models you trust, on
a machine whose files you can afford to expose.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

_WRAPPER = r"""
import builtins
import faulthandler
import os
import sys

faulthandler.disable()
for name in (
    "kill", "system", "remove", "removedirs", "rmdir", "rename", "renames",
    "truncate", "replace", "unlink", "chmod", "chown", "chroot",
):
    if hasattr(os, name):
        setattr(os, name, None)
builtins.exit = None
builtins.quit = None
try:
    import shutil
    shutil.rmtree = None
    shutil.move = None
except Exception:
    pass
try:
    import subprocess
    subprocess.Popen = None
except Exception:
    pass

try:
    with open("program.py", "r", encoding="utf-8") as handle:
        source = handle.read()
    namespace = {"__name__": "__main__"}
    exec(compile(source, "program.py", "exec"), namespace, namespace)
except BaseException as exc:
    sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
    sys.exit(1)
"""


def _child_env() -> dict[str, str]:
    keep = (
        "SYSTEMROOT",
        "WINDIR",
        "PATH",
        "PATHEXT",
        "TEMP",
        "TMP",
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
    )
    env = {key: os_value for key in keep if (os_value := __import__("os").environ.get(key))}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    return env


def run_python(program: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Execute a complete program. Return (passed, detail)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with tempfile.TemporaryDirectory(prefix="local-bench-") as directory:
        root = Path(directory)
        (root / "program.py").write_text(program.replace("\x00", ""), encoding="utf-8")
        (root / "wrapper.py").write_text(_WRAPPER, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "wrapper.py"],
                cwd=root,
                env=_child_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=flags,
            )
        except subprocess.TimeoutExpired:
            return False, "timed out"
        detail = (completed.stderr or completed.stdout or "").strip()
        if completed.returncode == 0:
            return True, "passed"
        return False, (detail or f"exit {completed.returncode}")[:800]
