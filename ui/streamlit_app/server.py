"""Starting and stopping the API server from the console.

The dashboard is useless without the API behind it, and the first thing anyone
sees on a fresh machine is "API unreachable" next to an instruction to go and
run something in a terminal. This module lets that be a button instead.

It supervises a child process, nothing more. The console still talks to the API
over HTTP exactly as any other client does — it does not import the container,
and a server started from a terminal, a container, or another machine works
identically. That boundary is the point: this is convenience, not a back door.

Two things it is careful about:

- **It only manages what it started.** A server already answering on the port is
  reported as running and left alone. Offering to stop a process we do not own
  would be a lie at best and someone else's outage at worst.
- **It does not orphan the child.** The process is terminated when the console
  exits, so closing the dashboard does not leave a server holding port 8000
  until the next reboot puzzles someone.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

SERVER_MODULES = ("uvicorn", "atp")
"""What the interpreter running the server has to be able to import."""

VENV_INTERPRETERS = (Path("Scripts", "python.exe"), Path("bin", "python"))
"""Where a Python lives inside a virtualenv, on Windows and everywhere else."""

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::1"})
"""Hosts we could plausibly start a server on. Anything else belongs to someone
else's machine and the button is hidden rather than made to fail."""

STOP_TIMEOUT = 10.0
"""Seconds to wait for a polite shutdown before killing. Uvicorn closes its
listeners promptly; anything longer than this is stuck, not busy."""


@dataclass(frozen=True)
class Target:
    """Where the console expects the API to be."""

    host: str
    port: int

    @property
    def is_local(self) -> bool:
        return self.host in LOCAL_HOSTS


def parse_target(base_url: str) -> Target:
    """The host and port a base URL points at, with sensible fallbacks.

    A typo in the URL box should not raise on the way to rendering a page, so
    anything unparseable resolves to the defaults the CLI uses.
    """
    base_url = base_url.strip()
    try:
        parsed = urlparse(base_url if "//" in base_url else f"//{base_url}", scheme="http")
        host, port = parsed.hostname, parsed.port
    except ValueError:  # a non-numeric port, e.g. "127.0.0.1:eight-thousand"
        host, port = None, None
    return Target(host=(host or "").strip() or DEFAULT_HOST, port=port or DEFAULT_PORT)


def _importable(module: str) -> bool:
    """Whether the running interpreter could import *module*, without doing so.

    A broken or half-installed distribution makes ``find_spec`` raise rather
    than answer; either way the honest answer here is "no".
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class ApiServer:
    """A supervised ``uvicorn`` child process.

    One instance is shared by every browser session (see ``st.cache_resource``
    in the dashboard), so two open tabs see one server rather than two.
    """

    def __init__(self, project_root: Path, log_path: Path) -> None:
        self._root = project_root
        self._log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        atexit.register(self.stop)

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def managed(self) -> bool:
        """Whether *this* console started the server that is running."""
        return self._process is not None and self._process.poll() is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self.managed and self._process else None

    @property
    def exit_code(self) -> int | None:
        """The child's return code, or ``None`` while it is alive.

        A non-``None`` value here after a start attempt means the server died —
        almost always a port already in use, and the log says which.
        """
        return self._process.poll() if self._process else None

    def interpreter(self) -> str:
        """The Python that will run the server.

        The console is routinely launched with whatever Streamlit is on the
        PATH, which on a machine with a system Python is not the project's
        environment at all — and that one has neither ``uvicorn`` nor ``atp``,
        so the button used to produce "No module named uvicorn" in the log and
        nothing else. When the current interpreter cannot serve, fall back to
        the project's own virtualenv, which by definition can.
        """
        if all(_importable(module) for module in SERVER_MODULES):
            return sys.executable
        for relative in VENV_INTERPRETERS:
            candidate = self._root / ".venv" / relative
            if candidate.exists():
                return str(candidate)
        return sys.executable  # Nothing better to offer; the log will say so.

    def command(self, target: Target) -> list[str]:
        """The command line, built from an interpreter that can serve.

        ``python -m uvicorn`` rather than the ``atp`` console script: both live
        in the same environment, but only one of them is guaranteed to be on
        PATH when Streamlit was launched from somewhere unexpected.
        """
        return [
            self.interpreter(),
            "-m",
            "uvicorn",
            "atp.interfaces.api:create_app",
            "--factory",
            "--host",
            target.host,
            "--port",
            str(target.port),
        ]

    def start(self, target: Target) -> None:
        """Launch the server, unless this console already has one running."""
        if self.managed:
            return

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self._log_path.open("wb")
        try:
            # Fixed argv, no shell: nothing user-supplied reaches the command.
            self._process = subprocess.Popen(
                self.command(target),
                cwd=self._root,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                # Detach from the console's signal group so Ctrl-C in the
                # terminal running Streamlit does not race us to the child.
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        finally:
            log.close()

    def stop(self) -> None:
        """Ask the server to exit, and insist if it will not."""
        process = self._process
        if process is None or process.poll() is not None:
            self._process = None
            return

        try:
            # CTRL_BREAK is the only signal a detached Windows process group
            # reliably receives; terminate() is the portable fallback.
            if hasattr(signal, "CTRL_BREAK_EVENT"):
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=STOP_TIMEOUT)
        except (subprocess.TimeoutExpired, OSError):
            process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=STOP_TIMEOUT)
        finally:
            self._process = None

    def log_tail(self, lines: int = 40) -> str:
        """The end of the server's output — where "why did it not start" lives."""
        try:
            content = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(content.splitlines()[-lines:])
