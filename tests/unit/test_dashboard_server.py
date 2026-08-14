"""Tests for the console's API-server supervisor (no server is ever started)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app"))

from server import DEFAULT_PORT, ApiServer, Target, parse_target


@pytest.mark.parametrize(
    ("base_url", "host", "port"),
    [
        ("http://127.0.0.1:8000", "127.0.0.1", 8000),
        ("http://localhost:9001", "localhost", 9001),
        ("127.0.0.1:8000", "127.0.0.1", 8000),
        ("http://127.0.0.1", "127.0.0.1", DEFAULT_PORT),
        ("https://atp.example.com", "atp.example.com", DEFAULT_PORT),
    ],
)
def test_the_target_is_read_from_the_base_url(base_url: str, host: str, port: int) -> None:
    target = parse_target(base_url)
    assert (target.host, target.port) == (host, port)


@pytest.mark.parametrize("base_url", ["", "   ", "http://", "127.0.0.1:eight-thousand"])
def test_a_nonsense_url_falls_back_rather_than_raising(base_url: str) -> None:
    """A typo in the URL box must not take the whole page down with it."""
    target = parse_target(base_url)
    assert (target.host, target.port) == ("127.0.0.1", DEFAULT_PORT)


def test_only_local_targets_are_startable() -> None:
    """A remote base URL is someone else's server; the button is not offered."""
    assert parse_target("http://127.0.0.1:8000").is_local
    assert parse_target("http://localhost:8000").is_local
    assert not parse_target("https://atp.example.com").is_local


def test_the_command_uses_the_interpreter_running_the_console(tmp_path: Path) -> None:
    """Not the `atp` script: it is in the same environment but not always on PATH."""
    server = ApiServer(tmp_path, tmp_path / "api.log")
    command = server.command(Target(host="127.0.0.1", port=9100))

    assert command[:3] == [sys.executable, "-m", "uvicorn"]
    assert "--factory" in command
    assert command[-4:] == ["--host", "127.0.0.1", "--port", "9100"]


def test_nothing_is_managed_before_a_start(tmp_path: Path) -> None:
    server = ApiServer(tmp_path, tmp_path / "api.log")

    assert not server.managed
    assert server.pid is None
    assert server.exit_code is None


def test_stopping_a_server_that_was_never_started_is_harmless(tmp_path: Path) -> None:
    ApiServer(tmp_path, tmp_path / "api.log").stop()


def test_the_log_tail_is_the_end_of_the_file(tmp_path: Path) -> None:
    """Where "why did it not start" lives - usually 'address already in use'."""
    log = tmp_path / "api.log"
    log.write_text("\n".join(f"line {index}" for index in range(100)), encoding="utf-8")
    server = ApiServer(tmp_path, log)

    tail = server.log_tail(lines=5)
    assert tail.splitlines() == ["line 95", "line 96", "line 97", "line 98", "line 99"]


def test_a_missing_log_is_empty_not_an_error(tmp_path: Path) -> None:
    assert ApiServer(tmp_path, tmp_path / "nope.log").log_tail() == ""
