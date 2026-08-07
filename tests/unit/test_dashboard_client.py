"""Tests for the dashboard's HTTP client (transport faked, no server needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui" / "streamlit_app"))

from client import ApiError, AtpClient


def make_client(handler: Any, api_key: str = "secret") -> AtpClient:
    api = AtpClient("http://testserver", api_key)
    # Swapping the transport is the point: no server, real client code.
    api._client = httpx.Client(
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        transport=httpx.MockTransport(handler),
    )
    return api


def test_api_key_is_sent_as_a_bearer_token() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization", "")
        return httpx.Response(200, json={"status": "ok", "version": "0.1.0"})

    api = make_client(handler)
    assert api.health()["status"] == "ok"
    assert seen["auth"] == "Bearer secret"


def test_analysis_sends_the_selected_timeframes_and_agents() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"symbol": "AAPL"})

    make_client(handler).analyze("AAPL", ["1d", "4h"], ["news_analysis"])
    assert seen == {
        "symbol": "AAPL",
        "intervals": ["1d", "4h"],
        "agents": ["news_analysis"],
    }


def test_every_call_is_recorded_for_the_processing_panel() -> None:
    """The panel is only as good as what the client bothered to record."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, headers={"X-Request-ID": "abc-123"})

    api = make_client(handler)
    api.analyze("XAUUSD", ["1d"], ["technical_analysis"])

    assert len(api.exchanges) == 1
    exchange = api.exchanges[0]
    assert exchange["method"] == "POST"
    assert exchange["url"].endswith("/analysis")
    assert exchange["request"]["symbol"] == "XAUUSD"
    assert exchange["response"] == {"ok": True}
    assert exchange["status"] == 200
    assert exchange["request_id"] == "abc-123"
    assert exchange["duration_ms"] >= 0


def test_a_failed_call_is_still_recorded() -> None:
    """A run that breaks is exactly when someone opens the panel."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"detail": {"message": "no report"}})

    api = make_client(handler)
    with pytest.raises(ApiError):
        api.analyze("XAUUSD")

    assert len(api.exchanges) == 1
    assert api.exchanges[0]["status"] == 502
    assert api.exchanges[0]["error"] == "no report"


def test_an_unreachable_api_records_the_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    api = make_client(handler)
    with pytest.raises(ApiError):
        api.health()

    assert api.exchanges[0]["status"] == 0
    assert "connection refused" in api.exchanges[0]["error"]


def test_memory_omits_an_empty_query() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.url.params))
        return httpx.Response(200, json={"matches": []})

    make_client(handler).memory("AAPL", None, 5)
    assert seen == {"limit": "5"}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"detail": "invalid credentials"}, "invalid credentials"),
        ({"detail": {"message": "no report", "failures": []}}, "no report"),
        ({"detail": [{"msg": "field required"}]}, "field required"),
    ],
)
def test_error_bodies_are_rendered_readably(body: dict[str, Any], expected: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json=body)

    with pytest.raises(ApiError) as caught:
        make_client(handler).portfolio()

    assert expected in caught.value.detail
    assert caught.value.status_code == 422


def test_request_id_is_surfaced_for_support() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"detail": "qdrant down"}, headers={"X-Request-ID": "req-42"}
        )

    with pytest.raises(ApiError, match="req-42") as caught:
        make_client(handler).portfolio()

    assert caught.value.request_id == "req-42"


def test_an_unreachable_api_is_a_clear_message_not_a_traceback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(ApiError, match="cannot reach the API") as caught:
        make_client(handler).health()

    assert caught.value.status_code == 0


def test_non_json_error_bodies_do_not_crash_the_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="<html>bad gateway</html>")

    with pytest.raises(ApiError, match="bad gateway"):
        make_client(handler).portfolio()
