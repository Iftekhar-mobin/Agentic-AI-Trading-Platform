"""Thin HTTP client for the ATP API.

The dashboard talks to the platform the same way any other consumer does — over
the authenticated HTTP API, never by importing the container. That keeps the UI
honest: if something is awkward to render here, the API is missing something,
and the React front end named in the roadmap will hit exactly the same
endpoints when it replaces this.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
MAX_BODY_CHARS = 20_000
"""Exchanges are held in session state and rendered; an unbounded analysis
response would bloat both."""
ANALYSIS_TIMEOUT = 300.0
"""Analysis runs several LLM calls; the default 5s timeout would always lose."""

SCREEN_TIMEOUT = 1800.0
"""A screen is ANALYSIS_TIMEOUT per symbol, several at a time. Fifteen symbols
against a local model is comfortably half an hour."""

PULL_TIMEOUT = 1800.0
"""Downloading a local model is gigabytes over the network."""


class ApiError(RuntimeError):
    """The API answered with an error status."""

    def __init__(self, status_code: int, detail: str, request_id: str = "") -> None:
        suffix = f" (request {request_id})" if request_id else ""
        super().__init__(f"HTTP {status_code}: {detail}{suffix}")
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id


class AtpClient:
    """An API client that also keeps a record of what it sent and received.

    The Processing panel is built from ``exchanges``: capturing traffic here,
    at the one place every call already passes through, means no endpoint
    method has to remember to report itself.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self._base_url, headers=headers, timeout=30.0)
        self.exchanges: list[dict[str, Any]] = []
        """Every HTTP call this client made, oldest first."""

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        exchange: dict[str, Any] = {
            "method": method,
            "url": f"{self._base_url}{path}",
            "request": kwargs.get("json") or kwargs.get("params") or {},
            "timeout_s": kwargs.get("timeout"),
        }
        self.exchanges.append(exchange)
        started = time.perf_counter()

        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            exchange["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
            exchange["status"] = 0
            exchange["error"] = str(exc)
            raise ApiError(0, f"cannot reach the API at {self._base_url}: {exc}") from exc

        exchange["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        exchange["status"] = response.status_code
        exchange["request_id"] = response.headers.get("X-Request-ID", "")
        exchange["bytes"] = len(response.content)

        if response.status_code >= 400:
            exchange["error"] = _detail(response)
            raise ApiError(
                response.status_code,
                _detail(response),
                response.headers.get("X-Request-ID", ""),
            )

        body = response.json()
        exchange["response"] = body
        return body

    def health(self) -> dict[str, Any]:
        return dict(self._request("GET", "/health"))

    def analyze(
        self,
        symbol: str,
        intervals: list[str] | None = None,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "symbol": symbol,
            "intervals": intervals or ["1d"],
            "agents": agents or [],
        }
        return dict(self._request("POST", "/analysis", json=payload, timeout=ANALYSIS_TIMEOUT))

    def portfolio(self) -> dict[str, Any]:
        return dict(self._request("GET", "/portfolio"))

    def orders(self, limit: int = 50) -> dict[str, Any]:
        return dict(self._request("GET", "/orders", params={"limit": limit}))

    def memory(self, symbol: str, query: str | None = None, limit: int = 10) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if query:
            params["query"] = query
        return dict(self._request("GET", f"/memory/{symbol}", params=params))

    def models(self, provider: str | None = None, free_only: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if provider:
            params["provider"] = provider
        if free_only:
            params["free_only"] = True
        return dict(self._request("GET", "/models", params=params))

    def select_model(self, provider: str, model: str) -> dict[str, Any]:
        payload = {"provider": provider, "model": model}
        return dict(self._request("POST", "/models/select", json=payload))

    def pull_model(self, provider: str, model: str) -> dict[str, Any]:
        payload = {"provider": provider, "model": model}
        # Downloading weights is gigabytes; the default timeout is nowhere near enough.
        return dict(self._request("POST", "/models/pull", json=payload, timeout=PULL_TIMEOUT))

    def consensus(self, **payload: Any) -> dict[str, Any]:
        # Runs the whole agent pool, so it needs the analysis timeout, not 30s.
        return dict(self._request("POST", "/consensus", json=payload, timeout=ANALYSIS_TIMEOUT))

    def universe(self) -> dict[str, Any]:
        return dict(self._request("GET", "/universe"))

    def screen(self, **payload: Any) -> dict[str, Any]:
        # symbols x agents LLM calls, run a few symbols at a time. The slowest
        # thing the platform does, and the one most worth waiting for.
        return dict(self._request("POST", "/screen", json=payload, timeout=SCREEN_TIMEOUT))

    def publish_signal(self, **payload: Any) -> dict[str, Any]:
        return dict(self._request("POST", "/signals", json=payload))

    def signals(self, limit: int = 20) -> dict[str, Any]:
        return dict(self._request("GET", "/signals", params={"limit": limit}))

    def risk_check(self, **payload: Any) -> dict[str, Any]:
        return dict(self._request("POST", "/risk-check", json=payload))

    def trade(self, **payload: Any) -> dict[str, Any]:
        return dict(self._request("POST", "/trade", json=payload))


def _detail(response: httpx.Response) -> str:
    """Pull a readable message out of whichever error shape came back."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or response.reason_phrase

    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail)
