"""Thin HTTP client for the ATP API.

The dashboard talks to the platform the same way any other consumer does — over
the authenticated HTTP API, never by importing the container. That keeps the UI
honest: if something is awkward to render here, the API is missing something,
and the React front end named in the roadmap will hit exactly the same
endpoints when it replaces this.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
ANALYSIS_TIMEOUT = 300.0
"""Analysis runs several LLM calls; the default 5s timeout would always lose."""


class ApiError(RuntimeError):
    """The API answered with an error status."""

    def __init__(self, status_code: int, detail: str, request_id: str = "") -> None:
        suffix = f" (request {request_id})" if request_id else ""
        super().__init__(f"HTTP {status_code}: {detail}{suffix}")
        self.status_code = status_code
        self.detail = detail
        self.request_id = request_id


class AtpClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(base_url=self._base_url, headers=headers, timeout=30.0)

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise ApiError(0, f"cannot reach the API at {self._base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise ApiError(
                response.status_code,
                _detail(response),
                response.headers.get("X-Request-ID", ""),
            )
        return response.json()

    def health(self) -> dict[str, Any]:
        return dict(self._request("GET", "/health"))

    def analyze(
        self, symbol: str, interval: str = "1d", agents: list[str] | None = None
    ) -> dict[str, Any]:
        payload = {"symbol": symbol, "interval": interval, "agents": agents or []}
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
