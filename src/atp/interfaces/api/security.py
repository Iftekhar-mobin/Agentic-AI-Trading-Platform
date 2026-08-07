"""API authentication and authorization.

Bearer API keys with coarse scopes. Two decisions worth stating:

- **Comparison is constant-time.** ``==`` on a secret leaks its prefix through
  timing, and an API key is exactly the kind of secret an attacker can probe
  repeatedly.
- **Unauthenticated mode is explicit and loud.** With no keys configured the
  API serves anyone, which is what makes `atp serve` usable on a laptop. It
  logs a warning per request rather than silently, and ``Settings`` refuses to
  start in production without keys — so the convenient mode cannot be deployed
  by accident.

Scopes are checked per route via ``requires(Scope.TRADE)``, so a dashboard key
that can read the portfolio cannot place an order.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable, Coroutine
from typing import Annotated, Any

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from atp.infrastructure.config import ApiKey, Scope, SecuritySettings

log = structlog.get_logger()

# auto_error=False so a missing header reaches our handler, which needs to
# distinguish "no credentials, and none are required" from "no credentials".
_bearer = HTTPBearer(auto_error=False, description="API key as a bearer token")

ANONYMOUS = ApiKey(name="anonymous", key=SecretStr(""), scopes=tuple(Scope))
"""The identity used when no keys are configured; holds every scope."""


def _settings(request: Request) -> SecuritySettings:
    return request.app.state.container.settings.api  # type: ignore[no-any-return]


def _match(candidate: str, keys: tuple[ApiKey, ...]) -> ApiKey | None:
    """Constant-time lookup: every configured key is compared, always."""
    found: ApiKey | None = None
    for api_key in keys:
        if hmac.compare_digest(candidate, api_key.key.get_secret_value()):
            found = api_key
    return found


async def authenticate(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> ApiKey:
    """Resolve the caller's identity, or reject the request."""
    settings = _settings(request)
    if not settings.auth_required:
        log.warning("api.unauthenticated_request", path=request.url.path)
        return ANONYMOUS

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = _match(credentials.credentials, settings.keys)
    if api_key is None:
        log.warning("api.invalid_credentials", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    structlog.contextvars.bind_contextvars(api_key=api_key.name)
    return api_key


Identity = Annotated[ApiKey, Depends(authenticate)]


def requires(scope: Scope) -> Callable[[ApiKey], Coroutine[Any, Any, ApiKey]]:
    """Build a dependency that authenticates and then checks one scope."""

    async def dependency(identity: Identity) -> ApiKey:
        if not identity.allows(scope):
            log.warning(
                "api.scope_denied",
                api_key=identity.name,
                required=scope.value,
                held=[held.value for held in identity.scopes],
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"this credential lacks the '{scope.value}' scope",
            )
        return identity

    return dependency


RequiresRead = Depends(requires(Scope.READ))
RequiresTrade = Depends(requires(Scope.TRADE))
RequiresSignal = Depends(requires(Scope.SIGNAL))
RequiresAdmin = Depends(requires(Scope.ADMIN))
