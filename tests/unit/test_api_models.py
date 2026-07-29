"""Tests for the /models endpoints: listing, selection, pulling, and scopes."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from atp.composition import Container
from atp.domain.errors import ModelUnavailableError
from atp.domain.models.llm import ActiveModel, LLMProvider, ModelInfo, ModelPullResult
from atp.domain.ports import ModelCatalog
from atp.infrastructure.config import ApiKey, LLMSettings, Scope, SecuritySettings, Settings
from atp.infrastructure.llm import LLMRouter
from atp.interfaces.api import create_app

READ_KEY = "read-key"
ADMIN_KEY = "admin-key"


class StubCatalog:
    def __init__(
        self,
        provider: LLMProvider,
        models: tuple[ModelInfo, ...] = (),
        *,
        supports_pull: bool = False,
    ) -> None:
        self._provider = provider
        self._models = models
        self._supports_pull = supports_pull
        self.pulled: list[str] = []

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def supports_pull(self) -> bool:
        return self._supports_pull

    async def list_models(self) -> tuple[ModelInfo, ...]:
        return self._models

    async def pull(self, model: str) -> ModelPullResult:
        if not self._supports_pull:
            raise ModelUnavailableError("hosted provider")
        self.pulled.append(model)
        return ModelPullResult(model=model, installed=True, detail="success")


class StubClient:
    async def generate_structured(self, response_model, *, system, prompt):  # type: ignore[no-untyped-def]
        raise AssertionError("not called in these tests")


def model(identifier: str, provider: LLMProvider, *, free: bool = False) -> ModelInfo:
    return ModelInfo(id=identifier, provider=provider, name=identifier, free=free)


def build(**overrides: object) -> Container:
    settings = Settings(
        _env_file=None,
        api=SecuritySettings(
            keys=(
                ApiKey(name="viewer", key=SecretStr(READ_KEY), scopes=(Scope.READ,)),
                ApiKey(
                    name="operator",
                    key=SecretStr(ADMIN_KEY),
                    scopes=(Scope.READ, Scope.ADMIN),
                ),
            )
        ),
        llm=LLMSettings(**overrides),
    )
    router = LLMRouter(
        lambda active: StubClient(),
        ActiveModel(provider=LLMProvider.ANTHROPIC, model="claude-opus-4-8"),
    )
    catalogs: dict[LLMProvider, ModelCatalog] = {
        LLMProvider.ANTHROPIC: StubCatalog(
            LLMProvider.ANTHROPIC, (model("claude-opus-4-8", LLMProvider.ANTHROPIC),)
        ),
        LLMProvider.OPENROUTER: StubCatalog(
            LLMProvider.OPENROUTER,
            (
                model("free/one:free", LLMProvider.OPENROUTER, free=True),
                model("paid/two", LLMProvider.OPENROUTER),
            ),
        ),
        LLMProvider.OLLAMA: StubCatalog(
            LLMProvider.OLLAMA,
            (model("llama3.1:8b", LLMProvider.OLLAMA, free=True),),
            supports_pull=True,
        ),
    }
    return dataclasses.replace(
        Container.build(settings), llm_router=router, model_catalogs=catalogs
    )


@pytest.fixture
def container() -> Container:
    return build()


@pytest.fixture
def client(container: Container) -> Iterator[TestClient]:
    with TestClient(create_app(container)) as test_client:
        yield test_client


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# --- Listing ----------------------------------------------------------------


def test_listing_defaults_to_the_active_provider(client: TestClient) -> None:
    body = client.get("/models", headers=auth(READ_KEY)).json()

    assert body["active"] == {"provider": "anthropic", "model": "claude-opus-4-8"}
    assert body["switching_enabled"] is True
    assert [m["id"] for m in body["models"]] == ["claude-opus-4-8"]
    assert set(body["providers"]) == {"anthropic", "openrouter", "ollama"}


def test_listing_another_provider(client: TestClient) -> None:
    body = client.get("/models", params={"provider": "openrouter"}, headers=auth(READ_KEY)).json()
    assert [m["id"] for m in body["models"]] == ["free/one:free", "paid/two"]


def test_free_only_filters_paid_models(client: TestClient) -> None:
    body = client.get(
        "/models",
        params={"provider": "openrouter", "free_only": True},
        headers=auth(READ_KEY),
    ).json()
    assert [m["id"] for m in body["models"]] == ["free/one:free"]


def test_listing_is_a_read_operation(client: TestClient) -> None:
    """Comparing backends on cost should not require admin."""
    assert client.get("/models", headers=auth(READ_KEY)).status_code == 200


def test_an_unknown_provider_is_rejected(client: TestClient) -> None:
    response = client.get("/models", params={"provider": "gpt5"}, headers=auth(READ_KEY))
    assert response.status_code == 422


# --- Selection --------------------------------------------------------------


def test_selecting_a_model_changes_the_active_one(client: TestClient, container: Container) -> None:
    response = client.post(
        "/models/select",
        json={"provider": "ollama", "model": "llama3.1:8b"},
        headers=auth(ADMIN_KEY),
    )

    assert response.status_code == 200
    assert response.json() == {"provider": "ollama", "model": "llama3.1:8b"}
    assert container.llm_router.active.model == "llama3.1:8b"

    listed = client.get("/models", headers=auth(READ_KEY)).json()
    assert listed["active"]["provider"] == "ollama"


def test_selection_requires_admin(client: TestClient) -> None:
    response = client.post(
        "/models/select",
        json={"provider": "ollama", "model": "llama3.1:8b"},
        headers=auth(READ_KEY),
    )
    assert response.status_code == 403
    assert "admin" in response.json()["detail"]


def test_selection_can_be_disabled_on_the_server() -> None:
    """Some deployments want the model to be a release decision, not a runtime one."""
    container = build(allow_runtime_model_switching=False)
    with TestClient(create_app(container)) as client:
        response = client.post(
            "/models/select",
            json={"provider": "ollama", "model": "llama3.1:8b"},
            headers=auth(ADMIN_KEY),
        )

    assert response.status_code == 409
    assert "disabled" in response.json()["detail"]
    assert container.llm_router.active.provider is LLMProvider.ANTHROPIC


def test_selection_validates_input(client: TestClient) -> None:
    assert (
        client.post(
            "/models/select", json={"provider": "ollama", "model": ""}, headers=auth(ADMIN_KEY)
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/models/select", json={"provider": "nope", "model": "x"}, headers=auth(ADMIN_KEY)
        ).status_code
        == 422
    )


# --- Pulling ----------------------------------------------------------------


def test_pulling_a_local_model(client: TestClient, container: Container) -> None:
    response = client.post(
        "/models/pull",
        json={"provider": "ollama", "model": "qwen2.5:7b"},
        headers=auth(ADMIN_KEY),
    )

    assert response.status_code == 200
    assert response.json() == {"model": "qwen2.5:7b", "installed": True, "detail": "success"}
    catalog = container.model_catalogs[LLMProvider.OLLAMA]
    assert isinstance(catalog, StubCatalog)
    assert catalog.pulled == ["qwen2.5:7b"]


def test_pulling_requires_admin(client: TestClient) -> None:
    response = client.post(
        "/models/pull", json={"provider": "ollama", "model": "x"}, headers=auth(READ_KEY)
    )
    assert response.status_code == 403


def test_pulling_a_hosted_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/models/pull",
        json={"provider": "openrouter", "model": "free/one:free"},
        headers=auth(ADMIN_KEY),
    )

    # ModelUnavailableError maps to 503 through the domain error handler.
    assert response.status_code == 503
    assert "nothing to download" in response.json()["detail"]
