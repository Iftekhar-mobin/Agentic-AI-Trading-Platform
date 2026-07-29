"""Model discovery, selection and installation.

Listing is a read operation — comparing backends on cost and context length is
the sort of thing anyone with dashboard access should be able to do. Changing
the active model or downloading weights is not: both alter how every subsequent
recommendation is produced, and a pull writes gigabytes to the host. Those need
the ``admin`` scope, and switching can be disabled outright with
``ATP_LLM__ALLOW_RUNTIME_MODEL_SWITCHING=false`` for deployments where the
model is a release decision rather than a runtime one.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from atp.composition import Container
from atp.domain.errors import ModelUnavailableError
from atp.domain.models.llm import ActiveModel, LLMProvider, ModelInfo, ModelPullResult
from atp.interfaces.api.security import RequiresAdmin, RequiresRead

router = APIRouter(tags=["models"], prefix="/models")


class ModelsResponse(BaseModel):
    active: ActiveModel
    switching_enabled: bool
    providers: list[LLMProvider]
    models: list[ModelInfo]


class SelectModelRequest(BaseModel):
    provider: LLMProvider
    model: str = Field(min_length=1, examples=["deepseek/deepseek-chat-v3.1:free"])


class PullModelRequest(BaseModel):
    provider: LLMProvider = LLMProvider.OLLAMA
    model: str = Field(min_length=1, examples=["llama3.1:8b"])


def _container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


@router.get("", response_model=ModelsResponse, dependencies=[RequiresRead])
async def list_models(
    request: Request,
    provider: Annotated[
        LLMProvider | None, Query(description="Defaults to the active provider")
    ] = None,
    free_only: Annotated[bool, Query(description="Only models with no per-token cost")] = False,
) -> ModelsResponse:
    """List one provider's models, alongside what is currently active."""
    container = _container(request)
    active = container.llm_router.active
    catalog = container.model_catalogs.get(provider or active.provider)

    # A provider that is unreachable yields an empty list, never a 500: the page
    # showing the other options is more useful than an error.
    models = list(await catalog.list_models()) if catalog is not None else []
    if free_only:
        models = [model for model in models if model.free]

    return ModelsResponse(
        active=active,
        switching_enabled=container.settings.llm.allow_runtime_model_switching,
        providers=list(LLMProvider),
        models=models,
    )


@router.post("/select", response_model=ActiveModel, dependencies=[RequiresAdmin])
async def select_model(request: SelectModelRequest, http_request: Request) -> ActiveModel:
    """Point every agent at a different model, effective on the next call."""
    container = _container(http_request)
    if not container.settings.llm.allow_runtime_model_switching:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "runtime model switching is disabled; set "
                "ATP_LLM__ALLOW_RUNTIME_MODEL_SWITCHING=true to enable it"
            ),
        )
    try:
        return await container.llm_router.select(request.provider, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/pull", response_model=ModelPullResult, dependencies=[RequiresAdmin])
async def pull_model(request: PullModelRequest, http_request: Request) -> ModelPullResult:
    """Download a local model. Slow — gigabytes over the network."""
    container = _container(http_request)
    catalog = container.model_catalogs.get(request.provider)
    if catalog is None or not catalog.supports_pull:
        raise ModelUnavailableError(
            f"provider '{request.provider.value}' is hosted; there is nothing to download"
        )
    return await catalog.pull(request.model)
