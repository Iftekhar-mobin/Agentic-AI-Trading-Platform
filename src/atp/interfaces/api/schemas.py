"""Shared API schemas."""

from __future__ import annotations

from pydantic import BaseModel


class AgentFailureSchema(BaseModel):
    agent: str
    error: str
