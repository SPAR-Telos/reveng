"""Pydantic models for LLM judge scoring responses."""

from pydantic import BaseModel, Field


class ActionResponse(BaseModel):
    """Action choice for a single step."""

    action: int = Field(
        description="The chosen action (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)"
    )

    explanation: str = Field(description="Brief explanation of the action choice")
