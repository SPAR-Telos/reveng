"""Pydantic models for LLM judge scoring responses."""

from pydantic import BaseModel, Field

from reveng.datatypes import Action


class ActionResponse(BaseModel):
    """Action choice for a single step."""

    action: Action = Field(
        description="The chosen action (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level of your ability to reach the goal square",
    )
    # explanation: str = Field(description="Brief explanation of the action choice")
