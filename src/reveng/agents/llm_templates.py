"""Pydantic models for LLM judge scoring responses."""

from pydantic import BaseModel, Field, field_validator

from reveng.datatypes import Action


class ActionResponse(BaseModel):
    """Action choice for a single step."""

    action: Action = Field(
        description="The chosen action (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)"
    )

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v):
        """Convert int or string to Action enum if needed."""
        if isinstance(v, int):
            return Action(v)
        if isinstance(v, str):
            # Try to convert numeric string to int first
            if v.isdigit():
                return Action(int(v))
            # Try to match enum name (case-insensitive)
            return Action[v.upper()]
        return v


class ActionWithNoteResponse(BaseModel):
    """Action choice for a single step with a note."""

    action: Action = Field(
        description="The chosen action (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN)"
    )
    note: str = Field(description="The note to guide the future actions.")

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v):
        """Convert int or string to Action enum if needed."""
        if isinstance(v, int):
            return Action(v)
        if isinstance(v, str):
            return Action[v.upper()]
        return v


class ActionSequenceResponse(BaseModel):
    """Full action sequence from start to goal."""

    action_sequence: list[int] = Field(
        description="Complete sequence of actions from agent start position to goal. Each action is an integer: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN"
    )

    @field_validator("action_sequence", mode="before")
    @classmethod
    def validate_action_sequence(cls, v):
        """Convert list of actions (int or string) to list of integers."""
        if isinstance(v, list):
            result = []
            for action in v:
                if isinstance(action, int):
                    if 0 <= action <= 3:
                        result.append(action)
                elif isinstance(action, str):
                    # Try numeric string first
                    if action.isdigit():
                        action_int = int(action)
                        if 0 <= action_int <= 3:
                            result.append(action_int)
                    else:
                        # Try enum name
                        try:
                            action_enum = Action[action.upper()]
                            result.append(action_enum.value)
                        except KeyError:
                            pass
            return result
        return v
