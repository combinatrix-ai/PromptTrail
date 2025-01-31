"""Base classes for templates."""

from pydantic import BaseModel


class Stack(BaseModel):
    """Stack frame for template execution."""

    template_id: str
