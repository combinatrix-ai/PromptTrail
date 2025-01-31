"""Subroutine package for PromptTrail agent."""

from ._base import SubroutineTemplate
from .session_init_strategy import (
    CleanSessionStrategy,
    FilteredInheritStrategy,
    SessionInitStrategy,
)
from .squash_strategy import (
    FilterByRoleStrategy,
    LastMessageStrategy,
    LLMFilteringStrategy,
    LLMSummarizingStrategy,
    SquashStrategy,
)

__all__ = [
    "SubroutineTemplate",
    "SessionInitStrategy",
    "CleanSessionStrategy",
    "FilteredInheritStrategy",
    "SquashStrategy",
    "LastMessageStrategy",
    "FilterByRoleStrategy",
    "LLMFilteringStrategy",
    "LLMSummarizingStrategy",
]
