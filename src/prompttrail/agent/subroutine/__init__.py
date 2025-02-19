"""Subroutine package for PromptTrail agent."""

from ._base import SubroutineTemplate
from .session_init_strategy import (
    FilteredInheritStrategy,
    GeneralSessionInitStrategy,
    InheritMetadataStrategy,
    InheritSystemStrategy,
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
    "GeneralSessionInitStrategy",
    "InheritMetadataStrategy",
    "InheritSystemStrategy",
    "FilteredInheritStrategy",
    "SquashStrategy",
    "LastMessageStrategy",
    "FilterByRoleStrategy",
    "LLMFilteringStrategy",
    "LLMSummarizingStrategy",
]
