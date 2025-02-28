"""Templates package for PromptTrail agent.

Control module provides control flow templates for building conversation templates.
Core module provides base classes and utilities for building conversation templates.
Tool module provides tool use related templates for building conversation templates.
"""

from prompttrail.agent.templates._control import (
    BreakTemplate,
    ControlTemplate,
    EndTemplate,
    IfTemplate,
    LinearTemplate,
    LoopTemplate,
)
from prompttrail.agent.templates._core import (
    AssistantTemplate,
    Event,
    GenerateTemplate,
    MessageTemplate,
    Stack,
    SystemTemplate,
    Template,
    UserTemplate,
)
from prompttrail.agent.templates._tool import (
    AnthropicToolingTemplate,
    ExecuteToolTemplate,
    OpenAIToolingTemplate,
    ToolingTemplate,
    ToolingTemplateBase,
)

__all__ = [
    "Stack",
    "Template",
    "MessageTemplate",
    "GenerateTemplate",
    "SystemTemplate",
    "UserTemplate",
    "AssistantTemplate",
    "ControlTemplate",
    "LoopTemplate",
    "IfTemplate",
    "LinearTemplate",
    "EndTemplate",
    "BreakTemplate",
    "ToolingTemplateBase",
    "AnthropicToolingTemplate",
    "OpenAIToolingTemplate",
    "ToolingTemplate",
    "ExecuteToolTemplate",
    "Event",
]
