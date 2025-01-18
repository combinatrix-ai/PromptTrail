import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional

from prompttrail.agent.templates import GenerateTemplate
from prompttrail.agent.tools import Tool, ToolResult
from prompttrail.core import Message, MessageRoleType, Session

logger = logging.getLogger(__name__)


class ToolingTemplate(GenerateTemplate):
    """Base template for tool handling across different models.

    This template provides a common interface for handling tool calls and results
    across different LLM providers. Each provider should implement their own
    format_tool_call and format_tool_result methods to handle provider-specific
    message formats.
    """

    def __init__(
        self,
        tools: List[Tool],
        role: MessageRoleType = "assistant",
        template_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the template with tools.

        Args:
            tools: List of tools available for the model to use
            role: Message role (defaults to "assistant")
            template_id: Optional template identifier
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(role=role, template_id=template_id, **kwargs)
        self.tools = {tool.name: tool for tool in tools}

    @abstractmethod
    def format_tool_call(self, message: Message) -> Optional[Dict[str, Any]]:
        """Extract tool call information from message.

        Args:
            message: Message potentially containing tool call information

        Returns:
            Optional dictionary containing tool call details with "name" and "arguments",
            or None if no tool call is present
        """

    @abstractmethod
    def format_tool_result(self, result: ToolResult) -> Message:
        """Format tool result as a message.

        Args:
            result: Result from tool execution

        Returns:
            Formatted message containing the tool result
        """

    def get_tool(self, name: str) -> Tool:
        """Get tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            Tool instance

        Raises:
            ValueError: If tool is not found
        """
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        return self.tools[name]

    @abstractmethod
    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Render the template, handling tool calls and results.

        Args:
            session: Current session state

        Yields:
            Messages generated during rendering

        Returns:
            Final session state
        """
