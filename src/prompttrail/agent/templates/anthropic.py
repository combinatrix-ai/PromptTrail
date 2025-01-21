import json
import logging
from typing import Any, Dict, Generator, Optional, Union

from prompttrail.agent.templates._core import GenerateTemplate
from prompttrail.agent.templates._tool import ToolingTemplate
from prompttrail.agent.tools import ToolResult
from prompttrail.core import Message, Session

logger = logging.getLogger(__name__)


class AnthropicToolingTemplate(ToolingTemplate):
    """Anthropic-specific implementation of tool handling template.

    This template handles the Anthropic-specific format for tool calls and results,
    adapting them to the common interface provided by ToolingTemplate.

    See documentation: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    """

    # TODO: allow tool_choice

    def format_tool_call(
        self, message: Union[Message, Session]
    ) -> Optional[Dict[str, Any]]:
        """Extract tool call information from Anthropic message format.

        Args:
            message: Message from Anthropic API response or Session containing messages

        Returns:
            Optional dictionary containing tool call details, or None if no tool call present
        """
        logger.debug(f"Checking for tool call in message: {message}")

        # If message is a Session, get the last message
        if isinstance(message, Session):
            if not message.messages:
                return None
            message = message.messages[-1]

        # Check for tool call in message metadata
        if (
            hasattr(message, "metadata")
            and message.metadata
            and "tool_use" in message.metadata
        ):
            tool_use = message.metadata["tool_use"]
            logger.debug(f"Found tool call: {tool_use}")
            return {"name": tool_use["name"], "arguments": tool_use["input"]}
        logger.debug("No tool call found")
        return None

    def format_tool_result(self, result: ToolResult) -> Message:
        """Format tool result for Anthropic message format.

        Args:
            result: Result from tool execution

        Returns:
            Formatted message containing the tool result

        Final message format should be like this::

            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
                        "content": "15 degrees"
                    }
                ]
            }

        """
        logger.debug(f"Formatting tool result: {result}")
        content = result.content
        content["type"] = "tool_result"
        content = json.dumps(result.content)
        logger.debug(f"Formatted content: {content}")
        message = Message(
            role="tool_result",
            content=content,
            metadata={},  # Explicitly set empty metadata
        )
        logger.debug(f"Created tool result message: {message}")
        return message

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Override _render to handle tool-specific message generation.

        This implementation ensures proper metadata handling and tool state tracking.
        """
        logger.debug(f"Starting render with session: {session}")

        # Generate initial message
        session = yield from GenerateTemplate(role=self.role).render(session)
        logger.debug(f"Generated initial message: {session}")

        # Check for tool call
        while tool_call := self.format_tool_call(session):
            try:
                # Get and execute tool
                tool = self.get_tool(tool_call["name"])
                logger.info(f"Executing tool: {tool.name}")
                result = tool.execute(**tool_call["arguments"])
                logger.debug(f"Tool execution result: {result}")

                # Format and append result
                result_message = self.format_tool_result(result)
                logger.debug(f"Appended result message to session: {result_message}")
                yield result_message
                session.append(result_message)

                if session.messages[-1].metadata.get("tool_use"):
                    del session.messages[-1].metadata["tool_use"]

                # Generate final response with clean metadata
                logger.debug("Generating final response")
                session = yield from GenerateTemplate(role=self.role).render(session)
                final_message = session.messages[-1]

                logger.debug(f"Generated final message: {final_message}")
            except Exception as e:
                logger.error(f"Error executing tool {tool_call['name']}: {str(e)}")
                raise

        logger.debug(f"No tool call found, returning message: {session}")
        return session
