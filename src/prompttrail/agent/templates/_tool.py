import json
import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, cast

from prompttrail.agent.templates import GenerateTemplate
from prompttrail.agent.tools import Tool, ToolResult
from prompttrail.core import Message, MessageRoleType, Session
from prompttrail.models.openai import OpenAIModel, OpenAIParam

logger = logging.getLogger(__name__)


class ToolingTemplateBase(GenerateTemplate):
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


class AnthropicToolingTemplate(ToolingTemplateBase):
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
        self.debug("Checking for tool call in message: %s", message)

        # If message is a Session, get the last message
        if isinstance(message, Session):
            if not message.messages:
                return None
            message = message.messages[-1]

        # Check for tool call in message metadata
        if message.tool_use:
            tool_use = message.tool_use
            self.debug("Found tool call: %s", tool_use)
            return {"name": tool_use["name"], "arguments": tool_use["input"]}
        self.debug("No tool call found")
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
        self.debug("Formatting tool result: %s", result)
        content = json.dumps(result.content)
        self.debug("Formatted content: %s", content)
        message = Message(
            role="tool_result",
            content=content,
            tool_use={},  # We used the tool, so we need to clear this metadata
        )
        self.debug("Created tool result message: %s", message)
        return message

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Override _render to handle tool-specific message generation.

        This implementation ensures proper metadata handling and tool state tracking.
        """
        self.debug("Starting render with session: %s", session)

        # Generate initial message
        session = yield from GenerateTemplate(role=self.role).render(session)
        self.debug("Generated initial message: %s", session)

        # Check for tool call
        while tool_call := self.format_tool_call(session):
            try:
                # Get and execute tool
                tool = self.get_tool(tool_call["name"])
                self.info("Executing tool: %s", tool.name)
                result = tool.execute(**tool_call["arguments"])
                self.debug("Tool execution result: %s", result)

                # Format and append result
                result_message = self.format_tool_result(result)
                self.debug("Appended result message to session: %s", result_message)
                yield result_message
                session.append(result_message)

                # Generate final response
                self.debug("Generating final response")
                session = yield from GenerateTemplate(role=self.role).render(session)
                final_message = session.messages[-1]

                self.debug("Generated final message: %s", final_message)
            except Exception as e:
                self.error("Error executing tool %s: %s", tool_call["name"], str(e))
                raise

        self.debug("No tool call found, returning message: %s", session)
        return session


def check_tool_arguments(args_str: str, tool: Tool) -> Dict[str, Any]:
    """Validate and process tool arguments

    Args:
        args_str: JSON string of arguments from the API
        tool: Tool instance to validate against

    Returns:
        Dict[str, Any]: Processed arguments

    Raises:
        ValueError: If required arguments are missing or types don't match
        json.JSONDecodeError: If arguments string is not valid JSON
    """
    try:
        args_dict = json.loads(args_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in arguments: {e}")

    result = {}

    # Check required arguments
    for name, arg in tool.arguments.items():
        if arg.required and name not in args_dict:
            raise ValueError(f"Missing required argument: {name}")

        if name in args_dict:
            value = args_dict[name]
            if not arg.validate_value(value):
                raise ValueError(
                    f"Invalid type for argument {name}: expected {arg.value_type}, got {type(value)}"
                )
            result[name] = value

    # Warn about unexpected arguments
    for name in args_dict:
        if name not in tool.arguments:
            logger.warning(f"Unexpected argument: {name}")

    return result


class OpenAIToolingTemplate(ToolingTemplateBase):
    """OpenAI-specific implementation of tool handling template.

    This template handles the OpenAI-specific format for function calls and results,
    adapting them to the common interface provided by ToolingTemplate.
    """

    def format_tool_call(self, message: Message) -> Optional[Dict[str, Any]]:
        """Extract tool call information from OpenAI message format.

        Args:
            message: Message from OpenAI API response

        Returns:
            Optional dictionary containing tool call details, or None if no tool call present
        """
        if "function_call" in message.metadata:
            function_call = message.metadata["function_call"]
            return {
                "name": function_call["name"],
                "arguments": check_tool_arguments(
                    function_call["arguments"], self.get_tool(function_call["name"])
                ),
            }
        return None

    def format_tool_result(self, result: ToolResult) -> Message:
        """Format tool result for OpenAI message format.

        Args:
            result: Result from tool execution

        Returns:
            Formatted message containing the tool result
        """
        return Message(
            role="tool_result",
            content=json.dumps(result.content),
            metadata={"function_call": {"name": result.metadata.get("tool_name")}}
            if result.metadata.get("tool_name")
            else {},
        )

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Render the template, handling tool calls and results.

        This implementation ensures proper metadata handling and tool state tracking
        for OpenAI's function calling format.
        """
        runner = session.runner
        if runner is None:
            raise ValueError(
                "Runner must be given to use GenerateTemplate. Do you use Runner correctly? Runner must be passed via Session."
            )
        if not isinstance(runner.models, OpenAIModel):
            raise ValueError(
                "Function calling can only be used with OpenAIChatCompletionModel."
            )

        # Update parameters with tools
        temporary_parameters = cast(OpenAIParam, runner.parameters.model_copy())
        temporary_parameters.tools = list(self.tools.values())

        # Generate initial message with function calling capability
        rendered_message = runner.models.send(temporary_parameters, session)
        message = Message(
            content=rendered_message.content,
            role=self.role,
            metadata={"template_id": self.template_id, **rendered_message.metadata},
        )
        session.append(message)
        yield message

        # Check for tool call
        tool_call = self.format_tool_call(rendered_message)
        if tool_call:
            try:
                # Get and execute tool
                tool = self.get_tool(tool_call["name"])
                self.info("Executing tool: %s", tool.name)
                result = tool.execute(**tool_call["arguments"])
                result.metadata["tool_name"] = tool.name  # Set tool name for metadata
                self.debug("Tool execution result: %s", result)

                # Format and append result
                result_message = self.format_tool_result(result)
                self.debug("Appended result message to session: %s", result_message)
                session.append(result_message)
                yield result_message

                # Generate final response
                second_response = runner.models.send(runner.parameters, session)
                message = Message(
                    content=second_response.content,
                    role=second_response.role,
                    metadata={"template_id": self.template_id},
                )
                session.append(message)
                yield message

            except Exception as e:
                self.error("Error executing tool %s: %s", tool_call["name"], str(e))
                raise

        return session
