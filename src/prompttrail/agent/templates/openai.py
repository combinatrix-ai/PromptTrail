import json
import logging
from typing import Any, Dict, Generator, List, Optional, cast

from prompttrail.agent.hooks import TransformHook
from prompttrail.agent.templates import MessageTemplate
from prompttrail.agent.templates._tool import ToolingTemplate
from prompttrail.agent.tools import Tool, ToolResult
from prompttrail.core import Message, MessageRoleType, Session
from prompttrail.models.openai import OpenAIModel, OpenAIParam

logger = logging.getLogger(__name__)


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


class OpenAIToolingTemplate(ToolingTemplate):
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
                logger.info(f"Executing tool: {tool.name}")
                result = tool.execute(**tool_call["arguments"])
                result.metadata["tool_name"] = tool.name  # Set tool name for metadata
                logger.debug(f"Tool execution result: {result}")

                # Format and append result
                result_message = self.format_tool_result(result)
                logger.debug(f"Appended result message to session: {result_message}")
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
                logger.error(f"Error executing tool {tool_call['name']}: {str(e)}")
                raise

        return session


class OpenAIMessageTemplate(MessageTemplate):
    """MessageTemplate for OpenAI. `role` is narrowed down to OpenAIrole."""

    def __init__(
        self,
        content: str,
        role: MessageRoleType,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=role,
            before_transform=before_transform,
            after_transform=after_transform,
        )
