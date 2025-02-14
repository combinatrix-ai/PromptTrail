import json
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

from prompttrail.agent.templates._core import Event, GenerateTemplate
from prompttrail.core import Message, MessageRoleType, Model, Session
from prompttrail.core.const import ReachedEndTemplateException
from prompttrail.models.anthropic import AnthropicModel
from prompttrail.models.openai import OpenAIModel

if TYPE_CHECKING:
    from prompttrail.agent.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolingTemplateBase(GenerateTemplate):
    """Base template for tool handling across different models.

    This template provides a common interface for handling tool calls and results
    across different LLM providers. Each provider should implement their own
    format_tool_call and format_tool_result methods to handle provider-specific
    message formats.
    """

    tools: Dict[str, "Tool"] = {}

    def __init__(
        self,
        tools: List["Tool"],
        role: MessageRoleType = "assistant",
        template_id: Optional[str] = None,
        model: Optional["Model"] = None,
        **kwargs,
    ):
        """Initialize the template with tools.

        Args:
            tools: List of tools available for the model to use
            role: Message role (defaults to "assistant")
            template_id: Optional template identifier
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(role=role, template_id=template_id, model=model, **kwargs)
        self.tools = {tool.name: tool for tool in tools}

    def get_tool(self, name: str) -> "Tool":
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
    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
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
    def __init__(
        self,
        tools: List["Tool"],
        role: MessageRoleType = "assistant",
        template_id: Optional[str] = None,
        model: Optional[AnthropicModel] = None,
        **kwargs,
    ):
        super().__init__(
            role=role, template_id=template_id, model=model, tools=tools, **kwargs
        )

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

    def format_tool_result(self, result: "ToolResult") -> Message:
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

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        """Override _render to handle tool-specific message generation.

        This implementation ensures proper metadata handling and tool state tracking.
        """
        self.debug("Starting render with session: %s", session)
        runner = session.runner
        if runner is None:
            raise ValueError(
                "Runner must be given to use ToolingTemplate. Do you use Runner correctly? Runner must be passed via Session."
            )
        model = self.model or runner.model
        if not isinstance(model, AnthropicModel):
            raise ValueError(
                "You must pass an AnthropicModel to use AnthropicToolingTemplate."
            )

        # Generate initial message
        session = yield from GenerateTemplate(role=self.role, model=model).render(
            session
        )
        self.debug("Generated initial message: %s", session)

        # Check for tool call
        while tool_call := self.format_tool_call(session):
            try:
                # Get and execute tool
                tool = self.get_tool(tool_call["name"])
                self.info("Executing tool: %s", tool.name)
                try:
                    result = tool.execute(**tool_call["arguments"])
                except ReachedEndTemplateException as e:
                    if e.farewell_message:
                        self.debug(
                            "Reached end of conversation from tooling with farewell message: %s",
                            e.farewell_message,
                        )
                        m = Message(
                            role="assistant",
                            content=e.farewell_message,
                            metadata=session.metadata,
                        )
                        yield m
                        session.messages.append(m)
                    else:
                        self.debug(
                            "Reached end of conversation from tooling without farewell message"
                        )
                    raise e
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


class OpenAIToolingTemplate(ToolingTemplateBase):
    """OpenAI-specific implementation of tool handling template.

    This template handles the OpenAI-specific format for function calls and results,
    adapting them to the common interface provided by ToolingTemplate.
    """

    def __init__(
        self,
        tools: List["Tool"],
        role: MessageRoleType = "assistant",
        template_id: Optional[str] = None,
        model: Optional[OpenAIModel] = None,
        **kwargs,
    ):
        super().__init__(
            role=role, template_id=template_id, model=model, tools=tools, **kwargs
        )

    def check_tool_arguments(self, args_str: str, tool: "Tool") -> Dict[str, Any]:
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

        # Create argument map for validation
        arg_map = {arg.name: arg for arg in tool.arguments}

        # Check required arguments
        for arg in tool.arguments:
            if arg.required and arg.name not in args_dict:
                raise ValueError(f"Missing required argument: {arg.name}")

            if arg.name in args_dict:
                value = args_dict[arg.name]
                if not arg.validate_value(value):
                    raise ValueError(
                        f"Invalid type for argument {arg.name}: expected {arg.value_type}, got {type(value)}"
                    )
                result[arg.name] = value

        # Warn about unexpected arguments
        for name in args_dict:
            if name not in arg_map:
                self.warning(f"Unexpected argument: {name}")

        return result

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
                "arguments": self.check_tool_arguments(
                    function_call["arguments"], self.get_tool(function_call["name"])
                ),
            }
        return None

    @staticmethod
    def format_tool_result(result: "ToolResult") -> Message:
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
        model = self.model or runner.model
        if not isinstance(model, OpenAIModel):
            raise ValueError(
                "You must pass an OpenAIModel to use OpenAIToolingTemplate."
            )

        # Generate initial message with function calling capability
        rendered_message = model.send(session)
        message = Message(
            content=rendered_message.content or "Processing your request...",
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
                second_response = model.send(session)
                message = Message(
                    content=second_response.content
                    or "Here's the result of your request.",
                    role=second_response.role,
                    metadata={"template_id": self.template_id},
                )
                session.append(message)
                yield message

            except Exception as e:
                self.error("Error executing tool %s: %s", tool_call["name"], str(e))
                raise

        return session


class ToolingTemplate(ToolingTemplateBase):
    """Unified tooling template for different LLM providers."""

    def __init__(
        self,
        tools: List["Tool"],
        role: MessageRoleType = "assistant",
        template_id: Optional[str] = None,
        model: Optional[Model] = None,
        **kwargs,
    ):
        super().__init__(
            role=role, template_id=template_id, model=model, tools=tools, **kwargs
        )

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        """Render the template, handling tool calls and results.

        This implementation ensures proper metadata handling and tool state tracking
        for different LLM providers.
        """
        if session.runner is None:
            raise ValueError(
                "Runner must be given to use ToolingTemplate. Please set runner to the session."
            )

        template: Optional[ToolingTemplateBase] = None
        if self.model:
            if isinstance(self.model, OpenAIModel):
                template = OpenAIToolingTemplate(
                    tools=list(self.tools.values()),
                    role=self.role,
                    template_id=self.template_id,
                    model=self.model,
                )
            elif isinstance(self.model, AnthropicModel):
                template = AnthropicToolingTemplate(
                    tools=list(self.tools.values()),
                    role=self.role,
                    template_id=self.template_id,
                    model=self.model,
                )
            else:
                raise ValueError(
                    "Unsupported model type, use OpenAIModel or AnthropicModel"
                )
        else:
            if isinstance(session.runner.model, OpenAIModel):
                template = OpenAIToolingTemplate(
                    tools=list(self.tools.values()),
                    role=self.role,
                    template_id=self.template_id,
                    model=session.runner.model,
                )
            elif isinstance(session.runner.model, AnthropicModel):
                template = AnthropicToolingTemplate(
                    tools=list(self.tools.values()),
                    role=self.role,
                    template_id=self.template_id,
                    model=session.runner.model,
                )
            else:
                raise ValueError(
                    "Unsupported model type, use OpenAIModel or AnthropicModel"
                )

        session = yield from template.render(session)
        return session


class ExecuteToolTemplate(GenerateTemplate):
    """Template for executing a tool with arguments automatically extracted from metadata."""

    def __init__(
        self,
        tool: "Tool",
        role: MessageRoleType = "user",
        template_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the template with tool and arguments.

        Args:
            tool: Tool instance to execute
            role: Message role (defaults to "user")
            template_id: Optional template identifier
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(role=role, template_id=template_id, **kwargs)
        self.tool = tool

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Execute the tool with the provided arguments.

        This implementation executes the tool and returns the result as a message.
        """
        # Filter out metadata to only include valid arguments
        arg_names = {arg.name for arg in self.tool.arguments}
        valid_args = {k: v for k, v in session.metadata.items() if k in arg_names}

        # Execute tool with allow_redundant=True
        self.tool.validate_arguments(valid_args, allow_redundant=True)
        result = self.tool.execute(**valid_args)

        message = Message(
            role="tool_result",
            content=json.dumps(result.content),
            metadata=session.metadata,
        )
        session.append(message)
        yield message
        return session
