from logging import getLogger
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, cast

import anthropic
from pydantic import ConfigDict
from typing_extensions import TypedDict

from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

if TYPE_CHECKING:
    from prompttrail.agent.tools import Tool, ToolResult

logger = getLogger(__name__)


AnthropicRole = Literal["user", "assistant", "system"]


class AnthropicMessageDict(TypedDict):
    role: AnthropicRole
    content: str


MessageDict = Dict[str, str]


class AnthropicConfig(Configuration):
    """Configuration for Anthropic Claude API."""

    api_key: str
    """API key for Anthropic API."""


class AnthropicParam(Parameters):
    """Parameters for Anthropic Claude models.

    Inherits common parameters from Parameters base class and adds Anthropic-specific parameters.
    For detailed description of each parameter, see https://github.com/anthropics/anthropic-sdk-python/blob/main/api.md
    """

    model_name: str = "claude-3-opus-latest"
    """ Name of the model to use. Use AnthoropicClaudeModel.list_models() to get the list of available models. """
    temperature: Optional[float] = 1.0
    """ Temperature for sampling. """
    max_tokens: int = 1024
    """ Maximum number of tokens to generate. """
    top_p: Optional[float] = None
    """ Top-p value for sampling. """
    top_k: Optional[int] = None
    """ Top-k value for sampling. """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class AnthropicModel(Model):
    """Model for Anthropic Claude API."""

    configuration: AnthropicConfig
    client: Optional[anthropic.Anthropic] = None
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def _authenticate(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic(api_key=self.configuration.api_key)

    def format_tool(self, tool: "Tool") -> Dict[str, Any]:
        """Convert tool to Anthropic format"""
        schema = tool.to_schema()
        properties = {}
        for name, arg in tool.arguments.items():
            properties[name] = {
                "type": "string",  # Currently only supporting string type
                "description": arg.description,
            }

        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": [
                    name for name, arg in tool.arguments.items() if arg.required
                ],
            },
        }

    def format_tool_result(self, result: "ToolResult") -> Dict[str, Any]:
        """Format result for Anthropic API"""
        return {"type": "tool_result", "content": str(result.content)}

    def validate_tools(self, tools: List["Tool"]) -> None:
        """Validate tools according to Anthropic API requirements"""
        for tool in tools:
            # Validate tool name
            if not tool.name.replace("-", "").replace("_", "").isalnum():
                raise ParameterValidationError(
                    f"Tool name must be alphanumeric, hyphen, underscore: {tool.name}"
                )
            # Validate description
            if not tool.description:
                raise ParameterValidationError(
                    f"Tool description is required: {tool.name}"
                )
            # Validate arguments
            if not tool.arguments:
                raise ParameterValidationError(
                    f"Tool must have at least one argument: {tool.name}"
                )

    def _send(self, parameters: Parameters, session: Session) -> Message:
        self._authenticate()
        if not isinstance(parameters, AnthropicParam):
            raise ParameterValidationError(
                f"{AnthropicParam.__name__} is expected, but {type(parameters).__name__} is given."
            )

        messages, system_prompt = self._session_to_anthropic_messages(session)

        # Convert AnthropicMessageDict to MessageDict for tool use check
        [dict(msg) for msg in messages]

        create_params = {
            "model": parameters.model_name,
            "max_tokens": parameters.max_tokens,
            "messages": messages,
        }

        if parameters.temperature is not None:
            create_params["temperature"] = parameters.temperature
        if parameters.top_p is not None:
            create_params["top_p"] = parameters.top_p
        if parameters.top_k is not None:
            create_params["top_k"] = parameters.top_k
        if system_prompt is not None:
            create_params["system"] = system_prompt

        # Add tools if present
        if parameters.tools:
            create_params["tools"] = [
                self.format_tool(tool) for tool in parameters.tools
            ]

        if self.client is None:
            raise RuntimeError("Anthropic client not initialized")

        response: anthropic.Message = self.client.messages.create(**create_params)  # type: ignore
        self.debug("Response: %s", pformat(response))

        # Handle response content
        content = ""
        tool_use_block = None

        # Process all blocks
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
                self.debug("Added text content: %s", block.text)
            elif hasattr(block, "type") and block.type == "tool_use":
                tool_use_block = block
                self.debug("Found tool use block: %s", pformat(tool_use_block))

        # Create message with appropriate content and tool_use
        tool_use = None
        if tool_use_block:
            tool_use = {
                "name": cast(str, tool_use_block.name),
                "input": cast(str, tool_use_block.input),
            }
            self.debug("Added tool use: %s", tool_use)

        # Create message
        message = Message(
            content=content,
            role="assistant",
            tool_use=tool_use,
        )
        self.debug("Created final message: %s", message)
        return message

    def validate_session(self, session: Session, is_async: bool) -> None:
        """Validate session for Anthropic Claude models.

        Extends the base validation with Anthropic-specific validations:
        - At most one system message at the beginning
        - Only specific roles allowed
        - No empty messages
        """
        super().validate_session(session, is_async)

        # Anthropic-specific validation for checking at least one non-system message

        # Filter out control template messages
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        non_system_messages = [
            message for message in messages if message.role != "system"
        ]
        if len(non_system_messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session must contain at least one non-system message."
            )

        # Anthropic-specific validation for empty messages
        if any([message.content == "" for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Empty messages are not allowed. (Anthropic API restriction)"
            )

    @staticmethod
    def _session_to_anthropic_messages(
        session: Session,
    ) -> Tuple[List[AnthropicMessageDict], Optional[str]]:
        """Convert session messages to Anthropic format and extract system prompt"""
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        # Handle system message
        if messages[0].role == "system":
            return (
                [
                    AnthropicMessageDict(
                        role=cast(
                            AnthropicRole,
                            "assistant"
                            if message.role == "assistant"
                            else "user"  # Convert tool_result to user
                            if message.role == "tool_result"
                            else message.role,
                        ),
                        content=str(message.content),
                    )
                    for message in messages[1:]
                ],
                str(messages[0].content),
            )
        else:
            return (
                [
                    AnthropicMessageDict(
                        role=cast(
                            AnthropicRole,
                            "assistant"
                            if message.role == "assistant"
                            else "user"  # Convert tool_result to user
                            if message.role == "tool_result"
                            else message.role,
                        ),
                        content=str(message.content),
                    )
                    for message in messages
                ],
                None,
            )

    def list_models(self) -> List[str]:
        """List available Anthropic models"""
        self._authenticate()
        if self.client is None:
            raise RuntimeError("Failed to initialize Anthropic client")
        models = self.client.models.list()
        return [model.id for model in models.data]
