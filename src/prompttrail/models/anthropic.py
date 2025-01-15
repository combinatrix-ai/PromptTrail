from logging import getLogger
from pprint import pformat
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import anthropic
from pydantic import ConfigDict
from typing_extensions import TypedDict

from prompttrail.agent.tools import Tool, ToolResult
from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = getLogger(__name__)


AnthropicRole = Literal["user", "assistant", "system"]


class AnthropicMessageDict(TypedDict):
    role: AnthropicRole
    content: str


class AnthropicConfig(Configuration):
    """Configuration for Anthropic Claude API."""

    api_key: str
    """API key for Anthropic API."""


class AnthropicParam(Parameters):
    """Parameters for Anthropic Claude models.

    Inherits common parameters from Parameters base class and adds Anthropic-specific parameters.
    For detailed description of each parameter, see https://github.com/anthropics/anthropic-sdk-python/blob/main/api.md
    """

    model_name: str = "claude-3-opus-20240229"
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

    configuration: AnthropicConfig  # type: ignore
    client: Optional[anthropic.Anthropic] = None
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def _authenticate(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic(api_key=self.configuration.api_key)

    def format_tool(self, tool: Tool) -> Dict[str, Any]:
        """Convert tool to Anthropic format"""
        schema = tool.to_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }

    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """Format result for Anthropic API"""
        return {"role": "assistant", "content": str(result.content)}

    def validate_tools(self, tools: List[Tool]) -> None:
        """Validate tools according to Anthropic API requirements"""
        for tool in tools:
            # Validate tool name
            if not tool.name.replace("-", "").isalnum():
                raise ParameterValidationError(
                    f"Tool name must be alphanumeric or hyphen: {tool.name}"
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
        logger.debug(pformat(object=response))  # type: ignore

        # Handle non-text response
        content = "".join([block.text for block in response.content])
        if content == "":
            raise ValueError("Response is empty.")

        result = Message(content=content, role=cast(str, response.role))

        # Process tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_call = response.tool_calls[0]  # Currently handle only first call
            result.metadata["function_call"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }

        return result

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

        # Anthropic API allow zero or one system message at the beginning

        # Anthropic-specific validation for system message
        if (
            messages[0].role == "system"
            and len([message for message in messages if message.role == "system"]) > 1
        ) or (
            messages[0].role != "system"
            and len([message for message in messages if message.role == "system"]) > 0
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should have at most one system message at the beginning. (Anthropic API restriction)"
            )

        # Anthropic-specific validation for allowed roles
        if any(
            [
                message.role not in ["user", "assistant", "system"]
                for message in messages
            ]
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have role of 'user', 'assistant', or 'system'. (Anthropic API restriction)"
            )
        if any([not isinstance(message.content, str) for message in messages]):  # type: ignore
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )

        # Anthropic-specific validation for empty messages
        if any([message.content == "" for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should not be empty string. (Anthoropic API restriction)"
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
                        role=cast(AnthropicRole, message.role),
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
                        role=cast(AnthropicRole, message.role),
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
