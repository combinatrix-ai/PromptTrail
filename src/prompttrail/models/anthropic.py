from logging import getLogger
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    cast,
)

import anthropic
from pydantic import ConfigDict
from typing_extensions import TypedDict

from prompttrail.core import Config, Message, Model, Session
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


class AnthropicConfig(Config):
    """Integration configuration class for Anthropic Claude API.

    Manages authentication credentials and model parameters in a centralized way.
    """

    # Authentication
    api_key: str
    """API key for Anthropic API."""

    # Model parameters (inherited and overridden)
    model_name: str = "claude-3-opus-latest"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024

    # Anthropic-specific parameters
    top_p: Optional[float] = None
    """ Top-p value for sampling. """
    top_k: Optional[int] = None
    """ Top-k value for sampling. """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def _validate_model_settings(self) -> None:
        """Anthropic-specific configuration validation"""
        super()._validate_model_settings()
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        if self.temperature is not None and (
            self.temperature < 0 or self.temperature > 1
        ):
            raise ValueError("temperature must be between 0 and 1")
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1):
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")

    def _validate_tools(self) -> None:
        """Anthropic-specific tool validation"""
        super()._validate_tools()
        for tool in self.tools:  # type: ignore
            if not tool.name.replace("-", "").replace("_", "").isalnum():
                raise ValueError(
                    f"Tool name must be alphanumeric, hyphen, underscore: {tool.name}"
                )


class AnthropicModel(Model):
    """Model class for Anthropic Claude API."""

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

    def _send(self, session: Session) -> Message:
        """Send messages and return the response."""
        self._authenticate()
        messages, system_prompt = self._session_to_anthropic_messages(session)

        # Convert AnthropicMessageDict to MessageDict for tool use check
        [dict(msg) for msg in messages]

        create_params: Dict[str, Any] = {
            "model": self.configuration.model_name,
            "messages": messages,
        }

        if self.configuration.max_tokens is not None:
            create_params["max_tokens"] = self.configuration.max_tokens
        if self.configuration.temperature is not None:
            create_params["temperature"] = self.configuration.temperature
        if self.configuration.top_p is not None:
            create_params["top_p"] = self.configuration.top_p
        if self.configuration.top_k is not None:
            create_params["top_k"] = self.configuration.top_k
        if system_prompt is not None:
            create_params["system"] = system_prompt

        # Add tools if present
        if self.configuration.tools:
            create_params["tools"] = [
                self.format_tool(tool) for tool in self.configuration.tools
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

    def _send_async(
        self,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """Send messages asynchronously and return the response."""
        raise NotImplementedError(
            "Async method is not implemented for Anthropic model."
        )

    def list_models(self) -> List[str]:
        """Return a list of available models."""
        self._authenticate()
        if self.client is None:
            raise RuntimeError("Failed to initialize Anthropic client")
        models = self.client.models.list()
        return [model.id for model in models.data]
