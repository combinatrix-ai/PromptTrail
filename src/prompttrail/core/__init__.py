import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prompttrail.core.cache import CacheProvider
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.mocks import MockProvider
from prompttrail.core.utils import Debuggable

if TYPE_CHECKING:
    from prompttrail.agent.runners import Runner
    from prompttrail.agent.templates import Stack
    from prompttrail.agent.tools import Tool, ToolResult
else:
    # This is required to avoid circular imports and disable Pydantic errors
    Runner = Any  # type: ignore
    Stack = Any  # type: ignore
    Tool = Any  # type: ignore
    ToolResult = Any  # type: ignore

logger = logging.getLogger(__name__)


# Define standard message roles
MessageRoleType = Literal["system", "user", "assistant", "tool_result", "control"]


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate string to max_length."""
    if len(s) > max_length:
        return s[: max_length - 3] + "..."
    return s


class Metadata(Dict[str, Any]):
    """Base class providing type-safe metadata"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 1:
            if isinstance(args[0], (dict, Metadata)):
                self.update(dict(args[0]))
        elif kwargs:
            self.update(kwargs)

    def copy(self) -> "Metadata":
        return self.__class__(dict(self))

    def model_copy(self, *args: Any, **kwargs: Any) -> "Metadata":
        return self.copy()

    def dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return dict(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: Any,
    ) -> Any:
        from pydantic_core.core_schema import (
            dict_schema,
            no_info_after_validator_function,
            union_schema,
        )

        def validate_dict(value: Any) -> "Metadata":
            if isinstance(value, cls):
                return cls(dict(value))
            if isinstance(value, dict):
                return cls(value)
            if value is None:
                return cls()
            try:
                return cls(dict(value))
            except (TypeError, ValueError):
                raise ValueError(f"Cannot convert {type(value)} to Metadata")

        def after_validate(value: Any) -> "Metadata":
            if isinstance(value, dict):
                return cls(value)
            return value

        return union_schema(
            [
                no_info_after_validator_function(after_validate, dict_schema()),
                no_info_after_validator_function(validate_dict, dict_schema()),
            ]
        )

    def __str__(self) -> str:
        return str(dict(self))


class Message(BaseModel):
    """A message represents a single message from a user, model, or API etc..."""

    # TODO: Non-text content
    content: str
    role: MessageRoleType
    tool_use: Optional[Dict[str, Any]] = Field(default=None)
    metadata: Metadata = Field(default_factory=Metadata, validate_default=True)

    def __init__(
        self,
        content: str,
        role: MessageRoleType,
        metadata: Optional[Dict[str, Any] | Metadata] = None,
        tool_use: Optional[Dict[str, Any]] = None,
    ):
        metadata = (
            # Message save the snapshot of metadata, so we need to copy it.
            metadata.copy()
            if isinstance(metadata, Metadata)
            else Metadata(metadata)
            if isinstance(metadata, dict)
            else Metadata()
        )

        super().__init__(
            content=content, role=role, metadata=metadata, tool_use=tool_use
        )

    def __hash__(self) -> int:
        return hash((self.content, self.role))

    def __str__(self) -> str:
        parts = []

        # Handle content with proper quoting
        content = truncate_string(self.content)
        if "\n" in content:
            parts.append(f'content="""\n{content}\n"""')
        else:
            parts.append(f'content="{content}"')

        parts.append(f'role="{self.role}"')

        if self.metadata:
            parts.append(f"metadata={truncate_string(str(self.metadata))}")
        if self.tool_use:
            parts.append(f"tool_use={truncate_string(str(self.tool_use))}")

        return f"Message({', '.join(parts)})"


class Config(BaseModel):
    """Unified configuration base class.

    Centralizes configuration and parameter management with integrated validation.
    """

    # Core Configuration
    cache_provider: Optional["CacheProvider"] = None
    """Cache provider to cache the response from the model."""
    mock_provider: Optional["MockProvider"] = None
    """Mock provider to mock the response from the model."""

    # Core Parameters
    tools: Optional[List["Tool"]] = None
    """Optional list of tools that can be used by the model."""

    # Common Model Settings
    model_name: str
    """Name of the model to use."""
    temperature: Optional[float] = 1.0
    """Temperature for sampling."""
    max_tokens: Optional[int] = 1024
    """Maximum number of tokens to generate."""

    # Common Optional Parameters
    top_p: Optional[float] = None
    """Top-p (nucleus) sampling parameter."""
    top_k: Optional[int] = None
    """Top-k sampling parameter."""
    repetition_penalty: Optional[float] = None
    """Repetition penalty parameter."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_providers(self) -> "Config":
        """Validate that only one provider is used."""
        if self.cache_provider is not None and self.mock_provider is not None:
            raise ValueError("You can only use either cache_provider or mock_provider.")
        return self

    def validate_all(self, session: Optional["Session"] = None) -> None:
        """Integrated validation method"""
        self._validate_providers()
        self._validate_model_settings()
        if session:
            self.validate_session(session)
        if self.tools:
            self._validate_tools()

    def _validate_providers(self) -> None:
        """Validate cache and mock providers"""
        if self.cache_provider and self.mock_provider:
            raise ValueError("Cannot use both cache and mock providers")

    def _validate_model_settings(self) -> None:
        """Validate model-specific settings."""
        if not self.model_name:
            raise ValueError("model_name is required")
        if self.temperature is not None:
            if self.temperature < 0 or self.temperature > 2:
                raise ValueError("temperature must be between 0 and 2")
        if self.max_tokens is not None:
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be greater than 0")

    def validate_session(self, session: "Session", is_async: bool = False) -> None:
        """Perform basic session validation."""
        # Filter out control template messages
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        # Check for empty session
        if len(messages) == 0:
            raise ParameterValidationError("Session must have at least one message")

        # Check message content type
        if any([not isinstance(message.content, str) for message in messages]):
            raise ParameterValidationError("All messages in session must be strings")

        # Check system message position
        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages:
            if len(system_messages) > 1:
                raise ParameterValidationError("Only one system message is allowed")
            if messages[0].role != "system":
                raise ParameterValidationError(
                    "System message must be at the beginning"
                )

        # Check for empty messages
        if any([message.content == "" for message in messages]):
            raise ParameterValidationError("Empty messages are not allowed")

    def _validate_tools(self) -> None:
        """Validate tools configuration"""
        if not isinstance(self.tools, list):
            raise ValueError("tools must be a list")
        for tool in self.tools:
            if not tool.name:
                raise ValueError("Tool name is required")
            if not tool.description:
                raise ValueError("Tool description is required")


class Session(BaseModel):
    """A session represents a conversation between a user and a model, or API etc..."""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, validate_assignment=True
    )

    # Session is a list of messages with some metadata
    messages: List[Message] = Field(default_factory=list)
    metadata: Metadata = Field(
        default_factory=lambda: Metadata(), validate_default=True
    )

    # Runner and template related fields
    runner: Optional["Runner"] = Field(default=None, exclude=True)
    debug_mode: bool = Field(default=False)
    stack: List["Stack"] = Field(default_factory=list)
    jump_to_id: Optional[str] = Field(default=None)

    def __init__(
        self,
        messages: List[Message] = [],
        metadata: Optional[Dict[str, Any] | Metadata] = None,
        runner: Optional["Runner"] = None,
        debug_mode: bool = False,
        stack: List["Stack"] = [],
        jump_to_id: Optional[str] = None,
    ) -> None:
        metadata = (
            metadata
            if isinstance(metadata, Metadata)
            else Metadata(metadata)
            if isinstance(metadata, dict)
            else Metadata()
        )
        super().__init__(
            messages=messages,
            metadata=metadata,
            runner=runner,
            debug_mode=debug_mode,
            stack=stack,
            jump_to_id=jump_to_id,
        )

    def __hash__(self) -> int:
        return hash(tuple(self.messages))

    def append(self, message: Message) -> None:
        """Append a message to the session."""
        messages_list = list(self.messages)
        messages_list.append(message)
        self.messages = messages_list

    def get_last(self) -> Message:
        """Get the last message in the session."""
        if not self.messages:
            raise IndexError("Session has no messages")
        return self.messages[-1]

    def get_last_message(self) -> Message:
        """Alias for get_last()."""
        return self.get_last()

    def get_current_template_id(self) -> Optional[str]:
        """Get the ID of the current template."""
        if not self.stack:
            return None
        return self.stack[-1].template_id

    def push_stack(self, stack: "Stack") -> None:
        """Push a stack frame."""
        self.stack.append(stack)

    def pop_stack(self) -> "Stack":
        """Pop a stack frame."""
        if not self.stack:
            raise IndexError("Stack is empty")
        return self.stack.pop()

    def head_stack(self) -> "Stack":
        """Get the top stack frame."""
        if not self.stack:
            raise IndexError("Stack is empty")
        return self.stack[-1]

    def get_jump(self) -> Optional[str]:
        """Get the jump target template ID."""
        return self.jump_to_id

    def set_jump(self, jump_to_id: Optional[str]) -> None:
        """Set the jump target template ID."""
        self.jump_to_id = jump_to_id

    def __str__(self) -> str:
        """Return a string representation of the session."""
        messages_str = "\n".join(f"    {msg}" for msg in self.messages)
        template_id = self.get_current_template_id()
        return f"""Session(
    messages=[
{messages_str}
    ],
    current_template={template_id},
    jump={self.jump_to_id},
    debug_mode={self.debug_mode}
)"""


class Model(BaseModel, ABC, Debuggable):
    """Class defining the interface for interaction with LLM models."""

    configuration: Config
    logger: Optional[logging.Logger] = None
    enable_logging: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.setup_logger_for_pydantic()

    def is_mocked(self) -> bool:
        """Return whether the model is mocked."""
        return self.configuration.mock_provider is not None

    def is_cached(self) -> bool:
        """Return whether the model is using cache."""
        return self.configuration.cache_provider is not None

    @abstractmethod
    def _send(self, session: Session) -> Message:
        """Abstract method for sending messages to the model."""
        raise NotImplementedError("Any model should implement _send method.")

    def format_tool(self, tool: "Tool") -> Dict[str, Any]:
        """Convert tool to model-specific format."""
        raise NotImplementedError(
            "Any model should implement format_tool method if it can use tools."
        )

    def format_tool_result(self, result: "ToolResult") -> Dict[str, Any]:
        """Convert tool execution result to model-specific format."""
        raise NotImplementedError(
            "Any model should implement format_tool_result method if it can use tools."
        )

    def prepare(self, session: Optional[Session] = None) -> Session:
        """Perform session preprocessing."""
        if session is None:
            session = Session()

        # 統合されたバリデーション
        self.configuration.validate_all(session)
        return session

    def send(self, session: Session) -> Message:
        """Define standard procedure for sending messages to the model."""
        if self.configuration.mock_provider is not None:
            self.configuration.validate_all(session)
            return self.configuration.mock_provider.call(session)

        session = self.prepare(session)
        self.debug("Communications %s", session)

        if self.configuration.cache_provider is not None:
            message = self.configuration.cache_provider.search(
                self.configuration, session
            )
            if message is not None:
                return message

        message = self._send(session)
        return self.after_send(session, message)

    def _send_async(
        self,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """Abstract method for receiving responses asynchronously."""
        raise NotImplementedError("Async method is not implemented for this model.")

    def send_async(
        self,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """Define standard procedure for sending messages asynchronously."""
        message: Optional[Message] = None
        if self.configuration.cache_provider is not None:
            message = self.configuration.cache_provider.search(
                self.configuration, session
            )
        if self.configuration.mock_provider is not None:
            message = self.configuration.mock_provider.call(session)
        if message is not None:
            if yield_type == "all":
                seq = ""
                for char in message.content:
                    seq = seq + char
                    yield Message(content=seq, role=message.role)
            else:
                for char in message.content:
                    yield Message(content=char, role=message.role)
            return

        session = self.prepare(session)
        messages = self._send_async(session, yield_type)
        for message in messages:
            yield self.after_send(session, message)

    def after_send(
        self,
        session: Optional[Session],
        message: Message,
    ) -> Message:
        """Perform message post-processing."""
        return message

    def validate_session(self, session: Session, is_async: bool = False) -> None:
        """Perform session validation.

        Args:
            session: Session to validate
            is_async: Whether validation is for asynchronous processing

        Raises:
            ParameterValidationError: Validation error
        """
        # Filter out control template messages
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        # Check for empty session
        if len(messages) == 0:
            raise ParameterValidationError("Session must have at least one message")

        # Check message content type
        if any([not isinstance(message.content, str) for message in messages]):
            raise ParameterValidationError("All messages in session must be strings")

        # Check for empty messages
        if any([message.content == "" for message in messages]):
            raise ParameterValidationError("Empty messages are not allowed")

    def list_models(self) -> List[str]:
        """Return a list of available models."""
        raise NotImplementedError("List models is not implemented for this model.")
