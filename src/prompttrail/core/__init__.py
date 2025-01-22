import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from prompttrail.agent.runners import Runner
    from prompttrail.agent.templates import Stack
    from prompttrail.agent.tools import Tool, ToolResult
else:
    Runner = Any  # type: ignore
    Stack = Any  # type: ignore
    Tool = Any  # type: ignore
    ToolResult = Any  # type: ignore

from prompttrail.core.cache import CacheProvider
from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.mocks import MockProvider
from prompttrail.core.utils import logger_multiline

logger = logging.getLogger(__name__)


# Define standard message roles
MessageRoleType = Literal["system", "user", "assistant", "tool_result", "control"]


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate string to max_length."""
    if len(s) > max_length:
        return s[: max_length - 3] + "..."
    return s


class Message(BaseModel):
    """A message represents a single message from a user, model, or API etc..."""

    # TODO: Non-text content
    content: str
    role: MessageRoleType
    tool_use: Optional[Dict[str, Any]] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

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


class Configuration(BaseModel):
    """A configuration represents a set of data that is used to configure model."""

    cache_provider: Optional["CacheProvider"] = None
    """Cache provider to cache the response from the model."""
    mock_provider: Optional["MockProvider"] = None
    """Mock provider to mock the response from the model."""

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def either_cache_or_mock(self) -> "Configuration":
        if self.cache_provider is not None and self.mock_provider is not None:
            raise ValueError("You can only use either cache_provider or mock_provider.")
        return self


class Parameters(BaseModel):
    """A parameters represents a set of data that is used to define the behavior of the model at runtime."""

    # Parameters is a set of data that is used to define the behavior of the model.
    tools: Optional[List["Tool"]] = None
    """Optional list of tools that can be used by the model."""


class Session(BaseModel):
    """A session represents a conversation between a user and a model, or API etc..."""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, validate_assignment=True
    )

    # Session is a list of messages with some metadata
    messages: List[Message] = Field(default_factory=list)
    initial_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Runner and template related fields
    runner: Optional["Runner"] = Field(default=None, exclude=True)
    debug_mode: bool = Field(default=False)
    stack: List["Stack"] = Field(default_factory=list)
    jump_to_id: Optional[str] = Field(default=None)

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

    def get_latest_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last message or initial metadata if no messages exist."""
        if not self.messages:
            return self.initial_metadata.copy()
        return self.messages[-1].metadata

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


# TypeAlias to let users know that they return updated parameters and session.
UpdatedConfiguration: TypeAlias = Configuration
UpdatedParameters: TypeAlias = Parameters
UpdatedMessage: TypeAlias = Message


class Model(BaseModel, ABC):
    """A model define an interface to interact with LLM models."""

    configuration: Configuration

    def is_mocked(self) -> bool:
        """is_mocked method returns True if the model is mocked."""
        return self.configuration.mock_provider is not None

    def is_cached(self) -> bool:
        """is_cached method returns True if the model use cache."""
        return self.configuration.cache_provider is not None

    @abstractmethod
    def _send(self, parameters: Parameters, session: Session) -> Message:
        """A model should implement _send method to send a message to the model."""
        raise NotImplementedError("Any model should implement _send method.")

    def format_tool(self, tool: "Tool") -> Dict[str, Any]:
        """Convert tool to model-specific format"""
        raise NotImplementedError(
            "Any model should implement format_tool method if it can use tools."
        )

    def format_tool_result(self, result: "ToolResult") -> Dict[str, Any]:
        """Convert tool result to model-specific format"""
        raise NotImplementedError(
            "Any model should implement format_tool_result method if it can use tools."
        )

    def validate_tools(self, tools: List["Tool"]) -> None:
        """Validate tools according to model-specific requirements"""
        raise NotImplementedError(
            "Any model should implement validate_tools method if it can use tools."
        )

    def prepare(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[UpdatedParameters, Session]:
        """prepare method defines the standard procedure to pre/post-process parameters and session."""
        if session is None:
            session = Session()

        self.validate_configuration(self.configuration, False)
        self.validate_parameters(parameters, False)
        self.validate_session(session, False)
        if parameters.tools:
            self.validate_tools(parameters.tools)
        self.vaidate_other(parameters, session, False)
        configuration_, parameters_, session_ = self.before_send(
            parameters, session, False
        )
        if configuration_ is not None:
            self.configuration = configuration_
        if parameters_ is not None:
            parameters = parameters_
        if session_ is not None:
            session = session_
        return parameters, session

    def send(self, parameters: Parameters, session: Session) -> Message:
        """send method defines the standard procedure to send a message to the model."""
        if self.configuration.mock_provider is not None:
            # Even with mock provider, we still need to validate
            self.validate_configuration(self.configuration, False)
            self.validate_parameters(parameters, False)
            self.validate_session(session, False)
            if parameters.tools:
                self.validate_tools(parameters.tools)
            return self.configuration.mock_provider.call(session)

        parameters, session = self.prepare(parameters, session, False)

        if self.configuration.cache_provider is not None:
            message = self.configuration.cache_provider.search(parameters, session)
            if message is not None:
                return message

        message = self._send(parameters, session)
        logger_multiline(logger, f"Message from Provider: {message}", logging.DEBUG)
        return self.after_send(parameters, session, message, False)

    def _send_async(
        self,
        parameters: Parameters,
        session: Session,
        yiled_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """A model should implement _send_async method to receive response asynchronously."""
        raise NotImplementedError("Async method is not implemented for this model.")

    def send_async(
        self,
        parameters: Parameters,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """send_async method defines the standard procedure to send a message to the model asynchronously."""
        message: Optional[Message] = None
        if self.configuration.cache_provider is not None:
            message = self.configuration.cache_provider.search(parameters, session)
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
        parameters, session = self.prepare(parameters, session, True)
        messages = self._send_async(parameters, session, yield_type)
        for message in messages:
            yield self.after_send(parameters, session, message, True)

    def validate_configuration(
        self, configuration: Configuration, is_async: bool
    ) -> None:
        """validate_configuration method define the concrete procedure to validate configuration."""
        ...

    def validate_parameters(self, parameters: Parameters, is_async: bool) -> None:
        """validate_parameters method define the concrete procedure to validate parameters."""
        ...

    def validate_session(self, session: Session, is_async: bool) -> None:
        """validate_session method defines the basic validation procedure for sessions."""
        if len(session.messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should be a Session object and have at least one message."
            )
        if any([not isinstance(message.content, str) for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )

        # Filter out control template messages for validation
        messages = [
            message for message in session.messages if message.role != "control"
        ]

        # Check for system messages
        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages:
            if len(system_messages) > 1:
                raise ParameterValidationError(
                    f"{self.__class__.__name__}: Only one system message is allowed in a session."
                )
            if messages[0].role != "system":
                raise ParameterValidationError(
                    f"{self.__class__.__name__}: System message must be at the beginning of the session."
                )

    def vaidate_other(
        self, parameters: Parameters, session: Session, is_async: bool
    ) -> None:
        """vaidate_other method define the concrete procedure to validate other parameters and session."""
        ...

    def before_send(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[
        Optional[UpdatedConfiguration],
        Optional[UpdatedParameters],
        Optional[Session],
    ]:
        """before_send method define the concrete procedure to pre-process parameters and session."""
        return (None, None, None)

    def after_send(
        self,
        parameters: Parameters,
        session: Optional[Session],
        message: Message,
        is_async: bool,
    ) -> UpdatedMessage:
        """after_send method define the concrete procedure to post-process parameters, session, and message."""
        return message

    def list_models(self) -> List[str]:
        """list_models method define the concrete procedure to list models."""
        raise NotImplementedError("List models is not implemented for this model.")
