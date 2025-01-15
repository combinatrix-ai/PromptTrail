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
else:
    Runner = Any  # type: ignore
    Stack = Any  # type: ignore

from prompttrail.core.cache import CacheProvider
from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.mocks import MockProvider
from prompttrail.core.utils import logger_multiline

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A message represents a single message from a user, model, or API etc..."""

    # We may soon get non-textual messages maybe, so we should prepare for that.
    content: str
    role: Optional[str] = None

    # Store metadata in dict
    metadata: Dict[str, Any] = {}

    def __hash__(self) -> int:
        return hash((self.content, self.role))

    def __str__(self) -> str:
        if "\n" in self.content:
            content_part = 'content="""\n' + self.content + '\n"""'
        else:
            content_part = 'content="' + self.content + '"'
        if self.role is None:
            return "Message(" + content_part + '")'
        return "Message(" + content_part + ', role="' + self.role + '")'


class Configuration(BaseModel):
    """A configuration represents a set of data that is used to configure model."""

    # Configuration is a set of data that is used to configure model.
    # The name is following the naming convention of openai.
    # Configuration does not have parameters that define the behavior of the model.

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
    ...


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
        """A model should implement _send method to send a message to the model. You must implement this method to create a new model."""
        raise NotImplementedError("Any model should implement _send method.")

    def prepare(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[UpdatedParameters, Session]:
        """prepare method defines the standard procedure to pre/post-process parameters and session. You dont need to override this method usually."""
        if session is None:
            session = Session()

        self.validate_configuration(self.configuration, False)
        self.validate_parameters(parameters, False)
        self.validate_session(session, False)
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
        """send method defines the standard procedure to send a message to the model. You dont need to override this method usually."""
        if self.configuration.cache_provider is not None:
            message = self.configuration.cache_provider.search(parameters, session)
            if message is not None:
                return message
        if self.configuration.mock_provider is not None:
            # TODO: Should mock also process parameters?
            return self.configuration.mock_provider.call(session)

        parameters, session = self.prepare(parameters, session, False)
        message = self._send(parameters, session)
        logger_multiline(logger, f"Message from Provider: {message}", logging.DEBUG)
        return self.after_send(parameters, session, message, False)

    def _send_async(
        self,
        parameters: Parameters,
        session: Session,
        yiled_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """A model should implement _send_async method to receive response asynchronously. You must implement this method to create a new model if you need async feature."""
        raise NotImplementedError("Async method is not implemented for this model.")

    def send_async(
        self,
        parameters: Parameters,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """send_async method defines the standard procedure to send a message to the model asynchronously. You dont need to override this method usually."""
        message: Optional[Message] = None
        if self.configuration.cache_provider is not None:
            message = self.configuration.cache_provider.search(parameters, session)
        if self.configuration.mock_provider is not None:
            message = self.configuration.mock_provider.call(session)
        if message is not None:
            # character by character yield
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
        """validate_configuration method define the concrete procedure to validate configuration. You can override this method to add validation logic."""
        ...

    def validate_parameters(self, parameters: Parameters, is_async: bool) -> None:
        """validate_parameters method define the concrete procedure to validate parameters. You can override this method to add validation logic."""
        ...

    def validate_session(self, session: Session, is_async: bool) -> None:
        """validate_session method defines the basic validation procedure for sessions.

        Validates:
        - Session must have at least one message
        - All messages must have string content
        - All messages must have a role

        You can override this method to add model-specific validation logic.
        """
        if len(session.messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should be a Session object and have at least one message."
            )
        if any([not isinstance(message.content, str) for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )
        if any([message.role is None for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have role."
            )

    def vaidate_other(
        self, parameters: Parameters, session: Session, is_async: bool
    ) -> None:
        """vaidate_other method define the concrete procedure to validate other parameters and session. You can override this method to add validation logic."""
        ...

    def before_send(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[
        Optional[UpdatedConfiguration],
        Optional[UpdatedParameters],
        Optional[Session],
    ]:
        """before_send method define the concrete procedure to pre-process parameters and session. You can override this method to add pre-processing logic."""
        return (None, None, None)

    def after_send(
        self,
        parameters: Parameters,
        session: Optional[Session],
        message: Message,
        is_async: bool,
    ) -> UpdatedMessage:
        """after_send method define the concrete procedure to post-process parameters, session, and message. You can override this method to add post-processing logic."""
        return message

    def list_models(self) -> List[str]:
        """list_models method define the concrete procedure to list models. You can override this method to add list_models logic."""
        raise NotImplementedError("List models is not implemented for this model.")
