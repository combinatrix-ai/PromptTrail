import logging
from abc import abstractmethod
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, TypeAlias

from pydantic import BaseModel, ConfigDict

from prompttrail.cache import CacheProvider
from prompttrail.util import logger_multiline

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A message represents a single message from a user, model, or API etc..."""

    # We may soon get non-textual messages maybe, so we should prepare for that.
    content: Any
    sender: Optional[str] = None

    # Store extra information in dict
    # TODO: Update __str__ to include this
    data: Dict[str, Any] = {}

    def __hash__(self) -> int:
        return hash((self.content, self.sender))


class TextMessage(Message):
    """A message that accepts only text."""

    # However, text is the most common message type, so we should have a shortcut for it.
    content: str

    def __str__(self) -> str:
        if "\n" in self.content:
            content_part = 'content="""\n' + self.content + '\n"""'
        else:
            content_part = 'content="' + self.content + '"'
        if self.sender is None:
            return "TextMessage(" + content_part + '")'
        return "TextMessage(" + content_part + ', sender="' + self.sender + '")'


class Session(BaseModel):
    """A session represents a conversation between a user and a model, or API etc..."""

    # Session is a list of messages with some metadata
    # Sequence is used to ensure that the session is covariant with the message type.
    messages: Sequence[Message] = []

    def __hash__(self) -> int:
        return hash(tuple(self.messages))


class TextSession(Session):
    """A session that accepts only text messages."""

    messages: Sequence[TextMessage] = []


class Configuration(BaseModel):
    """A configuration represents a set of data that is used to configure model."""

    # Configuration is a set of data that is used to configure model.
    # The name is following the naming convention of openai.
    # Configuration does not have parameters that define the behavior of the model.

    cache_provider: Optional[CacheProvider] = None

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Parameters(BaseModel):
    """A parameters represents a set of data that is used to define the behavior of the model at runtime."""

    # Parameters is a set of data that is used to define the behavior of the model.
    ...


# TypeAlias to let users know that they return updated parameters and session.
UpdatedConfiguration: TypeAlias = Configuration
UpdatedParameters: TypeAlias = Parameters
UpdatedSession: TypeAlias = Session
UpdatedMessage: TypeAlias = Message


class Model(BaseModel):
    """A model define an interface to interact with LLM models."""

    configuration: Configuration

    @abstractmethod
    def _send(self, parameters: Parameters, session: Session) -> Message:
        """A model should implement _send method to send a message to the model. You must implement this method to create a new model."""
        raise NotImplementedError("Any model should implement _send method.")

    def prepare(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[UpdatedParameters, UpdatedSession]:
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

        parameters, session = self.prepare(parameters, session, False)
        message = self._send(parameters, session)
        logger_multiline(logger, f"Message from Provider: {message}", logging.DEBUG)
        message_ = self.after_send(parameters, session, message, False)
        if message_ is not None:
            message = message_
        return message

    def _send_async(
        self, parameters: Parameters, session: Session
    ) -> Generator[Message, None, None]:
        """A model should implement _send_async method to receive response asynchronously. You must implement this method to create a new model if you need async feature."""
        raise NotImplementedError("Async method is not implemented for this model.")

    def send_async(
        self, parameters: Parameters, session: Optional[Session] = None
    ) -> Generator[Message, None, None]:
        """send_async method defines the standard procedure to send a message to the model asynchronously. You dont need to override this method usually."""
        parameters, session = self.prepare(parameters, session, True)
        messages = self._send_async(parameters, session)
        for message in messages:
            message_ = self.after_send(parameters, session, message, True)
            if message_ is not None:
                message = message_
            yield message

    def validate_configuration(
        self, configuration: Configuration, is_async: bool
    ) -> None:
        """validate_configuration method define the concrete procedure to validate configuration. You can override this method to add validation logic."""
        ...

    def validate_parameters(self, parameters: Parameters, is_async: bool) -> None:
        """validate_parameters method define the concrete procedure to validate parameters. You can override this method to add validation logic."""
        ...

    def validate_session(self, session: Session, is_async: bool) -> None:
        """validate_session method define the concrete procedure to validate session. You can override this method to add validation logic."""
        ...

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
        Optional[UpdatedSession],
    ]:
        """before_send method define the concrete procedure to pre-process parameters and session. You can override this method to add pre-processing logic."""
        return (None, None, None)

    def after_send(
        self,
        parameters: Parameters,
        session: Optional[Session],
        message: Message,
        is_async: bool,
    ) -> Optional[UpdatedMessage]:
        """after_send method define the concrete procedure to post-process parameters, session, and message. You can override this method to add post-processing logic."""
        return None

    def list_models(self) -> List[str]:
        """list_models method define the concrete procedure to list models. You can override this method to add list_models logic."""
        raise NotImplementedError("List models is not implemented for this model.")
