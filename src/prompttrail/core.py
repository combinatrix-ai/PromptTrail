import logging
from abc import abstractmethod
from typing import Any, Generator, List, Optional, Sequence, Tuple, TypeAlias

from pydantic import BaseModel

from prompttrail.util import logger_multiline

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A message represents a single message from a user, model, or API etc..."""

    # We may soon get non-textual messages maybe, so we should prepare for that.
    content: Any
    sender: Optional[str] = None


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


class TextSession(Session):
    messages: Sequence[TextMessage] = []


class Configuration(BaseModel):
    # Configuration is a set of data that is used to configure model.
    # The name is following the naming convention of openai.
    # Configuration does not have parameters that define the behavior of the model.
    ...


class Parameters(BaseModel):
    # Parameters is a set of data that is used to define the behavior of the model.
    ...


# TypeAlias to let users know that they return updated parameters and session.
UpdatedConfiguration: TypeAlias = Configuration
UpdatedParameters: TypeAlias = Parameters
UpdatedSession: TypeAlias = Session
UpdatedMessage: TypeAlias = Message


class Model(BaseModel):
    configuration: Configuration

    @abstractmethod
    def _send(self, parameters: Parameters, session: Session) -> Message:
        raise NotImplementedError("Any model should implement _send method.")

    def prepare(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[UpdatedParameters, UpdatedSession]:
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
        raise NotImplementedError("Async method is not implemented for this model.")

    def send_async(
        self, parameters: Parameters, session: Optional[Session] = None
    ) -> Generator[Message, None, None]:
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
        ...

    def validate_parameters(self, parameters: Parameters, is_async: bool) -> None:
        ...

    def validate_session(self, session: Session, is_async: bool) -> None:
        ...

    def vaidate_other(
        self, parameters: Parameters, session: Session, is_async: bool
    ) -> None:
        ...

    def before_send(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[
        Optional[UpdatedConfiguration],
        Optional[UpdatedParameters],
        Optional[UpdatedSession],
    ]:
        return (None, None, None)

    def after_send(
        self,
        parameters: Parameters,
        session: Optional[Session],
        message: Message,
        is_async: bool,
    ) -> Optional[UpdatedMessage]:
        return None

    def list_models(self) -> List[str]:
        raise NotImplementedError("List models is not implemented for this model.")
