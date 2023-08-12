import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict

from prompttrail.core import Message, Session

logger = logging.getLogger(__name__)


class MockProvider(ABC):
    """A mock provider is an abstract class that should be inherited by any mock provider to ensure implementation of `call` method, which is used to actually define the response."""

    @abstractmethod
    def call(self, session: Session) -> Message:
        ...


class MockModel(ABC, BaseModel):
    mock_provider: Optional[MockProvider] = None

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True)

    """ A mock model is an abstract classs that should be inherited by any mock model to ensure implementation of `setup` method, which is used to inject the mock provider. """

    @abstractmethod
    def setup(self, mock_provider: MockProvider):
        """A mock model use the same interface as the model. To override the behavior of the model, you must implement the setup method to inject the mock provider."""
        ...


class OneTurnConversationMockProvider(MockProvider):
    """A mock provider that returns a predefined response based on the last message."""

    def __init__(self, conversation_table: Dict[str, Message], sender: str):
        self.conversation_table = conversation_table
        self.sender = sender

    def call(self, session: Session) -> Message:
        valid_messages = [x for x in session.messages if x.sender != "prompttrail"]
        if len(valid_messages) == 0:
            logger.warning("No message is passed to OneTurnConversationMockProvider.")
            return Message(content="Hello", sender=self.sender)
        last_message = valid_messages[-1]
        if last_message.content in self.conversation_table:
            return self.conversation_table[last_message.content]
        else:
            raise ValueError(
                "Unexpected message is passed to mock provider: " + last_message.content
            )


class FunctionalMockProvider(MockProvider):
    def __init__(self, func: Callable[[Session], Message]):
        self.func = func

    def call(self, session: Session) -> Message:
        return self.func(session)


class EchoMockProvider(FunctionalMockProvider):
    def __init__(self, sender: str):
        self.func: Callable[[Session], Message] = lambda session: Message(
            content=session.messages[-1].content, sender=sender
        )

    def call(self, session: Session) -> Message:
        return Message(content=session.messages[-1].content)
