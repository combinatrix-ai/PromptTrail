import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel

from prompttrail.core import Message, Session, TextMessage

logger = logging.getLogger(__name__)


class MockProvider(ABC):
    """A mock provider is an abstract class that should be inherited by any mock provider to ensure implementation of `call` method, which is used to actually define the response."""

    @abstractmethod
    def call(self, session: Session) -> Message:
        ...


class MockModel(ABC, BaseModel):
    mock_provider: Optional[MockProvider] = None

    class Config:
        arbitrary_types_allowed = True

    """ A mock model is an abstract classs that should be inherited by any mock model to ensure implementation of `setup` method, which is used to inject the mock provider. """

    @abstractmethod
    def setup(self, mock_provider: MockProvider):
        """A mock model use the same interface as the model. To override the behavior of the model, you must implement the setup method to inject the mock provider."""
        ...


class OneTurnConversationMockProvider(MockProvider):
    """A mock provider that returns a predefined response based on the last message."""

    def __init__(self, conversation_table: Dict[str, str], sender: str):
        self.conversation_table = conversation_table
        self.sender = sender

    def call(self, session: Session) -> Message:
        valid_messages = [x for x in session.messages if x.sender != "prompttrail"]
        if len(valid_messages) == 0:
            logger.warning("No message is passed to OneTurnConversationMockProvider.")
            return TextMessage(content="Hello", sender=self.sender)
        last_message = valid_messages[-1]
        if last_message.content in self.conversation_table:
            return TextMessage(
                content=self.conversation_table[last_message.content],
                sender=self.sender,
            )
        else:
            raise ValueError(
                "Unexpected message is passed to mock provider: " + last_message.content
            )
