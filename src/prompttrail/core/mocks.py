import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict

from prompttrail.core.const import CONTROL_TEMPLATE_ROLE

if TYPE_CHECKING:
    from prompttrail.core import Message, Session

logger = logging.getLogger(__name__)


class MockProvider(ABC):
    """A mock provider is an abstract class that should be inherited by any mock provider to ensure implementation of `call` method, which is used to actually define the response."""

    @abstractmethod
    def call(self, session: "Session") -> "Message":
        ...


class OneTurnConversationMockProvider(MockProvider):
    """A mock provider that returns a predefined response based on the last message."""

    def __init__(self, conversation_table: Dict[str, "Message"], sender: str):
        self.conversation_table = conversation_table

    def call(self, session: "Session") -> "Message":
        valid_messages = [
            x for x in session.messages if x.sender != CONTROL_TEMPLATE_ROLE
        ]
        if len(valid_messages) == 0:
            raise ValueError("No valid messages are passed to mock provider.")
        last_message = valid_messages[-1]
        if last_message.content in self.conversation_table:
            return self.conversation_table[last_message.content]
        else:
            raise ValueError(
                "Unexpected message is passed to mock provider: " + last_message.content
            )


class FunctionalMockProvider(MockProvider):
    def __init__(self, func: Callable[["Session"], "Message"]):
        self.func = func

    def call(self, session: "Session") -> "Message":
        return self.func(session)


class EchoMockProvider(FunctionalMockProvider):
    def __init__(self, sender: str):
        # To avoid circular import
        from prompttrail.core import Message

        self.func: Callable[["Session"], "Message"] = lambda session: Message(
            content=session.messages[-1].content, sender=sender
        )

    def call(self, session: "Session") -> "Message":
        # To avoid circular import
        from prompttrail.core import Message

        return Message(content=session.messages[-1].content)
