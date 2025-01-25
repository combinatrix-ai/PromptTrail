"""Mock providers for testing LLM interactions without calling actual APIs."""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict

from prompttrail.core.const import CONTROL_TEMPLATE_ROLE

if TYPE_CHECKING:
    from prompttrail.core import Message, MessageRoleType, Session

logger = logging.getLogger(__name__)


class MockProvider(ABC):
    """Abstract base class for mock providers.

    Mock providers simulate LLM responses without making actual API calls.
    Subclasses must implement the call() method to define response behavior.
    """

    @abstractmethod
    def call(self, session: "Session") -> "Message":
        """Return a mock response for the given session.

        Parameters
        ----------
        session : Session
            The conversation session including all messages

        Returns
        -------
        Message
            The mocked response message
        """


class OneTurnConversationMockProvider(MockProvider):
    """Mock provider that returns predefined responses based on the last message.

    Parameters
    ----------
    conversation_table : Dict[str, Message]
        Mapping of input messages to their predefined responses
    """

    def __init__(self, conversation_table: Dict[str, "Message"]):
        self.conversation_table = conversation_table

    def call(self, session: "Session") -> "Message":
        valid_messages = [
            x for x in session.messages if x.role != CONTROL_TEMPLATE_ROLE
        ]
        if not valid_messages:
            raise ValueError("No valid messages found in session")

        last_message = valid_messages[-1]
        if last_message.content in self.conversation_table:
            return self.conversation_table[last_message.content]
        else:
            raise ValueError(f"Unexpected message content: {last_message.content}")


class FunctionalMockProvider(MockProvider):
    """Mock provider that generates responses using a custom function.

    Parameters
    ----------
    func : Callable[[Session], Message]
        Function that takes a session and returns a response message
    """

    def __init__(self, func: Callable[["Session"], "Message"]):
        self.func = func

    def call(self, session: "Session") -> "Message":
        return self.func(session)


class EchoMockProvider(FunctionalMockProvider):
    """Mock provider that echoes back the last message with a specified role.

    Parameters
    ----------
    role : MessageRoleType
        Role to assign to the echo response messages
    """

    def __init__(self, role: "MessageRoleType"):
        from prompttrail.core import Message

        super().__init__(
            lambda session: Message(content=session.messages[-1].content, role=role)
        )
        self.role = role
