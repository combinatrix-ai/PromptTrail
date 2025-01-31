import logging
from abc import abstractmethod
from typing import Dict, Optional

from prompttrail.core import Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.utils import Debuggable

logger = logging.getLogger(__name__)


class UserInterface(Debuggable):
    @abstractmethod
    def ask(
        self,
        session: Session,
        description: Optional[str],
        default: Optional[str] = None,
    ) -> str:
        """
        Ask the user for input.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The user's input as a string.
        """
        raise NotImplementedError("ask method is not implemented")


class CLIInterface(UserInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = "Input>",
        default: Optional[str] = None,
    ) -> str:
        """
        Ask the user for input via the command line interface.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The user's input as a string.
        """
        raw = input(description).strip()
        while 1:
            if (not raw) and default is not None:
                self.info("No input. Using default value: %s", default)
                raw = default
            if raw:
                break
            else:
                self.warning(
                    "You must input something or set default value for template. Please try again."
                )
                raw = input(description).strip()
        return raw


class MockInterface(UserInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Base mock provider that returns an empty string.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            An empty string.
        """
        return ""


class SingleTurnResponseMockInterface(MockInterface):
    def __init__(self, conversation_table: Dict[str, str]):
        self.conversation_table = conversation_table

    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Mock the user interaction by providing pre-defined responses based on the conversation history.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The pre-defined response based on the conversation history.
        """
        valid_messages = [
            x for x in session.messages if x.role != CONTROL_TEMPLATE_ROLE
        ]
        last_message = valid_messages[-1].content
        if last_message not in self.conversation_table:
            raise ValueError("No conversation found for " + last_message)
        return self.conversation_table[last_message]


class EchoMockInterface(MockInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Mock the user interaction by echoing the last message.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The last message as the user's input.
        """
        return session.get_last().content


class DefaultOrEchoMockInterface(MockInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        prompt = description if description else "Enter your message"
        if default:
            prompt += f" (default: {default})"
        prompt += ": "

        user_input = input(prompt)
        if not user_input and default:
            return default
        return user_input
