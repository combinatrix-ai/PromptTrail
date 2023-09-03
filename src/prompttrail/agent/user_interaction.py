import logging
from abc import abstractmethod
from typing import Dict, Optional

from prompttrail.agent import State
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE

logger = logging.getLogger(__name__)


class UserInteractionProvider:
    @abstractmethod
    def ask(
        self,
        state: State,
        description: Optional[str],
        default: Optional[str] = None,
    ) -> str:
        """
        Ask the user for input.

        Args:
            state: The current state of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The user's input as a string.
        """
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextCLIProvider(UserInteractionProvider):
    def ask(
        self,
        state: State,
        description: Optional[str] = "Input>",
        default: Optional[str] = None,
    ) -> str:
        """
        Ask the user for input via the command line interface.

        Args:
            state: The current state of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The user's input as a string.
        """
        raw = input(description).strip()
        while 1:
            if (not raw) and default is not None:
                logger.info(f"No input. Using default value: {default}")
                raw = default
            if raw:
                break
            else:
                logger.warning(
                    "You must input something or set default value for template. Please try again."
                )
                raw = input(description).strip()
        return raw


class UserInteractionMockProvider(UserInteractionProvider):
    ...


class OneTurnConversationUserInteractionTextMockProvider(UserInteractionMockProvider):
    def __init__(self, conversation_table: Dict[str, str]):
        self.conversation_table = conversation_table

    def ask(
        self,
        state: State,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Mock the user interaction by providing pre-defined responses based on the conversation history.

        Args:
            state: The current state of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The pre-defined response based on the conversation history.
        """
        valid_messages = [
            x
            for x in state.session_history.messages
            if x.sender != CONTROL_TEMPLATE_ROLE
        ]
        last_message = valid_messages[-1].content
        if last_message not in self.conversation_table:
            raise ValueError("No conversation found for " + last_message)
        return self.conversation_table[last_message]


class EchoUserInteractionTextMockProvider(UserInteractionMockProvider):
    def ask(
        self,
        state: State,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Mock the user interaction by echoing the last message.

        Args:
            state: The current state of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The last message as the user's input.
        """
        return state.get_last_message().content
