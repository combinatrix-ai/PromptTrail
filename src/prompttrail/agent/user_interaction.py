import logging
from abc import abstractmethod
from typing import Dict, Optional

from prompttrail.agent.core import State
from prompttrail.const import CONTROL_TEMPLATE_ROLE

logger = logging.getLogger(__name__)


class UserInteractionProvider(object):
    @abstractmethod
    def ask(
        self,
        state: State,
        description: Optional[str],
        default: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextCLIProvider(UserInteractionProvider):
    def ask(
        self,
        state: State,
        description: Optional[str] = "Input>",
        default: Optional[str] = None,
    ) -> str:
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
        return state.get_last_message().content
