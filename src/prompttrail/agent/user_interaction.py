from abc import abstractmethod
from typing import Dict, Optional

from prompttrail.agent.core import FlowState


class UserInteractionProvider(object):
    @abstractmethod
    def ask(
        self,
        flow_state: FlowState,
        description: Optional[str],
        default: Optional[str] = None,
    ) -> str:
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextCLIProvider(UserInteractionProvider):
    def ask(
        self,
        flow_state: FlowState,
        description: Optional[str] = "Input>",
        default: Optional[str] = None,
    ) -> str:
        raw = input(description).strip()
        if (not raw) and default is not None:
            raw = default
        return raw


class UserInteractionMockProvider(UserInteractionProvider):
    ...


class OneTurnConversationUserInteractionTextMockProvider(UserInteractionMockProvider):
    def __init__(self, conversation_table: Dict[str, str]):
        self.conversation_table = conversation_table

    def ask(
        self,
        flow_state: FlowState,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        valid_messages = [
            x for x in flow_state.session_history.messages if x.sender != "prompttrail"
        ]
        last_message = valid_messages[-1].content
        if last_message not in self.conversation_table:
            raise ValueError("No conversation found for " + last_message)
        return self.conversation_table[last_message]


class EchoUserInteractionTextMockProvider(UserInteractionMockProvider):
    def ask(
        self,
        flow_state: FlowState,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        return flow_state.get_last_message().content
