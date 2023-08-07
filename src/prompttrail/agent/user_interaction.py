from abc import abstractmethod
from typing import Any, Dict, Optional

from prompttrail.agent.core import FlowState


class UserInteractionProvider(object):
    @abstractmethod
    def ask(self, flow_state: FlowState, description: Any, default: Any = None) -> Any:
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextProvider(UserInteractionProvider):
    @abstractmethod
    def ask(
        self, flow_state: FlowState, description: str, default: Optional[str] = None
    ) -> str:
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextCLIProvider(UserInteractionTextProvider):
    def ask(
        self, flow_state: FlowState, description: str, default: Optional[str] = None
    ) -> str:
        raw = input(description).strip()
        if (not raw) and default is not None:
            raw = default
        return raw


class OneTurnConversationUserInteractionTextMockProvider(UserInteractionTextProvider):
    def __init__(self, conversation_table: Dict[str, str]):
        self.conversation_table = conversation_table

    def ask(
        self, flow_state: FlowState, description: str, default: Optional[str] = None
    ) -> str:
        valid_messages = [
            x for x in flow_state.session_history.messages if x.sender != "prompttrail"
        ]
        last_message = valid_messages[-1].content
        if last_message not in self.conversation_table:
            raise ValueError("No conversation found for " + last_message)
        return self.conversation_table[last_message]
