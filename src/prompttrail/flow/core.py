import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeAlias

from src.prompttrail.core import Message, Model, Parameters, Session

logger = logging.getLogger(__name__)

TemplateId: TypeAlias = str

if TYPE_CHECKING:
    from src.prompttrail.flow.templates import Template


class StatefulMessage(Message):
    data: Dict[str, Any] = {}
    template_id: Optional[str] = None


class StatefulSession(Session):
    data: Dict[str, Any] = {}
    messages: List[StatefulMessage] = []


class FlowState(object):
    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        data: Dict[str, Any] = {},
        session_history: StatefulSession = StatefulSession(),
        jump: Optional["TemplateId | Template"] = None,
    ):
        self.data = data
        self.model = model
        self.parameters = parameters
        self.session_history = session_history
        self.current_template: Optional["Template"] = None
        self.jump: "TemplateId | Template | None" = jump

    def get_last_message(self) -> StatefulMessage:
        if len(self.session_history.messages) == 0:
            raise IndexError("Session has no message.")
        return self.session_history.messages[-1]

    def get_current_data(self) -> Dict[str, Any]:
        return self.data

    def get_jump(self) -> "TemplateId | Template | None":
        return self.jump

    def set_jump(self, jump: "TemplateId | Template | None") -> None:
        self.jump = jump

    def get_current_template(self) -> "Template":
        if self.current_template is None:
            raise Exception("Current template is not set.")
        return self.current_template
