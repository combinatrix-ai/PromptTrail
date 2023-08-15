import logging
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from prompttrail.core import Message, Model, Parameters, Session

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from prompttrail.agent.runner import Runner
    from prompttrail.agent.template import Template, TemplateId


class StatefulMessage(Message):
    data: Dict[str, Any] = {}
    template_id: Optional[str] = None

    def __str__(self) -> str:
        # construct json
        return (
            "StatefulMessage(\n"
            + pformat(
                {
                    "content": self.content,
                    "data": self.data,
                    "template_id": self.template_id,
                    "sender": self.sender,
                }
            )
            + ",\n)"
        )

    def __hash__(self) -> int:
        # TODO: How to hash this?
        return hash(str(self))


class StatefulSession(Session):
    data: Dict[str, Any] = {}
    messages: Sequence[StatefulMessage] = []


# TODO: Use control message below to make control flow more explicit.
# class ControlMessage(StatefulMessage):
#     content: Any = None

#     # ControlMessage is not expected to edit by human, so kill the edit method.
#     def __init__(self, sender: str, template_id: str, data: Dict[str, Any]):
#         self.template_id = template_id
#         self.data = data
#         self.sender = sender


class FlowState(object):
    """FlowState hold the all state of the conversation."""

    def __init__(
        self,
        runner: Optional["Runner"] = None,
        model: Optional[Model] = None,
        parameters: Optional[Parameters] = None,
        data: Dict[str, Any] = {},
        session_history: StatefulSession = StatefulSession(),
        current_template_id: Optional["TemplateId"] = None,
        jump_to_id: Optional["TemplateId"] = None,
    ):
        self.runner = runner
        self.model = model
        self.parameters = parameters
        self.data = data
        self.session_history = session_history
        self.current_template_id = current_template_id
        self.jump_to_id = jump_to_id

    def get_last_message(self) -> StatefulMessage:
        if len(self.session_history.messages) == 0:
            raise IndexError("Session has no message.")
        return self.session_history.messages[-1]

    def get_current_data(self) -> Dict[str, Any]:
        return self.data

    def get_jump(self) -> Optional["TemplateId"]:
        return self.jump_to_id

    def set_jump(self, jump_to_id: Optional["TemplateId"]) -> None:
        self.jump_to_id = jump_to_id

    def get_current_template(self) -> "Template":
        if self.current_template_id is None:
            raise ValueError("Current template is not set.")
        if self.runner is None:
            raise ValueError("Runner is not set.")
        return self.runner.search_template(self.current_template_id)

    def __str__(self):
        # Create PrettyPrinted string
        data_json_line_list = "\n".join(
            ["    " + line for line in pformat(self.data).splitlines()]
        )
        message_history_json = "\n".join(
            [
                "    " + line
                for line in pformat(
                    [
                        {"content": mes.content, "sender": mes.sender}
                        for mes in self.session_history.messages
                    ]
                ).splitlines()
            ]
        )
        # If runner is set, template can be searched.
        if self.runner is not None:
            current_template_json = pformat(
                self.runner.search_template(self.current_template_id)
                if self.current_template_id is not None
                else self.current_template_id
            )
        else:
            current_template_json = self.current_template_id
        jump_json = pformat(self.jump_to_id)

        return f"""FlowState(
    data=
    {data_json_line_list},
    message_history=
    {message_history_json},
    current_template={current_template_json},
    jump={jump_json},
)"""
