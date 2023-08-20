import logging
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from prompttrail.core import Message, Session

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from prompttrail.agent.runner import Runner
    from prompttrail.agent.template import Stack


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


class State(object):
    """State hold the all state of the conversation."""

    def __init__(
        self,
        runner: Optional["Runner"] = None,
        data: Dict[str, Any] = {},
        session_history: StatefulSession = StatefulSession(),
        stack: Sequence["Stack"] = [],
    ):
        self.runner = runner
        self.data = data
        self.session_history = session_history
        self.stack = stack

    def get_last_message(self) -> StatefulMessage:
        if len(self.session_history.messages) == 0:
            raise IndexError("Session has no message.")
        return self.session_history.messages[-1]

    def get_current_data(self) -> Dict[str, Any]:
        return self.data

    def get_jump(self) -> Optional[str]:
        return self.jump_to_id

    def set_jump(self, jump_to_id: Optional[str]) -> None:
        self.jump_to_id = jump_to_id

    def get_current_template_id(self) -> str:
        return self.stack[-1].template_id

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
        current_template_json = self.get_current_template_id()
        jump_json = pformat(self.jump_to_id)

        return f"""State(
    data=
    {data_json_line_list},
    message_history=
    {message_history_json},
    current_template={current_template_json},
    jump={jump_json},
)"""

    def push_stack(self, stack: "Stack") -> None:
        self.stack.append(stack)  # type: ignore

    def pop_stack(self) -> "Stack":
        return self.stack.pop()  # type: ignore

    def head_jump(self) -> "Stack":
        return self.stack[-1]  # type: ignore
