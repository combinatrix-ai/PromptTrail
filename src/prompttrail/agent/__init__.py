import logging
from pprint import pformat
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from prompttrail.core import Message, Session

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from prompttrail.agent.runners import Runner
    from prompttrail.agent.templates import Stack


class StatefulMessage(Message):
    """
    A message that holds additional data and template ID.
    """

    data: Dict[str, Any] = {}
    template_id: Optional[str] = None

    def __str__(self) -> str:
        """
        Return a string representation of the message.
        """
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
        """
        Return a hash value of the message.
        """
        return hash(str(self))


class StatefulSession(Session):
    """
    A session that holds additional data and stateful messages.
    """

    data: Dict[str, Any] = {}
    messages: Sequence[StatefulMessage] = []


class State(object):
    """
    State holds all the state of the conversation.
    """

    def __init__(
        self,
        runner: Optional["Runner"] = None,
        data: Optional[Dict[str, Any]] = None,
        session_history: Optional[StatefulSession] = None,
        stack: Optional[Sequence["Stack"]] = None,
        debug_mode: Optional[bool] = None,
    ):
        if session_history is None:
            session_history = StatefulSession()
        self.runner = runner
        self.data = data if data is not None else {}
        self.session_history = session_history
        self.stack: Sequence[Stack] = stack if stack is not None else []
        self.debug_mode = debug_mode

    def get_last_message(self) -> StatefulMessage:
        """
        Get the last message in the session history.
        """
        if len(self.session_history.messages) == 0:
            raise IndexError("Session has no message.")
        return self.session_history.messages[-1]

    def get_current_data(self) -> Dict[str, Any]:
        """
        Get the current data in the state.
        """
        return self.data

    def get_jump(self) -> Optional[str]:
        """
        Get the jump ID in the state.
        """
        return self.jump_to_id

    def set_jump(self, jump_to_id: Optional[str]) -> None:
        """
        Set the jump ID in the state.
        """
        self.jump_to_id = jump_to_id

    def get_current_template_id(self) -> Optional[str]:
        """
        Get the ID of the current template in the state.
        """
        if len(self.stack) == 0:
            return None
        return self.stack[-1].template_id

    def __str__(self):
        """
        Return a string representation of the state.
        """
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
        """
        Push a stack onto the state.
        """
        self.stack.append(stack)  # type: ignore

    def pop_stack(self) -> "Stack":
        """
        Pop a stack from the state.
        """
        return self.stack.pop()  # type: ignore

    def head_jump(self) -> "Stack":
        """
        Get the head of the stack in the state.
        """
        return self.stack[-1]  # type: ignore
