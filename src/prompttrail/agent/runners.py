import json
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Set, cast

from prompttrail.agent.templates._control import EndTemplate
from prompttrail.agent.templates._core import Event, Template, UserInteractionEvent
from prompttrail.agent.user_interface import UserInterface
from prompttrail.core import Message, MessageRoleType, Model, Session
from prompttrail.core.const import ReachedEndTemplateException
from prompttrail.core.utils import Debuggable

# Session is already imported from prompttrail.core

logger = logging.getLogger(__name__)


class Runner(Debuggable, metaclass=ABCMeta):
    def __init__(
        self,
        model: Model,
        template: "Template",
        user_interface: UserInterface,
    ):
        """Abstract class for runner. Runner is a class to run the templates. It is responsible for rendering templates and handling user interactions."""
        super().__init__()
        self.model = model
        self.user_interface = user_interface
        self.template = template
        self.template_dict: Dict[str, Template] = {}
        visited_templates: Set[Template] = set()
        for next_template in template.walk(visited_templates):
            if next_template.template_id in self.template_dict:
                raise ValueError(
                    f"Template id {next_template.template_id} is duplicated."
                )
            self.template_dict[next_template.template_id] = next_template

    @abstractmethod
    def run(
        self,
        session: Optional["Session"] = None,
        max_messages: Optional[int] = None,
        debug_mode: bool = False,
    ) -> "Session":
        """All runners should implement this method. This method should run the templates and return the final session."""
        raise NotImplementedError("run method is not implemented")

    def search_template(self, template_like: str) -> "Template":
        """Search template by template id. If template id is not found, raise ValueError."""
        if template_like == EndTemplate.template_id:
            return EndTemplate()
        if template_like not in self.template_dict:
            raise ValueError(f"Template id {template_like} is not found.")
        return self.template_dict[template_like]


def cutify_role(role: MessageRoleType) -> str:
    """Cutify role name based on OpenAI's naming convention."""
    if role == "system":
        return "📝 system"
    if role == "user":
        return "👤 user"
    if role == "assistant":
        return "🤖 assistant"
    if role == "function":
        return "🛠️ function"
    if role == "tool_result":
        return "📊 tool_result"
    return role


def pretty_print_metadata(metadata: Dict[str, Any]) -> str:
    TRUNCATION_THRESHOLD = 20  # Only truncate strings longer than this
    TRUNC_HEAD = 5  # Number of characters to keep at the start
    TRUNC_TAIL = 5  # Number of characters to keep at the end

    def format_value(value: Any) -> str:
        if isinstance(value, str):
            # Truncate string if it is too long.
            if len(value) > TRUNCATION_THRESHOLD:
                value = value[:TRUNC_HEAD] + "..." + value[-TRUNC_TAIL:]
            # Use json.dumps to produce a properly escaped string (with quotes)
            return json.dumps(value)
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, dict):
            # Nested dictionaries use curly braces.
            return format_dict(value)
        elif isinstance(value, list):
            # Format each element recursively for lists.
            return "[" + ", ".join(format_value(item) for item in value) + "]"
        elif value is None:
            return "None"
        else:
            # For custom objects, simply show the class name followed by ()
            return f"{value.__class__.__name__}()"

    def format_dict(d: dict) -> str:
        # Top-level dictionary uses parentheses, nested ones use curly braces.
        open_delim = "{"
        close_delim = "}"
        formatted_items = []
        for key, val in d.items():
            # We assume keys are strings and print them as is.
            formatted_items.append(f'"{key}": {format_value(val)}')
        return open_delim + ", ".join(formatted_items) + close_delim

    return format_dict(metadata)


class CommandLineRunner(Runner):
    def run(
        self,
        session: Optional["Session"] = None,
        max_messages: Optional[int] = 100,
        debug_mode: bool = False,
    ) -> "Session":
        """Command line runner. This runner is for debugging purpose. It prints out the messages to the console.

        Args:
            session (Optional[Session], optional): If set, use the session given. Otherwise, create a new session. Defaults to None.
            max_messages (Optional[int], optional): Maximum number of messages to yield. If number of messages exceeds this number, the conversation is forced to stop. Defaults to 100.
            debug_mode (bool, optional): If set, print out debug messages. Defaults to False.

        Returns:
            Session: Final session of the conversation.
        """

        # set / update session
        if session is None:
            session = Session(
                runner=self,
                debug_mode=debug_mode,
            )
        else:
            if session.runner is None or session.runner != self:
                session.runner = self
            session.debug_mode = debug_mode or session.debug_mode

        n_messages = 0
        template = self.template
        gen = template.render(session)
        print("===== Start =====")
        while 1:
            # render template until exhausted
            try:
                obj = next(gen)
            except ReachedEndTemplateException:
                self.warning(
                    "End template %s is reached. Flow is forced to stop.",
                    EndTemplate.template_id,
                )
                break
            except StopIteration as e:
                # For generator, type support for return value is not so good.
                session = cast(Session, e.value)
                break
            if isinstance(obj, Message):
                message = obj
                print("From: " + cutify_role(message.role))
                if message.content:
                    print("message: ", message.content)
                if message.tool_use:
                    print("tool_use: ", message.tool_use)
                if message.metadata:
                    print("metadata: ", pretty_print_metadata(message.metadata))
                    n_messages += 1
            elif isinstance(obj, Event):
                event = obj
                if isinstance(event, UserInteractionEvent):
                    instruction = event.instruction or "Input: "
                    default = event.default or None
                    content = self.user_interface.ask(session, instruction, default)
                    session.messages.append(
                        Message(
                            role="user",
                            content=content,
                            metadata=session.metadata,
                        )
                    )
                else:
                    self.warning(f"Unknown event type: {type(event)}")
                    raise ValueError(f"Unknown event type: {type(event)}")
            else:
                self.warning(f"Unknown object type: {type(obj)}")

            if max_messages and n_messages >= max_messages:
                self.warning(
                    "Max messages %s is reached. Flow is forced to stop.", max_messages
                )
                break
            print("=================")
        print("====== End ======")
        return session
