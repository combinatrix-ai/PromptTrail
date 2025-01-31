import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Set, cast

from prompttrail.agent.templates._control import EndTemplate
from prompttrail.agent.templates._core import Template
from prompttrail.agent.user_interface import UserInterface
from prompttrail.core import MessageRoleType, Model, Session
from prompttrail.core.const import JumpException, ReachedEndTemplateException
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
        self.models = model
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
        start_template_id: Optional[str] = None,
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


class CommandLineRunner(Runner):
    def run(
        self,
        start_template_id: Optional[str] = None,
        session: Optional["Session"] = None,
        max_messages: Optional[int] = 100,
        debug_mode: bool = False,
    ) -> "Session":
        """Command line runner. This runner is for debugging purpose. It prints out the messages to the console.

        Args:
            start_template_id (Optional[str], optional): If set, start from the template id given. Otherwise, start from the first template. Defaults to None.
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

        current_template_id = (
            start_template_id if start_template_id else self.template.template_id
        )

        # not to override session for type checking
        session_ = session
        # not to reuse it
        del session

        n_messages = 0
        template = self.search_template(current_template_id)
        gen = template.render(session_)
        print("===== Start =====")
        while 1:
            # render template until exhausted
            try:
                message = next(gen)
            except ReachedEndTemplateException:
                self.warning(
                    "End template %s is reached. Flow is forced to stop.",
                    EndTemplate.template_id,
                )
                break
            except JumpException as e:
                # Jump to another template
                current_template_id = e.jump_to
                template = self.search_template(current_template_id)
                # reset stack
                assert len(session_.stack) == 0
                session_.stack = []
                gen = template.render(session_)
                continue
            except StopIteration as e:
                # For generator, type support for return value is not so good.
                session_ = cast(Session, e.value)
                break
            if message:
                print("From: " + cutify_role(message.role))
                if message.content:
                    print("message: ", message.content)
                if message.metadata and any(
                    key != "template_id" for key in message.metadata
                ):
                    # Filter out template_id from metadata
                    metadata = {
                        k: v for k, v in message.metadata.items() if k != "template_id"
                    }
                    if metadata:
                        print("metadata: ", metadata)
                if message.tool_use:
                    print("tool_use: ", message.tool_use)
                n_messages += 1
            if max_messages and n_messages >= max_messages:
                self.warning(
                    "Max messages %s is reached. Flow is forced to stop.", max_messages
                )
                break
            print("=================")
        print("====== End ======")
        return session_
