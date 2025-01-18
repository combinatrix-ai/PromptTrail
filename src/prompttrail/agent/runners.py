import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Set, cast

from prompttrail.agent.templates import EndTemplate, Template
from prompttrail.agent.user_interaction import UserInteractionProvider
from prompttrail.core import Model, Parameters, Session
from prompttrail.core.const import JumpException, ReachedEndTemplateException

# Session is already imported from prompttrail.core

logger = logging.getLogger(__name__)


class Runner(metaclass=ABCMeta):
    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        template: "Template",
        user_interaction_provider: UserInteractionProvider,
    ):
        self.models = model
        self.parameters = parameters
        self.user_interaction_provider = user_interaction_provider
        self.template = template
        self.template_dict: Dict[str, Template] = {}
        visited_templates: Set[Template] = set()
        for next_template in template.walk(visited_templates):
            if next_template.template_id in self.template_dict:
                raise ValueError(
                    f"Template id {next_template.template_id} is duplicated."
                )
            self.template_dict[next_template.template_id] = next_template  # type: ignore
        """Abstract class for runner. Runner is a class to run the templates. It is responsible for rendering templates and handling user interactions."""

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


def cutify_role(role: Optional[str]):
    """Cutify role name based on OpenAI's naming convention."""
    if role == "system":
        return "📝 system"
    if role == "user":
        return "👤 user"
    if role == "assistant":
        return "🤖 assistant"
    if role == "function":
        return "🧮 function"
    if role is None:
        return "❓ None"
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
                logger.warning(
                    f"Given session has different runner {session.runner} from the runner {self}. Overriding the session.",
                )
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
                logger.warning(
                    f"End template {EndTemplate.template_id} is reached. Flow is forced to stop."
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
                if not message.content and not any(
                    key != "template_id" for key in message.metadata
                ):
                    print("Empty message!")
                n_messages += 1
            if max_messages and n_messages >= max_messages:
                logger.warning(
                    f"Max messages {max_messages} is reached. Flow is forced to stop."
                )
                break
            print("=================")
        print("====== End ======")
        return session_
