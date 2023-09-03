import logging
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Set

from prompttrail.agent import State, StatefulSession
from prompttrail.agent.templates import EndTemplate, Template
from prompttrail.agent.user_interaction import UserInteractionProvider
from prompttrail.core import Model, Parameters
from prompttrail.core.const import JumpException, ReachedEndTemplateException

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
        state: Optional[State] = None,
        max_messages: Optional[int] = None,
        debug_mode: bool = False,
    ) -> State:
        """All runners should implement this method. This method should run the templates and return the final state."""
        raise NotImplementedError("run method is not implemented")

    def search_template(self, template_like: str) -> "Template":
        """Search template by template id. If template id is not found, raise ValueError."""
        if template_like == EndTemplate.template_id:
            return EndTemplate()
        if template_like not in self.template_dict:
            raise ValueError(f"Template id {template_like} is not found.")
        return self.template_dict[template_like]


def cutify_sender(sender: Optional[str]):
    """Cutify sender name based on OpenAI's naming convention."""
    if sender == "system":
        return "📝 system"
    if sender == "user":
        return "👤 user"
    if sender == "assistant":
        return "🤖 assistant"
    if sender == "function":
        return "🧮 function"
    if sender is None:
        return "❓ None"
    return sender


class CommandLineRunner(Runner):
    def run(
        self,
        start_template_id: Optional[str] = None,
        state: Optional[State] = None,
        max_messages: Optional[int] = 100,
        debug_mode: bool = False,
    ) -> State:
        """Command line runner. This runner is for debugging purpose. It prints out the messages to the console.

        Args:
            start_template_id (Optional[str], optional): If set, start from the template id given. Otherwise, start from the first template. Defaults to None.
            state (Optional[State], optional): If set, use the state given. Otherwise, create a new state. Defaults to None.
            max_messages (Optional[int], optional): Maximum number of messages to yield. If number of messages exceeds this number, the conversation is forced to stop. Defaults to 100.
            debug_mode (bool, optional): If set, print out debug messages. Defaults to False.

        Returns:
            State: Final state of the conversation.
        """

        # Debug Mode

        # set / update state
        if state is None:
            state = State(
                runner=self,
                data={},
                session_history=StatefulSession(),
                debug_mode=debug_mode,
            )
        else:
            if state.runner is None or state.runner != self:
                logger.warning(
                    f"Given flow state has different runner {state.runner} from the runner {self}. Overriding the flow state.",
                )
                state.runner = self
            state.debug_mode = debug_mode or state.debug_mode

        current_template_id = (
            start_template_id if start_template_id else self.template.template_id
        )

        # not to override state for type checking
        state_ = state
        # not to reuse it
        del state

        n_messages = 0
        template = self.search_template(current_template_id)
        gen = template.render(state_)
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
                assert len(state_.stack) == 0  # type: ignore
                state_.stack = []
                gen = template.render(state_)
                continue
            except StopIteration as e:
                # For generator, type support for return value is not so good.
                state_ = e.value
                break
            if message:
                print("From: " + cutify_sender(message.sender))
                if message.content:
                    print("message: ", message.content)
                elif message.data:
                    print("data: ", end=" ")
                    print(message.data)
                else:
                    print("Empty message!")
                n_messages += 1
            if max_messages and n_messages >= max_messages:
                logger.warning(
                    f"Max messages {max_messages} is reached. Flow is forced to stop."
                )
                break
            print("=================")
        print("====== End ======")
        return state_
