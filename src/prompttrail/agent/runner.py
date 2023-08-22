import logging
from abc import abstractmethod
from typing import Dict, Optional, Set

from prompttrail.agent import State
from prompttrail.agent.core import StatefulSession
from prompttrail.agent.template import EndTemplate, Template
from prompttrail.agent.user_interaction import UserInteractionProvider
from prompttrail.const import JumpException, ReachedEndTemplateException
from prompttrail.core import Model, Parameters

logger = logging.getLogger(__name__)


class Runner(object):
    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        template: "Template",
        user_interaction_provider: UserInteractionProvider,
    ):
        self.model = model
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

    @abstractmethod
    def run(
        self,
        start_template_id: Optional[str] = None,
        state: Optional[State] = None,
        max_messages: Optional[int] = None,
        debug_mode: bool = False,
    ) -> State:
        raise NotImplementedError("run method is not implemented")

    def search_template(self, template_like: str) -> "Template":
        if template_like == EndTemplate.template_id:
            return EndTemplate()
        if template_like not in self.template_dict:
            raise ValueError(f"Template id {template_like} is not found.")
        return self.template_dict[template_like]


def cutify_sender(sender: Optional[str]):
    if sender == "system":
        return "📝 system"
    if sender == "user":
        return "👤 user"
    if sender == "assistant":
        return "🤖 assistant"
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
                logger.log(
                    logging.INFO,
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
                print("==================")
                print("From: " + cutify_sender(message.sender))
                print(message.content)
                print("==================")
                n_messages += 1
            if max_messages and n_messages >= max_messages:
                logger.warning(
                    f"Max messages {max_messages} is reached. Flow is forced to stop."
                )
                break
        return state_
