import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from prompttrail.agent.core import State

if TYPE_CHECKING:
    from prompttrail.agent.template import TemplateId

logger = logging.getLogger(__name__)


class Hook(object):
    @abstractmethod
    def hook(self, state: State) -> Any:
        raise NotImplementedError("hook method is not implemented")


class TransformHook(Hook):
    def __init__(self, function: Callable[[State], State]):
        self.function = function

    def hook(self, state: State) -> State:
        return self.function(state)


class BooleanHook(Hook):
    def __init__(self, condition: Callable[[State], bool]):
        self.condition = condition

    def hook(self, state: State) -> bool:
        return self.condition(state)


class JumpHook(Hook):
    def __init__(self, function: Callable[[State], Optional["TemplateId"]]):
        self.function = function

    def hook(self, state: State) -> Optional["TemplateId"]:
        raise NotImplementedError("hook method is not implemented")


class IfJumpHook(JumpHook):
    def __init__(
        self,
        condition: Callable[[State], bool],
        true_template: "TemplateId",
        false_template: Optional["TemplateId"] = None,
    ):
        self.condition = condition
        self.true_template = true_template
        self.false_template = false_template

    def hook(self, state: State) -> Optional["TemplateId"]:
        if self.condition(state):
            state.set_jump(self.true_template)
            return self.true_template
        else:
            if self.false_template is None:
                return None
            state.set_jump(self.false_template)
            return self.false_template


class AskUserHook(TransformHook):
    def __init__(
        self,
        key: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ):
        self.key = key
        self.description = description
        self.default = default

    def hook(self, state: State) -> State:
        # show user a prompt on console
        raw = input(self.description).strip()
        if raw == "" and self.default is not None:
            raw = self.default
        state.data[self.key] = raw
        return state


class GenerateChatHook(TransformHook):
    def __init__(
        self,
        key: str,
    ):
        self.key = key

    def hook(self, state: State) -> State:
        if state.parameters is None:
            raise ValueError(
                "Parameters must be given to use GenerateChatHook. Please set parameters to the runner."
            )
        if state.model is None:
            raise ValueError(
                "Model must be given to use GenerateChatHook. Please set model to the runner."
            )
        message = state.model.send(
            state.parameters, state.session_history
        )
        state.data[self.key] = message.content
        return state


class CountUpHook(TransformHook):
    def __init__(self):
        pass

    def hook(self, state: State) -> State:
        template = state.get_current_template()
        if template.template_id not in state.data:
            state.data[template.template_id] = 0
        else:
            state.data[template.template_id] += 1
        return state
