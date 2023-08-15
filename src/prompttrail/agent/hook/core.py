import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from prompttrail.agent.core import FlowState

if TYPE_CHECKING:
    from prompttrail.agent.template import TemplateId

logger = logging.getLogger(__name__)


class Hook(object):
    @abstractmethod
    def hook(self, flow_state: FlowState) -> Any:
        raise NotImplementedError("hook method is not implemented")


class TransformHook(Hook):
    def __init__(self, function: Callable[[FlowState], FlowState]):
        self.function = function

    def hook(self, flow_state: FlowState) -> FlowState:
        return self.function(flow_state)


class BooleanHook(Hook):
    def __init__(self, condition: Callable[[FlowState], bool]):
        self.condition = condition

    def hook(self, flow_state: FlowState) -> bool:
        return self.condition(flow_state)


class JumpHook(Hook):
    def __init__(self, function: Callable[[FlowState], Optional["TemplateId"]]):
        self.function = function

    def hook(self, flow_state: FlowState) -> Optional["TemplateId"]:
        raise NotImplementedError("hook method is not implemented")


class IfJumpHook(JumpHook):
    def __init__(
        self,
        condition: Callable[[FlowState], bool],
        true_template: "TemplateId",
        false_template: Optional["TemplateId"] = None,
    ):
        self.condition = condition
        self.true_template = true_template
        self.false_template = false_template

    def hook(self, flow_state: FlowState) -> Optional["TemplateId"]:
        if self.condition(flow_state):
            flow_state.set_jump(self.true_template)
            return self.true_template
        else:
            if self.false_template is None:
                return None
            flow_state.set_jump(self.false_template)
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

    def hook(self, flow_state: FlowState) -> FlowState:
        # show user a prompt on console
        raw = input(self.description).strip()
        if raw == "" and self.default is not None:
            raw = self.default
        flow_state.data[self.key] = raw
        return flow_state


class GenerateChatHook(TransformHook):
    def __init__(
        self,
        key: str,
    ):
        self.key = key

    def hook(self, flow_state: FlowState) -> FlowState:
        if flow_state.parameters is None:
            raise ValueError(
                "Parameters must be given to use GenerateChatHook. Please set parameters to the runner."
            )
        if flow_state.model is None:
            raise ValueError(
                "Model must be given to use GenerateChatHook. Please set model to the runner."
            )
        message = flow_state.model.send(
            flow_state.parameters, flow_state.session_history
        )
        flow_state.data[self.key] = message.content
        return flow_state


class CountUpHook(TransformHook):
    def __init__(self):
        pass

    def hook(self, flow_state: FlowState) -> FlowState:
        template = flow_state.get_current_template()
        if template.template_id not in flow_state.data:
            flow_state.data[template.template_id] = 0
        else:
            flow_state.data[template.template_id] += 1
        return flow_state
