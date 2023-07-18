import logging
from abc import abstractmethod
from typing import Any, Callable, Optional, TypeAlias

logger = logging.getLogger(__name__)

TemplateId: TypeAlias = str

from src.prompttrail.flow.core import FlowState


class FeatherChainHook(object):
    @abstractmethod
    def hook(self, flow_state: FlowState) -> Any:
        raise NotImplementedError("hook method is not implemented")


class TransformHook(FeatherChainHook):
    def __init__(self, function: Callable[[FlowState], FlowState]):
        self.function = function

    def hook(self, flow_state: FlowState) -> FlowState:
        return self.function(flow_state)


class BooleanHook(FeatherChainHook):
    def __init__(self, condition: Callable[[FlowState], bool]):
        self.condition = condition

    def hook(self, flow_state: FlowState) -> bool:
        return self.condition(flow_state)


class JumpHook(FeatherChainHook):
    def __init__(self, function: Callable[[FlowState], Optional[TemplateId]]):
        self.function = function

    def hook(self, flow_state: FlowState) -> Optional[TemplateId]:
        raise NotImplementedError("hook method is not implemented")


class IfJumpHook(JumpHook):
    def __init__(
        self,
        condition: Callable[[FlowState], bool],
        true_template: Optional[TemplateId],
        false_template: Optional[TemplateId],
    ):
        self.condition = condition
        self.true_template = true_template
        self.false_template = false_template

    def hook(self, flow_state: FlowState) -> Optional[TemplateId]:
        if self.condition(flow_state):
            return flow_state.set_jump(self.true_template)
        else:
            return flow_state.set_jump(self.false_template)


# TODO: Ask and Generate should be treated differently from other hooks
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
        raw = input(self.description)
        if raw == "":
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
        message = flow_state.model.send(
            flow_state.parameters, flow_state.session_history
        )
        flow_state.data[self.key] = message.content
        return flow_state
