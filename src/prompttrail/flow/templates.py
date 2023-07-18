import logging
from abc import abstractmethod
from typing import List, Optional, Sequence, TypeAlias
from uuid import uuid4

import jinja2

logger = logging.getLogger(__name__)

TemplateId: TypeAlias = str

from src.prompttrail.flow.core import FlowState, StatefulMessage
from src.prompttrail.flow.hooks import BooleanHook, JumpHook, TransformHook


class Template(object):
    @abstractmethod
    def __init__(self, template_id: Optional[None] = None):
        self.id = (
            template_id
            if template_id is not None
            else "Unnamed_Template_" + str(uuid4())
        )

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(__name__ + "." + self.id)

    def render(self, flow_state: "FlowState") -> "FlowState":
        flow_state.current_template = self  # TODO: This kind of convention should be separated into another function
        jump = flow_state.get_jump()
        if jump is not None:
            jump_template_id = jump.id if isinstance(jump, Template) else jump
            if jump_template_id != self.id:
                logger = self.get_logger()
                logger.warning(
                    f"FlowState is set to jump to {jump_template_id} which is not the current template. "
                )
            flow_state.jump = None
        # try:
        if 1 == 1:
            return self._render(flow_state)
        # except Exception as e:
        #     logger = self.get_logger()
        #     logger.error(
        #     f"An error occurred while rendering {self.id}.\n"
        #         +f"Template Class: {self.__class__}\n"
        #         +f"History: {pprint(flow_state.session_history)}\n"
        #         +f"Data: {pprint(flow_state.data)}\n"
        #         f"Error: {e}"
        #     )
        #     logger.error(e)
        #     raise RenderingError("An error occurred while rendering.") from e

    @abstractmethod
    def _render(self, flow_state: "FlowState") -> "FlowState":
        raise NotImplementedError("render method is not implemented")

    def list_all_templates(self) -> List["Template"]:
        return [self]


class MessageTemplate(Template):
    def __init__(
        self,
        content: str,
        role: str,
        template_id: Optional[TemplateId] = None,
        before_transform: Sequence[TransformHook] = [],
        after_transform: Sequence[TransformHook] = [],
        before_control: Sequence[JumpHook] = [],
        after_control: Sequence[JumpHook] = [],
    ):
        self.id = (
            template_id
            if template_id is not None
            else "Unnamed_MessageTemplate_" + str(uuid4())
        )
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_jump: List[JumpHook] = before_control
        self.after_jump: List[JumpHook] = after_control

    def _render(self, flow_state: FlowState) -> FlowState:
        # before_transform
        for hook in self.before_transform:
            flow_state = hook.hook(flow_state)
        # before_jump
        for hook in self.before_jump:
            next_template_id = hook.hook(flow_state)
            if next_template_id is not None:
                flow_state.jump = next_template_id
                return flow_state
        # renderzw
        rendered_content = self.jinja_template.render(**flow_state.data)
        message = StatefulMessage(
            content=rendered_content, sender=self.role, template_id=self.id
        )
        flow_state.session_history.messages.append(message)
        from IPython import embed

        embed()
        # after_transform
        for hook in self.after_transform:
            flow_state = hook.hook(flow_state)
        # after_jump
        for hook in self.after_jump:
            next_template_id = hook.hook(flow_state)
            if next_template_id is not None:
                flow_state.jump = next_template_id
                return flow_state
        return flow_state


class MetaTemplate(Template):
    # MetaTemplate does not take care of Hook in ordinary way.
    ...

    @abstractmethod
    def list_all_templates(self) -> List[Template]:
        return [self]


class LoopTemplate(MetaTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        exit_condition: BooleanHook,
        template_id: Optional[TemplateId] = None,
    ):
        self.id = (
            template_id
            if template_id is not None
            else "Unnamed_LoopTemplate_" + str(uuid4())
        )
        self.templates = templates
        self.exit_condition = exit_condition

    def _render(
        self, flow_state: FlowState, exit_loop_count: Optional[int] = None
    ) -> FlowState:
        flag = False
        count = 0
        while 1:
            for template in self.templates:
                flow_state = template.render(flow_state)
                flag = self.exit_condition.hook(flow_state)
                count += 1
                if exit_loop_count is not None and count >= exit_loop_count:
                    flag = True
                if flag:
                    break
            if flag:
                break
        return flow_state

    def list_all_templates(self) -> List[Template]:
        return [self] + [template for template in self.templates]


class IfTemplate(MetaTemplate):
    def __init__(
        self,
        true_template: Template,
        false_template: Template,
        condition: BooleanHook,
        template_id: Optional[TemplateId] = None,
    ):
        self.id = (
            template_id
            if template_id is not None
            else "Unnamed_IfTemplate_" + str(uuid4())
        )
        self.true_template = true_template
        self.false_template = false_template
        self.condition = condition

    def _render(self, flow_state: FlowState) -> FlowState:
        flow_state.current_template = self  # TODO: This kind of convention should be separated into another function
        if self.condition.hook(flow_state):
            return self.true_template.render(flow_state)
        else:
            return self.false_template.render(flow_state)

    def list_all_templates(self) -> List[Template]:
        return [self, self.true_template, self.false_template]


class LinearTemplate(MetaTemplate):
    def __init__(
        self, templates: Sequence[Template], template_id: Optional[TemplateId] = None
    ):
        self.id = (
            template_id
            if template_id is not None
            else "Unnamed_LinearTemplate_" + str(uuid4())
        )
        self.templates = templates

    def _render(self, flow_state: FlowState) -> FlowState:
        for template in self.templates:
            flow_state = template.render(flow_state)
            if flow_state.jump is not None:
                break
        return flow_state

    def list_all_templates(self) -> List[Template]:
        return [self] + [template for template in self.templates]
