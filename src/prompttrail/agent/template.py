import json
import logging
from abc import abstractmethod
from pprint import pformat
from typing import Generator, List, Optional, Sequence, TypeAlias
from uuid import uuid4

import jinja2

from prompttrail.agent import State
from prompttrail.agent.core import StatefulMessage
from prompttrail.agent.hook import BooleanHook, IfJumpHook, JumpHook, TransformHook
from prompttrail.agent.tool import Tool, check_arguments
from prompttrail.const import (
    CONTROL_TEMPLATE_ROLE,
    END_TEMPLATE_ID,
    RESERVED_TEMPLATE_IDS,
)
from prompttrail.core import Message
from prompttrail.provider.openai import OpenAIChatCompletionModel, OpenAIrole

logger = logging.getLogger(__name__)

TemplateId: TypeAlias = str


# TODO: How can I force the user to use check_template_id?
def check_template_id(template_id: TemplateId) -> None:
    if template_id in RESERVED_TEMPLATE_IDS:
        raise ValueError(
            f"Template id {template_id} is reserved. Please use another template id."
        )


class Template(object):
    """A template represents a template that create some messages (usually one) when rendered and include some logic to control flow."""

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[None] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id: str = (
            template_id
            if template_id is not None
            else "Unnamed_Template_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.next_template_default = next_template_default
        """ This is the default next template selected when no jump is specified. This is usually specified by MetaTemplate. """
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_control = before_control
        self.after_control = after_control

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(__name__ + "." + str(self.template_id))

    def render(self, state: "State") -> "State":
        state.current_template_id = (
            self.template_id
        )  # TODO: This should be stack maybe? We can force push/pop? => Maybe not because in this impl. we don't render recursively.
        # TODO: This check may be redundant?
        jump: str | Template | None = state.get_jump()
        if jump is not None:
            jump_template_id = jump.template_id if isinstance(jump, Template) else jump
            if jump_template_id != self.template_id:
                logger = self.get_logger()
                logger.warning(
                    f"State is set to jump to {jump_template_id} which is not the current template. Something is wrong."
                )
            state.jump_to_id = None
        return self._render(state)

    @abstractmethod
    def _render(self, state: "State") -> "State":
        raise NotImplementedError("render method is not implemented")

    def list_child_templates(self) -> List["Template"]:
        return [self]

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self

    def __str__(self) -> str:
        return f"Template(id={self.template_id})"


class MessageTemplate(Template):
    def __init__(
        self,
        content: str,
        role: str,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_MessageTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.next_template_default = next_template_default
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_control = before_control
        self.after_control = after_control

    def _render(self, state: "State") -> "State":
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
        # before_jump
        for before_jump_hook in self.before_control:
            next_template_id = before_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        # render
        rendered_content = self.jinja_template.render(**state.data)
        message = StatefulMessage(
            content=rendered_content, sender=self.role, template_id=self.template_id
        )
        state.session_history.messages.append(message)  # type: ignore
        # after_transform
        for after_transform_hook in self.after_transform:
            state = after_transform_hook.hook(state)
        # after_jump
        for after_jump_hook in self.after_control:
            next_template_id = after_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        return state

    def __str__(self) -> str:
        if "\n" in self.content:
            content_part = 'content="""\n' + self.content + '\n"""'
        else:
            content_part = 'content="' + self.content + '"'

        if self.before_transform:
            before_transform_part = ", before_transform=" + pformat(
                self.before_transform
            )
        else:
            before_transform_part = ""
        if self.after_transform:
            after_transform_part = ", after_transform=" + pformat(self.after_transform)
        else:
            after_transform_part = ""
        if self.before_control:
            before_jump_part = ", before_jump=" + pformat(self.before_control)
        else:
            before_jump_part = ""
        if self.after_control:
            after_jump_part = ", after_jump=" + pformat(self.after_control)
        else:
            after_jump_part = ""

        return f"MessageTemplate(id={self.template_id}, {content_part}{before_transform_part}{after_transform_part}{before_jump_part}{after_jump_part})"


class ControlTemplate(MessageTemplate):
    # ControlTemplate must handle its child templates.

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_ControlTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.next_template_default = next_template_default
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_jump = before_control
        self.after_jump = after_control
        self.role = CONTROL_TEMPLATE_ROLE

    @abstractmethod
    def list_child_templates(self) -> List[Template]:
        return [self]

    @abstractmethod
    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        raise NotImplementedError("walk method is not implemented")


class LoopTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        exit_condition: BooleanHook,
        jump_to: Optional[TemplateId] = None,
        template_id: Optional[TemplateId] = None,
        exit_loop_count: Optional[int] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_LoopTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.templates = templates
        self.exit_condition = exit_condition
        self.exit_loop_count = exit_loop_count
        # Of course, LoopTemplate use self.templates[0] as next template
        self.next_template_default = templates[0].template_id
        self.role = CONTROL_TEMPLATE_ROLE
        self.before_transform = before_transform
        self.before_control = before_control
        self.after_transform = after_transform
        self.after_control = after_control

        # set loop link
        for template, next_template in zip(self.templates, self.templates[1:]):
            template.next_template_default = next_template.template_id
        self.templates[-1].next_template_default = self.templates[0].template_id

        # set jump link
        hook = IfJumpHook(
            condition=self.exit_condition.hook,
            true_template=jump_to if jump_to is not None else EndTemplate().template_id,
        )
        for template in self.templates:
            template.after_control.append(hook)

    def _render(self, state: State) -> State:
        # render
        message = StatefulMessage(
            content="",
            sender=self.role,
            template_id=self.template_id,
        )
        state.session_history.messages.append(message)  # type: ignore
        return state

    def list_child_templates(self) -> List[Template]:
        return [self] + [template for template in self.templates]

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self.template_id in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)


class IfTemplate(ControlTemplate):
    def __init__(
        self,
        true_template: Template,
        false_template: Template,
        condition: BooleanHook,
        template_id: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_IfTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.true_template = true_template
        self.false_template = false_template
        self.condition = condition
        self.next_template_default = None
        self.role = CONTROL_TEMPLATE_ROLE
        self.before_transform = before_transform
        self.before_control = before_control
        self.after_transform = after_transform
        self.after_control = after_control

    def _render(self, state: State) -> State:
        message = StatefulMessage(
            content="",
            sender=self.role,
            template_id=self.template_id,
        )
        state.session_history.messages.append(message)  # type: ignore
        if self.condition.hook(state):
            state.jump_to_id = self.true_template.template_id
        else:
            state.jump_to_id = self.false_template.template_id
        return state

    def list_child_templates(self) -> List[Template]:
        return [self, self.true_template, self.false_template]

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self.template_id in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self
        yield from self.true_template.walk(visited_templates)
        yield from self.false_template.walk(visited_templates)


class LinearTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_LinearTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.templates = templates

        self.next_template_default = self.templates[0].template_id
        # set linear link
        for template, next_template in zip(self.templates, self.templates[1:]):
            template.next_template_default = next_template.template_id
        # set last next_template
        if next_template_default is not None:
            self.templates[-1].next_template_default = next_template_default
        self.role = CONTROL_TEMPLATE_ROLE

        self.before_transform = before_transform
        self.before_control = before_control
        self.after_transform = after_transform
        self.after_control = after_control

    def _render(self, state: State) -> State:
        # render
        message = StatefulMessage(
            content="",
            sender=self.role,
            template_id=self.template_id,
        )
        state.session_history.messages.append(message)  # type: ignore
        return state

    def list_child_templates(self) -> List[Template]:
        return [self] + [template for template in self.templates]

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self.template_id in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)


class GenerateTemplate(MessageTemplate):
    def __init__(
        self,
        role: str,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_GenerateTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.next_template_default = next_template_default
        self.role = role
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_control = before_control
        self.after_control = after_control

    def _render(self, state: "State") -> "State":
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
        # before_jump
        for before_jump_hook in self.before_control:
            next_template_id = before_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        # render
        if state.model is None:
            raise ValueError(
                "Model must be given to use GenerateTemplate. Please set model to the runner."
            )
        if state.parameters is None:
            raise ValueError(
                "Parameters must be given to use GenerateTemplate. Please set parameters to the runner."
            )
        rendered_content = state.model.send(
            state.parameters, state.session_history
        ).content
        message = StatefulMessage(
            content=rendered_content,
            sender=self.role,
            template_id=self.template_id,
        )
        state.session_history.messages.append(message)  # type: ignore
        # after_transform
        for after_transform_hook in self.after_transform:
            state = after_transform_hook.hook(state)
        # after_jump
        for after_jump_hook in self.after_control:
            next_template_id = after_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        return state


class UserInputTextTemplate(MessageTemplate):
    def __init__(
        self,
        role: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id = (
            template_id
            if template_id is not None
            else "Unnamed_UserInputTemplate_" + str(uuid4())
        )
        check_template_id(self.template_id)
        self.next_template_default = next_template_default
        self.role = role
        self.description = description
        self.default = default
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_control = before_control
        self.after_control = after_control

    def _render(self, state: "State") -> "State":
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
        # before_jump
        for before_jump_hook in self.before_control:
            next_template_id = before_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        # render
        if state.runner is None:
            raise ValueError(
                "Runner must be given to use UserInputTextTemplate. Do you use Runner correctly? Runner must be passed via State."
            )

        rendered_content = state.runner.user_interaction_provider.ask(
            state, self.description, self.default
        )
        message = StatefulMessage(
            content=rendered_content,
            sender=self.role,
            template_id=self.template_id,
        )
        # TODO: Sequence cannot have append method, this causd a bug already. Need to be fixed
        state.session_history.messages.append(message)  # type: ignore
        # after_transform
        for after_transform_hook in self.after_transform:
            state = after_transform_hook.hook(state)
        # after_jump
        for after_jump_hook in self.after_control:
            next_template_id = after_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        return state


class OpenAIMessageTemplate(MessageTemplate):
    def __init__(
        self,
        content: str,
        role: OpenAIrole,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=role,
            next_template_default=next_template_default,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )


class OpenAIGenerateTemplate(GenerateTemplate):
    def __init__(
        self,
        role: OpenAIrole,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            role=role,
            next_template_default=next_template_default,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )


class OpenAIGenerateWithFunctionCallingTemplate(GenerateTemplate):
    def __init__(
        self,
        role: OpenAIrole,
        functions: Sequence[Tool],
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            role=role,
            next_template_default=next_template_default,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.functions = {func.name: func for func in functions}

    def _render(self, state: "State") -> "State":
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
        # before_jump
        for before_jump_hook in self.before_control:
            next_template_id = before_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        # render
        if state.model is None:
            raise ValueError(
                "Model must be given to use GenerateTemplate. Please set model to the runner."
            )
        if state.parameters is None:
            raise ValueError(
                "Parameters must be given to use GenerateTemplate. Please set parameters to the runner."
            )
        if not isinstance(state.model, OpenAIChatCompletionModel):
            raise ValueError(
                "Function calling can only be used with OpenAIChatCompletionModel."
            )
        # TODO: Temporaly rewrite parameters for function calling. Is this good?
        old_state_parameters = state.parameters.model_copy()
        state.parameters.functions = self.functions

        # 1st message
        rendered_message = state.model.send(state.parameters, state.session_history)
        state.parameters = old_state_parameters
        message = StatefulMessage(
            content=rendered_message.content,
            sender=self.role,
            template_id=self.template_id,
            data=rendered_message.data,
        )
        state.session_history.messages.append(message)  # type: ignore

        # 2nd message
        if rendered_message.data.get("function_call"):
            if rendered_message.data["function_call"]["name"] not in self.functions:
                raise ValueError(
                    f"Function {rendered_message.data['function_call']['name']} is not defined in the template."
                )
            function = self.functions[rendered_message.data["function_call"]["name"]]
            arguments = check_arguments(
                rendered_message.data["function_call"]["arguments"],
                function.argument_types,
            )
            result = function.call(arguments, state)  # type: ignore
            # Send result
            function_message = Message(
                sender="function",
                data={"function_call": {"name": function.name}},
                content=json.dumps(result.show()),
            )
            state.session_history.messages.append(function_message)  # type: ignore
            second_response = state.model.send(state.parameters, state.session_history)
            message = StatefulMessage(
                content=second_response.content,
                sender=second_response.sender,
                template_id=self.template_id,
            )
            state.session_history.messages.append(message)  # type: ignore

        # after_transform
        for after_transform_hook in self.after_transform:
            state = after_transform_hook.hook(state=state)
        # after_jump
        for after_jump_hook in self.after_control:
            next_template_id = after_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
                return state
        return state


# EndTemplate is singleton
class EndTemplate(Template):
    template_id = END_TEMPLATE_ID
    _instance = None

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EndTemplate, cls).__new__(cls)
        return cls._instance

    def _render(self, state: State) -> State:
        raise ValueError("EndTemplate should not be rendered. Something is wrong.")
