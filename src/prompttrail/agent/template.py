import json
import logging
from abc import abstractmethod
from pprint import pformat
from typing import Generator, List, Optional, Sequence, TypeAlias
from uuid import uuid4

import jinja2
from pydantic import BaseModel

from prompttrail.agent import State
from prompttrail.agent.core import StatefulMessage
from prompttrail.agent.hook import BooleanHook, JumpHook, TransformHook
from prompttrail.agent.tool import Tool, check_arguments
from prompttrail.const import END_TEMPLATE_ID, OPENAI_SYSTEM_ROLE, RESERVED_TEMPLATE_IDS
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


class Stack(BaseModel):
    template_id: TemplateId


class Template(object):
    """A template represents a template that create some messages (usually one) when rendered and include some logic to control flow."""

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        self.template_id: str = template_id if template_id is not None else self._name()
        check_template_id(self.template_id)
        self.before_transform = before_transform
        self.after_transform = after_transform
        self.before_control = before_control
        self.after_control = after_control

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(__name__ + "." + str(self.template_id))

    def render(self, state: "State") -> Generator[Message, None, State]:
        state.stack.append(self.create_stack(state))  # type: ignore
        try:
            res = yield from self._render(state)
        except Exception as e:
            self.get_logger().error(f"RenderingTemplateError@{self.template_id}")
            raise e
        finally:
            state.stack.pop()  # type: ignore
        return res

    @abstractmethod
    def create_stack(self, state: "State") -> "Stack":
        return Stack(template_id=self.template_id)

    @abstractmethod
    def _render(self, state: "State") -> Generator[Message, None, State]:
        raise NotImplementedError("render method is not implemented")

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self

    def __str__(self) -> str:
        return f"Template(id={self.template_id})"

    def _name(self) -> str:
        return "Unnamed_" + self.__class__.__name__ + "_" + str(uuid4())


class MessageTemplate(Template):
    def __init__(
        self,
        content: str,
        role: str,
        template_id: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role

    def _render(self, state: "State") -> Generator[Message, None, State]:
        state.jump_to_id = None
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
        # before_jump
        for before_jump_hook in self.before_control:
            next_template_id = before_jump_hook.hook(state)
            if next_template_id is not None:
                state.jump_to_id = next_template_id
        # no jump, then render
        if not state.jump_to_id:
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
                if next_template_id is not None and state.jump_to_id is None:
                    state.jump_to_id = next_template_id
                else:
                    logger.warning(
                        f"Jump is already specified by someone. Ignoring {after_jump_hook}."
                    )
            state.session_history.messages.append(message)  # type: ignore
            yield message
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

    def create_stack(self, state: "State") -> "Stack":
        return Stack(template_id=self.template_id)


class ControlTemplate(Template):
    # ControlTemplate must handle its child templates.

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )

    @abstractmethod
    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        raise NotImplementedError("walk method is not implemented")

    @abstractmethod
    def create_stack(self, state: "State") -> "Stack":
        raise NotImplementedError(
            "Derived class of ControlTemplate must implement its own create_stack method"
        )


class LoopTemplateStack(Stack):
    cumulative_index: int
    n_children: int

    def get_loop_idx(self) -> int:
        return self.cumulative_index // self.n_children

    def get_idx(self) -> int:
        return self.cumulative_index % self.n_children

    def next(self) -> None:
        self.cumulative_index += 1


class LoopTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        exit_condition: BooleanHook,
        template_id: Optional[TemplateId] = None,
        exit_loop_count: Optional[int] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.templates = templates
        self.exit_condition = exit_condition
        self.exit_loop_count = exit_loop_count

    def _render(self, state: "State") -> Generator[Message, None, State]:
        stack = state.stack[-1]
        if not isinstance(stack, LoopTemplateStack):
            raise RuntimeError("LoopTemplateStack is not the last stack")
        while True:
            idx = stack.get_idx()
            template = self.templates[idx]
            gen = template.render(state)
            state = yield from gen
            if self.exit_condition.hook(state):
                break
            stack.next()
            if (
                self.exit_loop_count is not None
                and stack.get_loop_idx() >= self.exit_loop_count
            ):
                logger.warning(
                    msg=f"Loop count is over {self.exit_loop_count}. Breaking the loop."
                )
                break
        return state

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self.template_id in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)

    def create_stack(self, state: "State") -> "Stack":
        return LoopTemplateStack(
            template_id=self.template_id,
            n_children=len(self.templates),
            cumulative_index=0,
        )


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
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.true_template = true_template
        self.false_template = false_template
        self.condition = condition

    def _render(self, state: "State") -> Generator[Message, None, State]:
        if self.condition.hook(state):
            state = yield from self.true_template.render(state)
        else:
            state = yield from self.false_template.render(state)
        return state

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self.template_id in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self
        yield from self.true_template.walk(visited_templates)
        yield from self.false_template.walk(visited_templates)

    def create_stack(self, state: "State") -> "Stack":
        return Stack(template_id=self.template_id)


class LinearTemplateStack(Stack):
    idx: int


class LinearTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        template_id: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.templates = templates

    def _render(self, state: "State") -> Generator[Message, None, State]:
        stack = state.stack[-1]
        if not isinstance(stack, LinearTemplateStack):
            raise RuntimeError("LinearTemplateStack is not the last stack")

        while 1:
            state = yield from self.templates[stack.idx].render(state)
            stack.idx += 1
            # when the last template is rendered, break the loop
            if stack.idx >= len(self.templates):
                break
        return state

    def walk(
        self, visited_templates: Sequence["Template"] = []
    ) -> Generator["Template", None, None]:
        if self.template_id in visited_templates:
            return
        visited_templates.append(self)  # type: ignore
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)

    def create_stack(self, state: "State") -> LinearTemplateStack:
        return LinearTemplateStack(template_id=self.template_id, idx=0)


class GenerateTemplate(MessageTemplate):
    def __init__(
        self,
        role: str,
        template_id: Optional[TemplateId] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            content="",  # TODO: This should be None. Or not use MessageTemplate?
            role=role,
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )

    def _render(self, state: "State") -> Generator[Message, None, State]:
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
            raise ValueError("runner is not set")
        rendered_content = state.runner.model.send(
            state.runner.parameters, state.session_history
        ).content
        message = StatefulMessage(
            content=rendered_content,
            sender=self.role,
            template_id=self.template_id,
        )
        state.session_history.messages.append(message)  # type: ignore
        yield message
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
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            content="",  # TODO: This should be None. Or not use MessageTemplate?
            role=role,
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.role = role
        self.description = description
        self.default = default

    def _render(self, state: "State") -> Generator[Message, None, State]:
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
        yield message
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
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=role,
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
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            role=role,
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
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
        before_control: List[JumpHook] = [],
        after_control: List[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            role=role,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )
        self.functions = {func.name: func for func in functions}

    def _render(self, state: "State") -> Generator[Message, None, State]:
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
        runner = state.runner
        if runner is None:
            raise ValueError(
                "Runner must be given to use GenerateTemplate. Do you use Runner correctly? Runner must be passed via State."
            )
        if not isinstance(runner.model, OpenAIChatCompletionModel):
            raise ValueError(
                "Function calling can only be used with OpenAIChatCompletionModel."
            )

        temporary_parameters = runner.parameters.model_copy()
        temporary_parameters.functions = self.functions

        # 1st message: pass functions and let the model use it
        rendered_message = runner.model.send(
            temporary_parameters, state.session_history
        )
        message = StatefulMessage(
            content=rendered_message.content,
            sender=self.role,
            template_id=self.template_id,
            data=rendered_message.data,
        )
        state.session_history.messages.append(message)  # type: ignore
        yield message

        if rendered_message.data.get("function_call"):
            # 2nd message: call function if the model asks for it
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
            yield function_message
            second_response = runner.model.send(
                runner.parameters, state.session_history
            )
            message = StatefulMessage(
                content=second_response.content,
                sender=second_response.sender,
                template_id=self.template_id,
            )
            state.session_history.messages.append(message)  # type: ignore
            yield message

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

    def _render(self, state: "State") -> Generator[Message, None, State]:
        raise ValueError("EndTemplate should not be rendered. Something is wrong.")

    def create_stack(self, state: State) -> Stack:
        return super().create_stack(state)


class OpenAISystemTemplate(MessageTemplate):
    def __init__(
        self,
        content: str,
        template_id: Optional[TemplateId] = None,
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=OPENAI_SYSTEM_ROLE,
        )
