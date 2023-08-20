import logging
from abc import abstractmethod
from pprint import pformat
from typing import Generator, List, Optional, Sequence
from uuid import uuid4

import jinja2
from pydantic import BaseModel

from prompttrail.agent import State
from prompttrail.agent.core import StatefulMessage
from prompttrail.agent.hook import JumpHook, TransformHook
from prompttrail.const import RESERVED_TEMPLATE_IDS
from prompttrail.core import Message

logger = logging.getLogger(__name__)


# TODO: How can I force the user to use check_template_id on init?
def check_template_id(template_id: str) -> None:
    if template_id in RESERVED_TEMPLATE_IDS:
        raise ValueError(
            f"Template id {template_id} is reserved. Please use another template id."
        )


class Stack(BaseModel):
    template_id: str


class Template(object):
    """A template represents a template that create some messages (usually one) when rendered and include some logic to control flow."""

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        self.template_id: str = template_id if template_id is not None else self._name()
        check_template_id(self.template_id)
        self.before_transform = before_transform
        self.after_transform = after_transform

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
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role

    def _render(self, state: "State") -> Generator[Message, None, State]:
        state.jump_to_id = None
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
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

        return f"MessageTemplate(id={self.template_id}, {content_part}{before_transform_part}{after_transform_part})"

    def create_stack(self, state: "State") -> "Stack":
        return Stack(template_id=self.template_id)


class GenerateTemplate(MessageTemplate):
    def __init__(
        self,
        role: str,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            content="",  # TODO: This should be None. Or not use MessageTemplate?
            role=role,
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )

    def _render(self, state: "State") -> Generator[Message, None, State]:
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
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
        return state


class UserInputTextTemplate(MessageTemplate):
    def __init__(
        self,
        role: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        template_id: Optional[str] = None,
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
        )
        self.role = role
        self.description = description
        self.default = default

    def _render(self, state: "State") -> Generator[Message, None, State]:
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
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
        return state
