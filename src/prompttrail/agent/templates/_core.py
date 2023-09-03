import logging
from abc import ABCMeta, abstractmethod
from pprint import pformat
from typing import Generator, List, Optional, Set
from uuid import uuid4

import jinja2
from pydantic import BaseModel

from prompttrail.agent import State, StatefulMessage
from prompttrail.agent.hooks import TransformHook
from prompttrail.core import Message
from prompttrail.core.const import (
    RESERVED_TEMPLATE_IDS,
    BreakException,
    JumpException,
    ReachedEndTemplateException,
)

logger = logging.getLogger(__name__)


# TODO: How can I force the user to use check_template_id on init?
def check_template_id(template_id: str) -> None:
    if template_id in RESERVED_TEMPLATE_IDS:
        raise ValueError(
            f"Template id {template_id} is reserved. Please use another template id."
        )


class Stack(BaseModel):
    template_id: str


class Template(object, metaclass=ABCMeta):
    """A template represents a template that create some messages (usually one) when rendered and include some logic to control flow.

    The user should inherit this class and implement `_render` method, which should yield messages and return the final state.
    Also, the user should implement `create_stack` method, which should create a stack object that store status of the template on rendering.
    """

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: Optional[List[TransformHook]] = None,
        after_transform: Optional[List[TransformHook]] = None,
    ):
        self.template_id: str = template_id if template_id is not None else self._name()
        check_template_id(self.template_id)
        self.before_transform = before_transform if before_transform is not None else []
        self.after_transform = after_transform if after_transform is not None else []

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(__name__ + "." + str(self.template_id))

    def render(self, state: "State") -> Generator[Message, None, State]:
        state.stack.append(self.create_stack(state))  # type: ignore
        try:
            for hook in self.before_transform:
                state = hook.hook(state)
            res = yield from self._render(state)
            # TODO: After transform is skipped if BreakTemplate, JumpTemplate, or EndTemplate is used. This is natural?
            for hook in self.after_transform:
                state = hook.hook(state)
        except BreakException as e:
            raise e
        except ReachedEndTemplateException as e:
            raise e
        except JumpException as e:
            raise e
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
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        if visited_templates is None:
            visited_templates = set()
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self

    def __str__(self) -> str:
        return f"Template(id={self.template_id})"

    def _name(self) -> str:
        return "Unnamed_" + self.__class__.__name__ + "_" + str(uuid4())


class MessageTemplate(Template):
    """A template that create a message when rendered.

    This is the most basic template without user interaction or calling LLM.
    Rendering is based on `content`, which is a string that will be rendered by jinja2 as a message.
    Before rendering, `before_transform` hooks are applied. After rendering, `after_transform` hooks are applied.
    """

    def __init__(
        self,
        content: str,
        role: str,
        template_id: Optional[str] = None,
        before_transform: Optional[List[TransformHook]] = None,
        after_transform: Optional[List[TransformHook]] = None,
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
        )
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role

    def _render(self, state: "State") -> Generator[Message, None, State]:
        state.jump_to_id = None
        # no jump, then render
        if not state.jump_to_id:
            # render
            rendered_content = self.jinja_template.render(**state.data)
            message = StatefulMessage(
                content=rendered_content, sender=self.role, template_id=self.template_id
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

        return f"MessageTemplate(id={self.template_id}, {content_part}{before_transform_part}{after_transform_part})"

    def create_stack(self, state: "State") -> "Stack":
        return Stack(template_id=self.template_id)


class GenerateTemplate(MessageTemplate):
    """A template that create a message by calling LLM."""

    def __init__(
        self,
        role: str,
        template_id: Optional[str] = None,
        before_transform: Optional[List[TransformHook]] = None,
        after_transform: Optional[List[TransformHook]] = None,
    ):
        super().__init__(
            content="",  # TODO: This should be None. Or not use MessageTemplate?
            role=role,
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
        )

    def _render(self, state: "State") -> Generator[Message, None, State]:
        # render
        if state.runner is None:
            raise ValueError("runner is not set")
        logger.info(
            f"Generating content with {state.runner.models.__class__.__name__}..."
        )
        rendered_content = state.runner.models.send(
            state.runner.parameters, state.session_history
        ).content
        message = StatefulMessage(
            content=rendered_content,
            sender=self.role,
            template_id=self.template_id,
        )
        state.session_history.messages.append(message)  # type: ignore
        yield message
        return state


class UserInputTextTemplate(MessageTemplate):
    def __init__(
        self,
        role: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        template_id: Optional[str] = None,
        before_transform: Optional[List[TransformHook]] = None,
        after_transform: Optional[List[TransformHook]] = None,
    ):
        super().__init__(
            content="",  # TODO: This should be None. Or not use MessageTemplate?
            role=role,
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
        )
        self.role = role
        self.description = description
        self.default = default

    def _render(self, state: "State") -> Generator[Message, None, State]:
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
        return state
