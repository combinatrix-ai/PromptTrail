import logging
from abc import ABCMeta, abstractmethod
from pprint import pformat
from typing import Generator, List, Optional, Set, Union
from uuid import uuid4

import jinja2
from pydantic import BaseModel

from prompttrail.agent.hooks import TransformHook
from prompttrail.core import Message, MessageRoleType, Session
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


class Template(metaclass=ABCMeta):
    """A template represents a template that create some messages (usually one) when rendered and include some logic to control flow.

    The user should inherit this class and implement `_render` method, which should yield messages and return the final state.
    Also, the user should implement `create_stack` method, which should create a stack object that store status of the template on rendering.
    """

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
        after_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
    ):
        self.template_id: str = (
            template_id if template_id is not None else self._generate_name()
        )
        check_template_id(self.template_id)
        self.before_transform = self._hooks_to_list(before_transform)
        self.after_transform = self._hooks_to_list(after_transform)

    def _hooks_to_list(
        self, hooks: Optional[Union[List[TransformHook], TransformHook]]
    ) -> List[TransformHook]:
        if hooks is None:
            return []
        elif isinstance(hooks, list):
            return hooks
        else:
            return [hooks]

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(__name__ + "." + str(self.template_id))

    def render(self, session: "Session") -> Generator[Message, None, "Session"]:
        logging.debug(f"Rendering {self.template_id}")
        session.push_stack(self.create_stack(session))
        try:
            for hook in self.before_transform:
                session = hook.hook(session)
            res = yield from self._render(session)
            # TODO: After transform is skipped if BreakTemplate, JumpTemplate, or EndTemplate is used. This is natural?
            for hook in self.after_transform:
                session = hook.hook(session)
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
            session.pop_stack()
        logging.debug(f"Rendered {self.template_id}")
        return res

    def create_stack(self, session: "Session") -> "Stack":
        """Create a stack frame for this template."""
        return Stack(template_id=self.template_id)

    @abstractmethod
    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Render the template and return a generator of messages."""
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

    @classmethod
    def _generate_name(cls) -> str:
        """Generate a unique name for an unnamed template."""
        return f"Unnamed_{cls.__name__}_{str(uuid4())}"


class MessageTemplate(Template):
    """A template that create a message when rendered.

    This is the most basic template without user interaction or calling LLM.
    Rendering is based on `content`, which is a string that will be rendered by jinja2 as a message.
    Before rendering, `before_transform` hooks are applied. After rendering, `after_transform` hooks are applied.
    """

    def __init__(
        self,
        content: str,
        role: MessageRoleType,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
        after_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
        )
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        session.set_jump(None)
        # no jump, then render
        if not session.get_jump():
            # render
            rendered_content = self.jinja_template.render(
                **session.get_latest_metadata()
            )
            # Get metadata from the last message or initial metadata
            metadata = session.get_latest_metadata().copy()
            message = Message(
                content=rendered_content,
                role=self.role,
                metadata=metadata,
            )
            session.append(message)
            yield message
        return session

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

    def create_stack(self, session: "Session") -> "Stack":
        return Stack(template_id=self.template_id)


class GenerateTemplate(MessageTemplate):
    """A template that create a message by calling LLM."""

    def __init__(
        self,
        role: MessageRoleType,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
        after_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
    ):
        super().__init__(
            content="",  # TODO: This should be None. Or not use MessageTemplate?
            role=role,
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
        )

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        # render
        if session.runner is None:
            raise ValueError("runner is not set")
        logger.info(
            f"Generating content with {session.runner.models.__class__.__name__}..."
        )
        response = session.runner.models.send(session.runner.parameters, session)
        # Get metadata from the response
        metadata = response.metadata if response.metadata else {}
        message = Message(
            content=response.content,
            role=self.role,
            metadata=metadata,
        )
        session.append(message)
        yield message
        return session


class SystemTemplate(MessageTemplate):
    """A template that creates a system message when rendered.
    This is a convenience class that sets the role to "system" automatically."""

    def __init__(
        self,
        content: str,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
        after_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
    ):
        super().__init__(
            content=content,
            role="system",
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )


class UserTemplate(MessageTemplate):
    """A template that creates a user message when rendered.
    This template can operate in two modes:
    1. Static mode (when content is provided):
       Creates a message with the given content.
    2. Interactive mode (when content is None):
       Prompts the user for input during runtime.

    Examples:
        # Static mode
        user_static = UserTemplate("Hello!")

        # Interactive mode
        user_interactive = UserTemplate(
            description="Please enter your message",
            default="Hello!"
        )
    """

    def __init__(
        self,
        content: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[str] = None,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
        after_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
    ):
        super().__init__(
            content=content or "",
            role="user",
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.is_interactive = content is None
        self.description = description
        self.default = default

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        if self.is_interactive:
            # Interactive mode
            if session.runner is None:
                raise ValueError(
                    "Runner must be given to use interactive mode. Do you use Runner correctly?"
                )
            rendered_content = session.runner.user_interaction_provider.ask(
                session, self.description, self.default
            )
        else:
            # Static mode
            rendered_content = self.jinja_template.render(
                **session.get_latest_metadata()
            )

        metadata = session.get_latest_metadata().copy()
        message = Message(
            content=rendered_content,
            role=self.role,
            metadata=metadata,
        )
        session.append(message)
        yield message
        return session


class AssistantTemplate(MessageTemplate):
    """A template that creates an assistant message.
    This template can operate in two modes:
    1. Generate mode (when content is None):
       Creates a message by calling LLM.
    2. Static mode (when content is provided):
       Creates a message with the given content.

    Examples:
        # Generate mode (LLM response)
        assistant_generate = AssistantTemplate()

        # Static mode
        assistant_static = AssistantTemplate(content="Hello, I'm here to help!")
    """

    def __init__(
        self,
        content: Optional[str] = None,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
        after_transform: Optional[Union[List[TransformHook], TransformHook]] = None,
    ):
        super().__init__(
            content=content or "",
            role="assistant",
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.is_generate = content is None

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        metadata = session.get_latest_metadata().copy()
        if self.is_generate:
            # Generate mode
            if session.runner is None:
                raise ValueError("runner is not set")
            logger.info(
                f"Generating content with {session.runner.models.__class__.__name__}..."
            )
            response = session.runner.models.send(session.runner.parameters, session)
            rendered_content = response.content
        else:
            # Static mode
            rendered_content = self.jinja_template.render(
                **session.get_latest_metadata()
            )

        message = Message(
            content=rendered_content,
            role=self.role,
            metadata=metadata,
        )
        session.append(message)
        yield message
        return session
