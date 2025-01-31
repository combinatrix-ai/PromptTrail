import logging
from abc import ABCMeta, abstractmethod
from pprint import pformat
from typing import Generator, List, Optional, Set, Union
from uuid import uuid4

import jinja2

from prompttrail.agent.session_transformers._core import SessionTransformer
from prompttrail.agent.templates._base import Stack
from prompttrail.core import Message, MessageRoleType, Session
from prompttrail.core.const import (
    RESERVED_TEMPLATE_IDS,
    BreakException,
    JumpException,
    ReachedEndTemplateException,
)
from prompttrail.core.utils import Debuggable

logger = logging.getLogger(__name__)


def check_template_id(template_id: str) -> None:
    """Check if template ID is not reserved."""
    if template_id in RESERVED_TEMPLATE_IDS:
        raise ValueError(
            f"Template id {template_id} is reserved. Please use another template id."
        )


class Template(Debuggable, metaclass=ABCMeta):
    """Base template class for creating messages and controlling flow.

    Subclasses must implement _render() and create_stack() methods.
    """

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        super().__init__()
        self.template_id = template_id if template_id else self._generate_name()
        check_template_id(self.template_id)
        self.before_transform = self._hooks_to_list(before_transform)
        self.after_transform = self._hooks_to_list(after_transform)
        self.enable_logging = enable_logging

    def _hooks_to_list(
        self, hooks: Optional[Union[List[SessionTransformer], SessionTransformer]]
    ) -> List[SessionTransformer]:
        """Convert hook(s) to list format."""
        if hooks is None:
            return []
        return [hooks] if not isinstance(hooks, list) else hooks

    def render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Render template with hooks and error handling."""
        logging.debug(f"Rendering {self.template_id}")
        session.push_stack(self.create_stack(session))
        try:
            for hook in self.before_transform:
                session = hook.process(session)
            res = yield from self._render(session)
            for hook in self.after_transform:
                session = hook.process(session)
        except (BreakException, ReachedEndTemplateException, JumpException) as e:
            raise e
        except Exception as e:
            self.error(f"RenderingTemplateError@{self.template_id}")
            raise e
        finally:
            session.pop_stack()
        logging.debug(f"Rendered {self.template_id}")
        return res

    @abstractmethod
    def create_stack(self, session: "Session") -> "Stack":
        """Create stack frame for this template."""
        return Stack(template_id=self.template_id)

    @abstractmethod
    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Render the template and return message generator."""
        raise NotImplementedError("render method is not implemented")

    def walk(
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        """Walk through template tree yielding each template once."""
        visited_templates = visited_templates or set()
        if self not in visited_templates:
            visited_templates.add(self)
            yield self

    def __str__(self) -> str:
        return f"Template(id={self.template_id})"

    @classmethod
    def _generate_name(cls) -> str:
        """Generate unique template ID."""
        return f"Unnamed_{cls.__name__}_{str(uuid4())}"


class MessageTemplate(Template):
    """Template that creates a message using Jinja2 templating."""

    def __init__(
        self,
        content: str,
        role: MessageRoleType,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            enable_logging=enable_logging,
        )
        self.content = content
        self.jinja_template = jinja2.Template(self.content)
        self.role = role

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Render message using Jinja template."""
        session.set_jump(None)
        if not session.get_jump():
            rendered_content = self.jinja_template.render(**session.metadata)
            message = Message(
                content=rendered_content, role=self.role, metadata=session.metadata
            )
            session.append(message)
            yield message
        return session

    def create_stack(self, session: "Session") -> "Stack":
        return Stack(template_id=self.template_id)

    def __str__(self) -> str:
        content_part = (
            f'content="""\n{self.content}\n"""'
            if "\n" in self.content
            else f'content="{self.content}"'
        )
        before_part = (
            f", before_transform={pformat(self.before_transform)}"
            if self.before_transform
            else ""
        )
        after_part = (
            f", after_transform={pformat(self.after_transform)}"
            if self.after_transform
            else ""
        )
        return f"MessageTemplate(id={self.template_id}, {content_part}{before_part}{after_part})"


class GenerateTemplate(MessageTemplate):
    """Template that generates content using an LLM."""

    def __init__(
        self,
        role: MessageRoleType,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging=True,
    ):
        super().__init__(
            content="",
            role=role,
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Generate content using LLM."""
        if not session.runner:
            raise ValueError("runner is not set")

        self.info(
            "Generating content with %s...", session.runner.models.__class__.__name__
        )
        response = session.runner.models.send(session)
        if self.role:
            response.role = self.role
        session.append(response)
        yield response
        return session


class SystemTemplate(MessageTemplate):
    """Template for system messages."""

    def __init__(
        self,
        content: str,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging=True,
    ):
        super().__init__(
            content=content,
            role="system",
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )


class UserTemplate(MessageTemplate):
    """Template for user messages with optional interactive input."""

    def __init__(
        self,
        content: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[str] = None,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        super().__init__(
            content=content or "",
            role="user",
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            enable_logging=enable_logging,
        )
        self.is_interactive = content is None
        self.description = description
        self.default = default

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Render user message, either from input or template."""
        metadata = session.metadata
        if self.is_interactive:
            if not session.runner:
                raise ValueError("Runner must be given to use interactive mode")
            rendered_content = session.runner.user_interface.ask(
                session, self.description, self.default
            )
        else:
            rendered_content = self.jinja_template.render(**metadata)

        message = Message(
            content=rendered_content,
            role=self.role,
            metadata=metadata,
        )
        session.append(message)
        yield message
        return session


class AssistantTemplate(MessageTemplate):
    """Template for assistant messages with optional LLM generation."""

    def __init__(
        self,
        content: Optional[str] = None,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        super().__init__(
            content=content or "",
            role="assistant",
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
            enable_logging=enable_logging,
        )
        self.is_generate = content is None

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        """Render assistant message, either generated or from template."""
        metadata = session.metadata

        if self.is_generate:
            if not session.runner:
                raise ValueError("runner is not set")
            self.info(
                "Generating content with %s...",
                session.runner.models.__class__.__name__,
            )
            response = session.runner.models.send(session)
            rendered_content = response.content
        else:
            rendered_content = self.jinja_template.render(**metadata)

        message = Message(
            content=rendered_content,
            role=self.role,
            metadata=metadata,
        )
        session.append(message)
        yield message
        return session
