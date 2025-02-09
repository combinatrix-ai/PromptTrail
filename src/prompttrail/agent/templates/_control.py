import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Generator, List, Optional, Sequence, Set, TypeAlias, Union

from prompttrail.agent.session_transformers._core import SessionTransformer
from prompttrail.agent.templates._core import Event, Stack, Template
from prompttrail.core import Message, Session
from prompttrail.core.const import (
    END_TEMPLATE_ID,
    BreakException,
    ReachedEndTemplateException,
)

logger = logging.getLogger(__name__)

TemplateId: TypeAlias = str


class ControlTemplate(Template, metaclass=ABCMeta):
    """
    Base class for control flow templates.
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
        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
            enable_logging=enable_logging,
        )

    @abstractmethod
    def walk(
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        """
        Traverse the template and its child templates in a depth-first manner. Control templates should override this method as they usually have child templates.

        Args:
            visited_templates: Set of visited templates to avoid infinite recursion.

        Yields:
            The template and its child templates.
        """
        raise NotImplementedError(
            "Derived class of ControlTemplate must implement its own walk method"
        )

    @abstractmethod
    def create_stack(self, session: "Session") -> "Stack":
        """
        Create a stack for the control template.

        Args:
            session: The current session of the conversation.

        Returns:
            The created stack.
        """
        raise NotImplementedError(
            "Derived class of ControlTemplate must implement its own create_stack method"
        )


class LoopTemplateStack(Stack):
    """
    Stack for LoopTemplate.
    """

    cumulative_index: int
    n_children: int

    def get_loop_idx(self) -> int:
        """
        Get the index of the current loop.

        Returns:
            The index of the current loop.
        """
        return self.cumulative_index // self.n_children

    def get_idx(self) -> int:
        """
        Get the index of the current child template.

        Returns:
            The index of the current child template.
        """
        return self.cumulative_index % self.n_children

    def next(self) -> None:
        """
        Move to the next child template.
        """
        self.cumulative_index += 1


class LoopTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        exit_condition: Optional[Callable[[Session], bool]] = None,
        template_id: Optional[str] = None,
        exit_loop_count: Optional[int] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        """A template for a loop control flow. Unlike `LinearTemplate`, this template loops over the child templates until the exit condition is met.

        Args:
            templates (Sequence[Template]): Templates to be looped. Execution order is the same as the order of the list.
            exit_condition (Optional[Condition], optional): If set, the loop is broken when the condition is met. Defaults to None (Infinite loop).
            template_id (Optional[str], optional): Template ID of this template. Defaults to None.
            exit_loop_count (Optional[int], optional): If set, the loop is broken when the loop count is over this number. Defaults to None (Infinite loop).
            before_transform (Optional[List[TransformHook]], optional): `TrnasformHook`s to be applied before rendering. Defaults to None.
            after_transform (Optional[List[TransformHook]], optional): `TrnasformHook`s to be applied after rendering. Defaults to None.
        """
        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
            enable_logging=enable_logging,
        )
        self.templates = templates
        self.exit_condition = exit_condition
        self.exit_loop_count = exit_loop_count

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        stack = session.stack[-1]
        if not isinstance(stack, LoopTemplateStack):
            raise RuntimeError("LoopTemplateStack is not the last stack")
        while True:
            idx = stack.get_idx()
            template = self.templates[idx]
            gen = template.render(session)
            try:
                session = yield from gen
            except BreakException:
                # Break the loop if a BreakException is raised.
                break
            if self.exit_condition and self.exit_condition(session):
                # Break the loop if the exit condition is met.
                break
            stack.next()
            if (
                self.exit_loop_count is not None
                and stack.get_loop_idx() >= self.exit_loop_count
            ):
                self.warning(
                    "Loop count is over %s. Breaking the loop.", self.exit_loop_count
                )
                break
        return session

    def walk(
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        visited_templates = visited_templates or set()
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)

    def create_stack(self, session: "Session") -> "Stack":
        return LoopTemplateStack(
            template_id=self.template_id,
            n_children=len(self.templates),
            cumulative_index=0,
        )


class IfTemplate(ControlTemplate):
    def __init__(
        self,
        condition: Callable[[Session], bool],
        true_template: Template,
        false_template: Optional[Template] = None,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        """A template for a conditional control flow.

        Args:
            condition (Condition): Condition to be checked.
            true_template (Template): Template to be rendered if the condition is met.
            false_template (Optional[Template], optional): Template to be rendered if the condition is not met. Defaults to None. If None, this template return no message.
            template_id (Optional[str], optional): Template ID of this template. Defaults to None.
            before_transform (Optional[List[TransformHook]], optional): `TrnasformHook`s to be applied before rendering. Defaults to None.
            after_transform (Optional[List[TransformHook]], optional): `TrnasformHook`s to be applied after rendering. Defaults to None.
        """

        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
            enable_logging=enable_logging,
        )
        self.true_template = true_template
        self.false_template = false_template
        self.condition = condition

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        if self.condition(session):
            session = yield from self.true_template.render(session)
        else:
            if self.false_template is None:
                pass  # Just skip to the next template
            else:
                session = yield from self.false_template.render(session)
        return session

    def walk(
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        visited_templates = visited_templates or set()
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self
        yield from self.true_template.walk(visited_templates)
        if self.false_template is not None:
            yield from self.false_template.walk(visited_templates)

    def create_stack(self, session: "Session") -> "Stack":
        return Stack(template_id=self.template_id)


class LinearTemplateStack(Stack):
    """
    Stack for LinearTemplate. It stores the index of child templates currently rendering in `LinearTemplate`.
    """

    idx: int


class LinearTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        enable_logging: bool = True,
    ):
        """A template for a linear control flow. Unlike `LoopTemplate`, this template exits after rendering all the child templates.

        Args:
            templates (Sequence[Template]): Templates to be rendered. Execution order is the same as the order of the list.
            template_id (Optional[str], optional): Template ID of this template. Defaults to None.
            before_transform (Optional[List[TransformHook]], optional): `TrnasformHook`s to be applied before rendering. Defaults to None.
            after_transform (Optional[List[TransformHook]], optional): `TrnasformHook`s to be applied after rendering. Defaults to None.
        """
        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
            enable_logging=enable_logging,
        )
        self.templates = templates

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        stack = session.stack[-1]
        if not isinstance(stack, LinearTemplateStack):
            raise RuntimeError("LinearTemplateStack is not the last stack")

        while 1:
            try:
                session = yield from self.templates[stack.idx].render(session)
            except BreakException:
                break
            stack.idx += 1
            # Break the loop when the last template is rendered
            if stack.idx >= len(self.templates):
                break
        return session

    def walk(
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        visited_templates = visited_templates or set()
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)

    def create_stack(self, session: "Session") -> LinearTemplateStack:
        return LinearTemplateStack(template_id=self.template_id, idx=0)


# EndTemplate is a singleton
class EndTemplate(Template):
    """
    A special template for the end of the conversation.

    When runner reaches this template, the conversation is forced to stop. This template is a singleton.
    before_transform and after_transform are unavailable for `EndTemplate`.
    """

    template_id = END_TEMPLATE_ID
    _instance = None
    before_transform = []

    def __init__(self, farewell_message: str | None = None):
        self.farewell_message = farewell_message

    def __new__(cls, farewell_message: str | None = None):
        if cls._instance is None:
            cls._instance = super(EndTemplate, cls).__new__(cls)
        return cls._instance

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        if self.farewell_message:
            yield Message(
                content=self.farewell_message,
                role="assistant",
                metadata=session.metadata,
            )
            session.messages.append(
                Message(content=self.farewell_message, role="assistant")
            )
        # farewell_message is consumed. Reset it to None.
        raise ReachedEndTemplateException()

    def create_stack(self, session: "Session") -> Stack:
        return super().create_stack(session)


class BreakTemplate(ControlTemplate):
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
    ):
        """A template for breaking the loop.

        after_transform is unavailable for `BreakTemplate` because it may skipped when the break is executed.
        """
        super().__init__(
            template_id=template_id,
            before_transform=before_transform if before_transform is not None else [],
            # after_transform is unavailable for the templates that raise errors
        )

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        self.info("Breaking the loop from %s.", self.template_id)
        raise BreakException()

    def walk(
        self, visited_templates: Optional[Set["Template"]] = None
    ) -> Generator["Template", None, None]:
        visited_templates = visited_templates or set()
        yield self

    def create_stack(self, session: "Session") -> Stack:
        return Stack(template_id=self.template_id)
