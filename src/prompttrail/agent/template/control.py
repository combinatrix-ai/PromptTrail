import logging
from abc import abstractmethod
from typing import Generator, List, Optional, Sequence, Set, TypeAlias

from prompttrail.agent.core import State
from prompttrail.agent.hook.core import BooleanHook, TransformHook
from prompttrail.agent.template.core import Stack, Template
from prompttrail.const import (
    END_TEMPLATE_ID,
    BreakException,
    JumpException,
    ReachedEndTemplateException,
)
from prompttrail.core import Message

logger = logging.getLogger(__name__)

TemplateId: TypeAlias = str


class ControlTemplate(Template):
    # ControlTemplate must handle its child templates.

    @abstractmethod
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )

    @abstractmethod
    def walk(
        self, visited_templates: Set["Template"] = set()
    ) -> Generator["Template", None, None]:
        raise NotImplementedError(
            "Derived class of ControlTemplate must implement its own walk method"
        )

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
        exit_condition: Optional[BooleanHook] = None,
        template_id: Optional[str] = None,
        exit_loop_count: Optional[int] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
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
            try:
                state = yield from gen
            except BreakException:
                # TODO: Should give state back?
                break
            if self.exit_condition and self.exit_condition.hook(state):
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
        self, visited_templates: Set["Template"] = set()
    ) -> Generator["Template", None, None]:
        if self in visited_templates:
            return
        visited_templates.add(self)
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
        condition: BooleanHook,
        true_template: Template,
        false_template: Optional[Template] = None,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.true_template = true_template
        self.false_template = false_template
        self.condition = condition

    def _render(self, state: "State") -> Generator[Message, None, State]:
        if self.condition.hook(state):
            state = yield from self.true_template.render(state)
        else:
            if self.false_template is None:
                pass  # Just skip to next template
            else:
                state = yield from self.false_template.render(state)
        return state

    def walk(
        self, visited_templates: Set["Template"] = set()
    ) -> Generator["Template", None, None]:
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self
        yield from self.true_template.walk(visited_templates)
        if self.false_template is not None:
            yield from self.false_template.walk(visited_templates)

    def create_stack(self, state: "State") -> "Stack":
        return Stack(template_id=self.template_id)


class LinearTemplateStack(Stack):
    idx: int


class LinearTemplate(ControlTemplate):
    def __init__(
        self,
        templates: Sequence[Template],
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.templates = templates

    def _render(self, state: "State") -> Generator[Message, None, State]:
        stack = state.stack[-1]
        if not isinstance(stack, LinearTemplateStack):
            raise RuntimeError("LinearTemplateStack is not the last stack")

        while 1:
            try:
                state = yield from self.templates[stack.idx].render(state)
            except BreakException:
                break
            stack.idx += 1
            # when the last template is rendered, break the loop
            if stack.idx >= len(self.templates):
                break
        return state

    def walk(
        self, visited_templates: Set["Template"] = set()
    ) -> Generator["Template", None, None]:
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self
        for template in self.templates:
            yield from template.walk(visited_templates)

    def create_stack(self, state: "State") -> LinearTemplateStack:
        return LinearTemplateStack(template_id=self.template_id, idx=0)


# EndTemplate is singleton
class EndTemplate(Template):
    template_id = END_TEMPLATE_ID
    _instance = None
    before_transform = []

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EndTemplate, cls).__new__(cls)
        return cls._instance

    def _render(self, state: "State") -> Generator[Message, None, State]:
        raise ReachedEndTemplateException()

    def create_stack(self, state: State) -> Stack:
        return super().create_stack(state)


class JumpTemplate(ControlTemplate):
    def __init__(
        self,
        jump_to: Template | TemplateId,
        condition: BooleanHook,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            # after_transform is unavailable for the templates raise errors
        )
        if isinstance(jump_to, Template):
            jump_to = jump_to.template_id
        self.jump_to = jump_to
        self.condition = condition

    def _render(self, state: "State") -> Generator[Message, None, State]:
        logger.warning(
            msg=f"Jumping to {self.jump_to} from {self.template_id}. This reset the stack therefore the dialogue will not come back to this template."
        )
        raise JumpException(self.jump_to)

    def walk(
        self, visited_templates: Set["Template"] = set()
    ) -> Generator["Template", None, None]:
        if self in visited_templates:
            return
        visited_templates.add(self)
        yield self

    def create_stack(self, state: "State") -> LinearTemplateStack:
        return LinearTemplateStack(template_id=self.template_id, idx=0)


class BreakTemplate(ControlTemplate):
    def __init__(
        self,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            # after_transform is unavailable for the templates raise errors
        )

    def _render(self, state: "State") -> Generator[Message, None, State]:
        logger.warning(msg=f"Breaking the loop from {self.template_id}.")
        raise BreakException()

    def walk(
        self, visited_templates: Set[Template] = set()
    ) -> Generator[Template, None, None]:
        yield self

    def create_stack(self, state: "State") -> Stack:
        return Stack(template_id=self.template_id)
