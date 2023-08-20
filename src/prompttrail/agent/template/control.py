import logging
from abc import abstractmethod
from typing import Generator, List, Optional, Sequence

from prompttrail.agent.core import State
from prompttrail.agent.hook.core import BooleanHook, TransformHook
from prompttrail.agent.template.core import Stack, Template
from prompttrail.const import END_TEMPLATE_ID
from prompttrail.core import Message

logger = logging.getLogger(__name__)


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
        self, visited_templates: Sequence[Template] = []
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
        exit_condition: BooleanHook,
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
