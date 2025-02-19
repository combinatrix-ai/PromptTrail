"""Base classes for subroutine functionality."""

from typing import TYPE_CHECKING, Generator, List, Optional, Union

from prompttrail.agent.session_transformers import SessionTransformer
from prompttrail.agent.subroutine.session_init_strategy import (
    InheritMetadataStrategy,
    SessionInitStrategy,
)
from prompttrail.agent.subroutine.squash_strategy import (
    LastMessageStrategy,
    SquashStrategy,
)
from prompttrail.agent.templates import Stack, Template
from prompttrail.agent.templates._core import Event
from prompttrail.core import Message, Model, Session

if TYPE_CHECKING:
    from prompttrail.agent.runners import Runner


class SubroutineTemplate(Template):
    """Template for executing subroutines with isolated session management.

    This template allows for executing other templates in an isolated session context,
    with flexible strategies for session initialization and message handling.
    The execution environment is completely isolated from the parent session,
    with the ability to override the model or provide a custom runner.

    Args:
        template: The template to execute as a subroutine
        template_id: Optional identifier for the template
        session_init_strategy: Strategy for initializing the subroutine session
        squash_strategy: Strategy for handling messages after subroutine execution
        before_transform: Transformers to apply before execution
        after_transform: Transformers to apply after execution
        runner: Optional complete runner override for isolated environment
        model: Optional model override (cannot be used with runner)
    """

    def __init__(
        self,
        template: Template,
        template_id: Optional[str] = None,
        session_init_strategy: Optional[SessionInitStrategy] = None,
        squash_strategy: Optional[SquashStrategy] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        runner: Optional["Runner"] = None,  # Complete environment override
        model: Optional[Model] = None,  # Model-only override
    ):
        """Initialize SubroutineTemplate.

        Args:
            template: The template to execute as a subroutine
            template_id: Optional identifier for the template
            session_init_strategy: Strategy for initializing the subroutine session
            squash_strategy: Strategy for handling messages after subroutine execution
            before_transform: Transformers to apply before execution
            after_transform: Transformers to apply after execution
            runner: Optional complete runner override for isolated environment
            model: Optional model override (cannot be used with runner)

        Raises:
            ValueError: If both runner and model are provided
        """
        if runner is not None and model is not None:
            raise ValueError("Cannot set both runner and model - use one or the other")

        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.template = template
        self.session_init_strategy = session_init_strategy or InheritMetadataStrategy()
        self.squash_strategy = squash_strategy or LastMessageStrategy()
        self.runner = runner
        self.model = model

    def _render(
        self, session: "Session"
    ) -> Generator[Union[Message, Event], None, "Session"]:
        """Execute the subroutine template with isolated session management.

        Args:
            session: The parent session context

        Returns:
            Generator yielding messages during execution and returning the final session

        Yields:
            Messages produced during subroutine execution

        Raises:
            ValueError: If runner is required but not set
        """
        # Initialize subroutine session
        parent_session = session
        del session  # For safety
        temp_session = self.session_init_strategy.initialize(parent_session)

        # Set up isolated environment
        if self.runner:
            # TODO: Should we allow model override with runner? This will break Template search etc.
            temp_session.runner = self.runner
        else:
            if self.model:
                if not parent_session.runner:
                    raise ValueError(
                        "Parent runner is required when using model override"
                    )
                # Create new runner with overridden model but inherit other settings
                temp_session.runner = type(parent_session.runner)(
                    model=self.model,
                    # This template is parent template
                    template=parent_session.runner.template,
                    user_interface=parent_session.runner.user_interface,
                )
            else:
                if not parent_session.runner:
                    raise ValueError("Runner is required for subroutine execution")
                # Create copy of parent runner for isolation
                temp_session.runner = type(parent_session.runner)(
                    model=parent_session.runner.model,
                    # This template is parent template
                    template=parent_session.runner.template,
                    user_interface=parent_session.runner.user_interface,
                )

        messages: List[Message] = []
        try:
            # Execute subroutine
            for message in self.template.render(temp_session):
                if isinstance(message, Event):
                    # TODO: Should raise error?
                    yield message
                    continue
                messages.append(message)
                yield message

            # Apply squash strategy
            squashed_messages = self.squash_strategy.squash(messages)
            for msg in squashed_messages:
                parent_session.append(msg)
        except Exception as e:
            raise e
        finally:
            # Temporary session will be garbage collected
            pass

        return parent_session

    def create_stack(self, session: Session) -> Stack:
        """Create stack frame for this template.

        Args:
            session: The current session context

        Returns:
            Stack frame for subroutine execution
        """
        return Stack(template_id=self.template_id)
