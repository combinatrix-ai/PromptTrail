"""Base classes for subroutine functionality."""

import copy
from typing import Generator, List, Optional, Union

from prompttrail.agent.session_transformers import SessionTransformer
from prompttrail.agent.subroutine.session_init_strategy import (
    CleanSessionStrategy,
    SessionInitStrategy,
)
from prompttrail.agent.subroutine.squash_strategy import (
    LastMessageStrategy,
    SquashStrategy,
)
from prompttrail.agent.templates import Stack, Template
from prompttrail.agent.templates._core import Event
from prompttrail.core import Message, Model, Runner, Session


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
        runner: Optional[Runner] = None,  # Complete environment override
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
        self.session_init_strategy = session_init_strategy or CleanSessionStrategy()
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
        temp_session = self.session_init_strategy.initialize(session)
        self.squash_strategy.initialize(session, temp_session)

        # Set up isolated environment
        if self.runner:
            temp_session.runner = self.runner
        elif self.model:
            if not session.runner:
                raise ValueError("Parent runner is required when using model override")
            # Create new runner with overridden model but inherit other settings
            temp_session.runner = type(session.runner)(
                model=self.model,
                template=session.runner.template,
                user_interface=session.runner.user_interface,
            )
        elif session.runner:
            # Create copy of parent runner for isolation
            temp_session.runner = copy.deepcopy(session.runner)

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
            selected_messages = self.squash_strategy.squash(messages)
            for msg in selected_messages:
                session.append(msg)

        finally:
            # Temporary session will be garbage collected
            pass

        return session

    def create_stack(self, session: Session) -> Stack:
        """Create stack frame for this template.

        Args:
            session: The current session context

        Returns:
            Stack frame for subroutine execution
        """
        return Stack(template_id=self.template_id)
