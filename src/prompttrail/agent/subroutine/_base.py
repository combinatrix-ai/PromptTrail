"""Base classes for subroutine functionality."""

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
from prompttrail.core import Message, Session


class SubroutineTemplate(Template):
    """Template for executing subroutines with isolated session management.

    This template allows for executing other templates in an isolated session context,
    with flexible strategies for session initialization and message handling.

    Args:
        template: The template to execute as a subroutine
        template_id: Optional identifier for the template
        session_init_strategy: Strategy for initializing the subroutine session
        squash_strategy: Strategy for handling messages after subroutine execution
        before_transform: Transformers to apply before execution
        after_transform: Transformers to apply after execution
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
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.template = template
        self.session_init_strategy = session_init_strategy or CleanSessionStrategy()
        self.squash_strategy = squash_strategy or LastMessageStrategy()

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Execute the subroutine template with isolated session management.

        Args:
            session: The parent session context

        Returns:
            Generator yielding messages during execution and returning the final session

        Yields:
            Messages produced during subroutine execution
        """
        # Initialize subroutine session
        temp_session = self.session_init_strategy.initialize(session)
        self.squash_strategy.initialize(session, temp_session)

        messages: List[Message] = []
        try:
            # Execute subroutine
            for message in self.template.render(temp_session):
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
