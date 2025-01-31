from abc import ABC, abstractmethod
from typing import Callable

from prompttrail.core import Message, Session


class SessionInitStrategy(ABC):
    """Base class defining how to initialize subroutine session"""

    @abstractmethod
    def initialize(self, parent_session: Session) -> Session:
        """Create and initialize a session for subroutine

        Args:
            parent_session: Parent session that invoked the subroutine

        Returns:
            Initialized session for subroutine use
        """


class CleanSessionStrategy(SessionInitStrategy):
    """Strategy to create a clean session with no messages"""

    def initialize(self, parent_session: Session) -> Session:
        return Session(
            metadata=parent_session.metadata.copy(), runner=parent_session.runner
        )


class InheritSystemStrategy(SessionInitStrategy):
    """Strategy to inherit system messages from parent"""

    def initialize(self, parent_session: Session) -> Session:
        system_messages = [
            msg for msg in parent_session.messages if msg.role == "system"
        ]
        return Session(
            messages=system_messages.copy(),
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner,
        )


class LastNMessagesStrategy(SessionInitStrategy):
    """Strategy to inherit last N messages from parent"""

    def __init__(self, n: int):
        self.n = n

    def initialize(self, parent_session: Session) -> Session:
        last_messages = (
            parent_session.messages[-self.n :] if parent_session.messages else []
        )
        return Session(
            messages=last_messages.copy(),
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner,
        )


class FilteredInheritStrategy(SessionInitStrategy):
    """Strategy to inherit messages based on custom filter"""

    def __init__(self, filter_fn: Callable[[Message], bool]):
        self.filter_fn = filter_fn

    def initialize(self, parent_session: Session) -> Session:
        filtered_messages = [
            msg for msg in parent_session.messages if self.filter_fn(msg)
        ]
        return Session(
            messages=filtered_messages.copy(),
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner,
        )
