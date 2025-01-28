from abc import ABC, abstractmethod
from typing import List

from prompttrail.core import Message, Session


class SquashStrategy(ABC):
    """Base class defining message squashing strategy"""

    def initialize(self, parent_session: Session, subroutine_session: Session) -> None:
        """Initialize session information"""
        self.parent_session = parent_session
        self.subroutine_session = subroutine_session

    @abstractmethod
    def squash(self, messages: List[Message]) -> List[Message]:
        """Execute message squashing process

        Args:
            messages: List of messages to squash

        Returns:
            Squashed list of messages
        """


class LastMessageStrategy(SquashStrategy):
    """Strategy to retain only the last message"""

    def squash(self, messages: List[Message]) -> List[Message]:
        return [messages[-1]] if messages else []


class FilterByRoleStrategy(SquashStrategy):
    """Strategy to retain messages with specific roles"""

    def __init__(self, roles: List[str]):
        self.roles = roles

    def squash(self, messages: List[Message]) -> List[Message]:
        return [msg for msg in messages if msg.role in self.roles]
