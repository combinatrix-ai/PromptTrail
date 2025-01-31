from abc import ABC, abstractmethod
from typing import List

from prompttrail.core import Message, Model, Session


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


class LLMFilteringStrategy(SquashStrategy):
    """Strategy to filter messages using LLM with custom prompt"""

    def __init__(self, model: Model, prompt: str):
        """Initialize LLMFilteringStrategy

        Args:
            model: LLM model to use for filtering
            prompt: Custom prompt template for filtering messages
        """
        self.model = model
        self.prompt = prompt

    def squash(self, messages: List[Message]) -> List[Message]:
        """Filter messages using LLM with custom prompt

        Args:
            messages: List of messages to filter

        Returns:
            Filtered list of messages based on LLM response
        """
        if not messages:
            return []

        # Create conversation context
        conversation = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

        # Format prompt with conversation
        formatted_prompt = self.prompt.format(conversation=conversation)

        # Get filtering decision from LLM
        response = self.model.send(
            Session(messages=[Message(role="user", content=formatted_prompt)])
        )

        # Parse response to determine which messages to keep
        # Expecting response in format: "1,3,4" (indices of messages to keep)
        try:
            indices = [int(idx.strip()) for idx in response.content.split(",")]
            return [messages[i] for i in indices if 0 <= i < len(messages)]
        except (ValueError, IndexError):
            # If parsing fails, return all messages
            return messages


class LLMSummarizingStrategy(SquashStrategy):
    """Strategy to summarize messages using LLM with custom prompt"""

    def __init__(self, model: Model, prompt: str):
        """Initialize LLMSummarizingStrategy

        Args:
            model: LLM model to use for summarization
            prompt: Custom prompt template for summarizing messages
        """
        self.model = model
        self.prompt = prompt

    def squash(self, messages: List[Message]) -> List[Message]:
        """Summarize messages using LLM with custom prompt

        Args:
            messages: List of messages to summarize

        Returns:
            List containing a single summarized message
        """
        if not messages:
            return []

        # Create conversation context
        conversation = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)

        # Format prompt with conversation
        formatted_prompt = self.prompt.format(conversation=conversation)

        # Get summary from LLM
        response = self.model.send(
            Session(messages=[Message(role="user", content=formatted_prompt)])
        )

        # Create a new message with the summary
        summary_message = Message(
            role="assistant",
            content=response.content,
        )

        return [summary_message]
