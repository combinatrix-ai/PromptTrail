import logging
from abc import abstractmethod
from typing import Any, Callable, List, Optional

from prompttrail.core import Session
from prompttrail.core.utils import Debuggable

logger = logging.getLogger(__name__)


class Hook(Debuggable):
    """Base class for hooks in the agent template."""

    @abstractmethod
    def hook(self, session: Session) -> Any:
        """Execute the hook functionality on the session.

        Args:
            session: Current conversation session

        Returns:
            Modified session or other value depending on implementation
        """
        raise NotImplementedError("hook method is not implemented")


class TransformHook(Hook):
    """Hook that transforms the session."""

    def __init__(self, function: Optional[Callable[[Session], Session]] = None):
        self.function = function

    def hook(self, session: Session) -> Session:
        """Transform session using provided function.

        Args:
            session: Current conversation session

        Returns:
            Modified session

        Raises:
            ValueError: If no transform function is provided or set
        """
        if self.function is None:
            raise ValueError(
                "function is not set. TransformHook can be used two ways: "
                "1. Set function in constructor "
                "2. Inherit TransformHook and override hook method"
            )
        return self.function(session)


class BooleanHook(Hook):
    """Hook that evaluates a boolean condition."""

    def __init__(self, condition: Callable[[Session], bool]):
        self.condition = condition

    def hook(self, session: Session) -> bool:
        """Evaluate boolean condition on session.

        Args:
            session: Current conversation session

        Returns:
            Result of the condition evaluation
        """
        return self.condition(session)


class AskUserHook(TransformHook):
    """Hook that prompts user for input."""

    def __init__(
        self,
        key: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ):
        self.key = key
        self.description = description
        self.default = default

    def hook(self, session: Session) -> Session:
        """Get user input and store in session metadata.

        Args:
            session: Current conversation session

        Returns:
            Session with updated metadata containing user input
        """
        raw = input(self.description).strip()
        if raw == "" and self.default is not None:
            raw = self.default
        metadata = session.get_latest_metadata()
        metadata[self.key] = raw
        return session


class GenerateChatHook(TransformHook):
    """Hook that generates LLM response."""

    def __init__(self, key: str):
        self.key = key

    def hook(self, session: Session) -> Session:
        """Generate LLM response and store in session metadata.

        Args:
            session: Current conversation session

        Returns:
            Session with updated metadata containing LLM response

        Raises:
            ValueError: If session has no runner set
        """
        if session.runner is None:
            raise ValueError("Runner must be set to use GenerateChatHook")
        message = session.runner.models.send(session.runner.parameters, session)
        metadata = session.get_latest_metadata()
        metadata[self.key] = message.content
        return session


class CountUpHook(TransformHook):
    """Hook that increments a counter in metadata."""

    def hook(self, session: Session) -> Session:
        """Increment counter for current template.

        Args:
            session: Current conversation session

        Returns:
            Session with updated counter

        Raises:
            ValueError: If template_id is not set
        """
        template_id = session.get_current_template_id()
        if template_id is None:
            raise ValueError("template_id is not set")
        metadata = session.get_latest_metadata()
        metadata[template_id] = metadata.get(template_id, 0) + 1
        return session


class DebugHook(TransformHook):
    """Hook that prints debug information."""

    def __init__(self, message_shown_when_called: str):
        self.message = message_shown_when_called

    def hook(self, session: Session) -> Session:
        """Print debug info about session.

        Args:
            session: Current conversation session

        Returns:
            Unmodified session
        """
        print(f"{self.message} template_id: {session.get_current_template_id()}")
        print(f"{self.message} metadata: {session.get_latest_metadata()}")
        return session


class ResetDataHook(TransformHook):
    """Hook that resets metadata in session."""

    def __init__(self, keys: Optional[str | List[str]] = None):
        self.keys = (
            keys if isinstance(keys, list) else [keys] if keys is not None else []
        )

    def hook(self, session: Session) -> Session:
        """Reset specified or all metadata keys.

        Args:
            session: Current conversation session

        Returns:
            Session with reset metadata
        """
        metadata = session.get_latest_metadata()
        if self.keys:
            for key in self.keys:
                try:
                    metadata.pop(key)
                except KeyError:
                    logger.warning(f"Key {key} not found in metadata")
        else:
            metadata.clear()
        return session


class UpdateHook(TransformHook):
    """Hook that updates a metadata value."""

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    def hook(self, session: Session) -> Session:
        """Update specified metadata key with new value.

        Args:
            session: Current conversation session

        Returns:
            Session with updated metadata
        """
        metadata = session.get_latest_metadata()
        metadata[self.key] = self.value
        return session


class IncrementHook(TransformHook):
    """Hook that increments a numeric metadata value."""

    def __init__(self, key: str, by: int = 1, initial: int = 0):
        self.key = key
        self.increment = by
        self.initial = initial

    def hook(self, session: Session) -> Session:
        """Increment metadata value by specified amount.

        Args:
            session: Current conversation session

        Returns:
            Session with incremented metadata value
        """
        metadata = session.get_latest_metadata()
        if self.key not in metadata:
            metadata[self.key] = self.initial
            return session
        metadata[self.key] += self.increment
        return session
