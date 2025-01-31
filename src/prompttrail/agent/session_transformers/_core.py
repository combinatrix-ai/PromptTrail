import logging
from abc import abstractmethod
from typing import Any, List, Optional, final

from prompttrail.core import Metadata, Session
from prompttrail.core.utils import Debuggable

logger = logging.getLogger(__name__)


class SessionTransformer(Debuggable):
    """Base class for hooks in the agent template."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, session: Session) -> Session:
        """Execute the hook functionality on the session.

        Args:
            session: Current conversation session

        Returns:
            Modified session
        """
        raise NotImplementedError("hook method is not implemented")


class MetadataTransformer(SessionTransformer):
    """Hook that transforms the session."""

    def __init__(self):
        super().__init__()

    @final
    def process(self, session: Session) -> Session:
        """Execute the hook functionality on the session.

        Args:
            session: Current conversation session

        Returns:
            Modified session
        """
        session.metadata = self.process_metadata(session.metadata, session)
        return session

    @abstractmethod
    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        """Transform session using provided function.

        Args:
            metadata: Current metadata
            session: Current conversation session

        Returns:
            Modified metadata
        """
        raise NotImplementedError("transform method is not implemented")


class LambdaSessionTransformer(SessionTransformer):
    """Hook that transforms the session using a lambda function."""

    def __init__(self, transform_fn):
        super().__init__()
        self.transform_fn = transform_fn

    def process(self, session: Session) -> Session:
        """Execute the hook functionality on the session.

        Args:
            session: Current conversation session

        Returns:
            Modified session
        """
        session.metadata = self.transform_fn(session)
        return session


class Debugger(MetadataTransformer):
    """Hook that prints debug information."""

    def __init__(self, message_shown_when_called: str):
        super().__init__()
        self.message = message_shown_when_called

    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        """Print debug info about session.

        Args:
            metadata: Current metadata
            session: Current conversation session

        Returns:
            Unmodified session
        """
        print(f"{self.message} template_id: {session.get_current_template_id()}")
        print(f"{self.message} metadata: {session.metadata}")
        return metadata


class ResetMetadata(MetadataTransformer):
    """Hook that resets metadata in session."""

    def __init__(self, keys: Optional[str | List[str]] = None):
        super().__init__()
        self.keys = (
            keys if isinstance(keys, list) else [keys] if keys is not None else []
        )

    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        """Reset specified or all metadata keys.

        Args:
            metadata: Current metadata
            session: Current conversation session

        Returns:
            Session with reset metadata
        """
        if self.keys:
            for key in self.keys:
                try:
                    metadata.pop(key)
                except KeyError:
                    raise KeyError(f"Key {key} not found in metadata: %s", metadata)
        else:
            metadata.clear()
        return metadata


class UpdateMetadata(MetadataTransformer):
    """Hook that updates a metadata value."""

    def __init__(self, key: str, value: Any):
        super().__init__()
        self.key = key
        self.value = value

    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        """Update specified metadata key with new value.

        Args:
            metadata: Current metadata
            session: Current conversation session

        Returns:
            Session with updated metadata
        """
        metadata[self.key] = self.value
        return metadata
