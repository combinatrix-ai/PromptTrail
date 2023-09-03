from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional

from cachetools import LRUCache

if TYPE_CHECKING:
    from prompttrail.core import Message, Parameters, Session


class CacheProvider(metaclass=ABCMeta):
    """
    Abstract base class for cache providers.

    Cache providers are responsible for storing and retrieving messages from a cache.
    Cache providers implement the `add` and `search` methods.
    """

    @abstractmethod
    def add(self, session: "Session", message: "Message") -> None:
        # TODO: Cache must see `Parameters` as well!
        """
        Add a message to the cache based on the session and parameters.

        Args:
            session: The session associated with the message.
            message: The message to be added to the cache.
        """
        raise NotImplementedError("add method is not implemented")

    @abstractmethod
    def search(
        self, parameters: "Parameters", session: "Session"
    ) -> Optional["Message"]:
        """
        Search for a message in the cache.

        Args:
            parameters: The parameters associated with the message.
            session: The session associated with the message.

        Returns:
            The message found in the cache, or None if no message is found.
        """
        raise NotImplementedError("search method is not implemented")


class LRUCacheProvider(CacheProvider):
    """
    Cache provider implementation using an LRU (Least Recently Used) cache.

    This cache provider stores messages in an LRU cache with a fixed number of items.
    """

    def __init__(self, n_items: int = 10000):
        """
        Initialize the LRUCacheProvider.

        Args:
            n_items: The maximum number of items to store in the cache.
        """
        self.cache: LRUCache["Session", "Message"] = LRUCache(n_items)

    def add(self, session: "Session", message: "Message"):
        """
        Add a message to the cache.

        Args:
            session: The session associated with the message.
            message: The message to be added to the cache.
        """
        self.cache[session] = message

    def search(
        self, parameters: "Parameters", session: "Session"
    ) -> Optional["Message"]:
        """
        Search for a message in the cache.

        Args:
            parameters: The parameters associated with the message.
            session: The session associated with the message.

        Returns:
            The message found in the cache, or None if no message is found.
        """
        return self.cache[session]
