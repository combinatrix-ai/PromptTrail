from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional

from cachetools import LRUCache

if TYPE_CHECKING:
    from prompttrail.core import Config, Message, Session


class CacheProvider(metaclass=ABCMeta):
    """Abstract base class for cache providers.

    Cache providers are responsible for storing and retrieving messages from a cache.
    Each provider must implement add() and search() methods.
    """

    @abstractmethod
    def add(self, session: "Session", message: "Message") -> None:
        """Add a message to the cache.

        Args:
            session: Session associated with the message
            message: Message to cache

        Note:
            Cache should also consider Parameters when storing messages
        """
        raise NotImplementedError("add method is not implemented")

    @abstractmethod
    def search(self, config: "Config", session: "Session") -> Optional["Message"]:
        """Search for a cached message.

        Args:
            config: Config associated with the message
            session: Session to search for

        Returns:
            Cached message if found, None otherwise
        """
        raise NotImplementedError("search method is not implemented")


class LRUCacheProvider(CacheProvider):
    """LRU (Least Recently Used) cache implementation.

    Caches messages in an LRU cache with a fixed maximum size.
    """

    def __init__(self, n_items: int = 10000):
        """Initialize LRU cache.

        Args:
            n_items: Maximum number of items in cache (default: 10000)
        """
        self.cache: LRUCache["Session", "Message"] = LRUCache(n_items)

    def add(self, session: "Session", message: "Message") -> None:
        """Add message to cache.

        Args:
            session: Session associated with the message
            message: Message to cache
        """
        self.cache[session] = message

    def search(self, parameters: "Config", session: "Session") -> Optional["Message"]:
        """Search for cached message.

        Args:
            config: Config associated with the message
            session: Session to search for

        Returns:
            Cached message if found, None otherwise
        """
        return self.cache[session]
