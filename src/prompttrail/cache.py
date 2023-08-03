from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Optional

from cachetools import LRUCache

if TYPE_CHECKING:
    from prompttrail.core import Message, Parameters, Session


class CacheProvider(metaclass=ABCMeta):
    @abstractmethod
    def add(self, session: "Session", message: "Message") -> None:
        raise NotImplementedError("add method is not implemented")

    @abstractmethod
    def search(
        self, parameters: "Parameters", session: "Session"
    ) -> Optional["Message"]:
        raise NotImplementedError("search method is not implemented")


class LRUCacheProvider(CacheProvider):
    def __init__(self, n_items: int):
        self.cache: LRUCache["Session", "Message"] = LRUCache(n_items)

    def add(self, session: "Session", message: "Message"):
        self.cache[session] = message

    def search(self, parameters: "Parameters", session: "Session") -> "Message":
        return self.cache[session]
