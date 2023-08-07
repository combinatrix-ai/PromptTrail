from abc import abstractmethod
from typing import Any, Optional


class UserInteractionProvider(object):
    @abstractmethod
    def ask(self, description: Any, default: Any = None) -> Any:
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextProvider(UserInteractionProvider):
    @abstractmethod
    def ask(self, description: str, default: Optional[str] = None) -> str:
        raise NotImplementedError("ask method is not implemented")


class UserInteractionTextCLIProvider(UserInteractionTextProvider):
    def ask(self, description: str, default: Optional[str] = None) -> str:
        raw = input(description).strip()
        if (not raw) and default is not None:
            raw = default
        return raw
