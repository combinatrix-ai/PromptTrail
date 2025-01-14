import logging
from abc import abstractmethod
from typing import Any, Callable, Optional

from prompttrail.core import Session

logger = logging.getLogger(__name__)


class Hook(object):
    """
    Base class for hooks in the agent template.
    """

    @abstractmethod
    def hook(self, session: Session) -> Any:
        """
        The hook method that is called during the execution of the template.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session or any other value depending on the hook implementation.
        """
        raise NotImplementedError("hook method is not implemented")


class TransformHook(Hook):
    """
    A hook that transforms the session of the conversation.
    """

    def __init__(self, function: Optional[Callable[[Session], Session]] = None):
        self.function = function

    def hook(self, session: Session) -> Session:
        """
        Transforms the session of the conversation using the provided function.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session.
        """
        if self.function is None:
            raise ValueError(
                "function is not set. TransformHook can be used two ways. 1. Set function in the constructor. 2. Inherit TransformHook and override hook method."
            )
        return self.function(session)


class BooleanHook(Hook):
    """
    A hook that evaluates a boolean condition.
    """

    def __init__(self, condition: Callable[[Session], bool]):
        self.condition = condition

    def hook(self, session: Session) -> bool:
        """
        Evaluates the boolean condition using the current session of the conversation.

        Args:
            session: The current session of the conversation.

        Returns:
            The result of the boolean condition evaluation.
        """
        return self.condition(session)


class AskUserHook(TransformHook):
    """
    A hook that asks the user for input and stores the result in the state.
    """

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
        """
        Asks the user for input and stores the result in the session metadata.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session.
        """
        # show user a prompt on console
        raw = input(self.description).strip()
        if raw == "" and self.default is not None:
            raw = self.default
        metadata = session.get_latest_metadata()
        metadata[self.key] = raw
        return session


class GenerateChatHook(TransformHook):
    """
    A hook that generates a chat message using the LLM model and stores the result in the session metadata.
    """

    def __init__(
        self,
        key: str,
    ):
        self.key = key

    def hook(self, session: Session) -> Session:
        """
        Generates a chat message using the LLM model and stores the result in the session metadata.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session.
        """
        if session.runner is None:
            raise ValueError(
                "Runner must be given to use GenerateChatHook. Please set runner to the session."
            )
        message = session.runner.models.send(session.runner.parameters, session)
        metadata = session.get_latest_metadata()
        metadata[self.key] = message.content
        return session


class CountUpHook(TransformHook):
    """
    A hook that counts up a value in the session metadata.
    """

    def __init__(self):
        pass  # No configuration is needed here.

    def hook(self, session: Session) -> Session:
        """
        Counts up a value in the session metadata.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session.
        """
        template_id = session.get_current_template_id()
        if template_id is None:
            raise ValueError("template_id is not set")
        metadata = session.get_latest_metadata()
        if template_id not in metadata:
            metadata[template_id] = 0
        else:
            metadata[template_id] += 1
        return session


class DebugHook(TransformHook):
    """
    A hook that prints debug information during the execution of the template.
    """

    def __init__(self, message_shown_when_called: str):
        self.message = message_shown_when_called

    def hook(self, session: Session) -> Session:
        """
        Prints debug information during the execution of the template.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session.
        """
        print(self.message + " template_id: " + str(session.get_current_template_id()))
        print(self.message + " metadata: " + str(session.get_latest_metadata()))
        return session


class ResetDataHook(TransformHook):
    """
    A hook that resets the metadata in the session.
    """

    def __init__(self):
        pass  # No configuration is needed here.

    def hook(self, session: Session) -> Session:
        """
        Resets the metadata in the session.

        Args:
            session: The current session of the conversation.

        Returns:
            The modified session.
        """
        metadata = session.get_latest_metadata()
        metadata.clear()
        return session
