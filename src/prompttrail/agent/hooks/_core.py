import logging
from abc import abstractmethod
from typing import Any, Callable, Optional

from prompttrail.agent import State

logger = logging.getLogger(__name__)


class Hook(object):
    """
    Base class for hooks in the agent template.
    """

    @abstractmethod
    def hook(self, state: State) -> Any:
        """
        The hook method that is called during the execution of the template.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state or any other value depending on the hook implementation.
        """
        raise NotImplementedError("hook method is not implemented")


class TransformHook(Hook):
    """
    A hook that transforms the state of the conversation.
    """

    def __init__(self, function: Callable[[State], State]):
        self.function = function

    def hook(self, state: State) -> State:
        """
        Transforms the state of the conversation using the provided function.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state.
        """
        return self.function(state)


class BooleanHook(Hook):
    """
    A hook that evaluates a boolean condition.
    """

    def __init__(self, condition: Callable[[State], bool]):
        self.condition = condition

    def hook(self, state: State) -> bool:
        """
        Evaluates the boolean condition using the current state of the conversation.

        Args:
            state: The current state of the conversation.

        Returns:
            The result of the boolean condition evaluation.
        """
        return self.condition(state)


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

    def hook(self, state: State) -> State:
        """
        Asks the user for input and stores the result in the state.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state.
        """
        # show user a prompt on console
        raw = input(self.description).strip()
        if raw == "" and self.default is not None:
            raw = self.default
        state.data[self.key] = raw
        return state


class GenerateChatHook(TransformHook):
    """
    A hook that generates a chat message using the LLM model and stores the result in the state.
    """

    def __init__(
        self,
        key: str,
    ):
        self.key = key

    def hook(self, state: State) -> State:
        """
        Generates a chat message using the LLM model and stores the result in the state.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state.
        """
        if state.runner is None:
            raise ValueError(
                "Runner must be given to use GenerateChatHook. Please set runner to the state."
            )
        message = state.runner.models.send(
            state.runner.parameters, state.session_history
        )
        state.data[self.key] = message.content
        return state


class CountUpHook(TransformHook):
    """
    A hook that counts up a value in the state.
    """

    def __init__(self):
        pass  # No configuration is needed here.

    def hook(self, state: State) -> State:
        """
        Counts up a value in the state.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state.
        """
        template_id = state.get_current_template_id()
        if template_id is None:
            raise ValueError("template_id is not set")
        if template_id not in state.data:
            state.data[template_id] = 0
        else:
            state.data[template_id] += 1
        return state


class DebugHook(TransformHook):
    """
    A hook that prints debug information during the execution of the template.
    """

    def __init__(self, message_shown_when_called: str):
        self.message = message_shown_when_called

    def hook(self, state: State) -> State:
        """
        Prints debug information during the execution of the template.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state.
        """
        print(self.message + " template_id: " + str(state.get_current_template_id()))
        print(self.message + " data: " + str(state.data))
        return state


class ResetDataHook(TransformHook):
    """
    A hook that resets the data in the state.
    """

    def __init__(self):
        pass  # No configuration is needed here.

    def hook(self, state: State) -> State:
        """
        Resets the data in the state.

        Args:
            state: The current state of the conversation.

        Returns:
            The modified state.
        """
        state.data = {}
        return state
