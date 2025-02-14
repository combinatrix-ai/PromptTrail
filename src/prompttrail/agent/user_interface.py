import logging
from abc import abstractmethod
from typing import Dict, Optional

from prompt_toolkit import prompt
from prompt_toolkit.application import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor
from prompt_toolkit.styles import Style

from prompttrail.core import Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.utils import Debuggable

logger = logging.getLogger(__name__)


class UserInterface(Debuggable):
    @abstractmethod
    def ask(
        self,
        session: Session,
        instruction: Optional[str],
        default: Optional[str] = None,
    ) -> str:
        """
        Ask the user for input.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The user's input as a string.
        """
        raise NotImplementedError("ask method is not implemented")


class CLIInterface(UserInterface):
    def ask(
        self,
        session: Session,
        instruction: str | None = "Input> ",
        default: str | None = None,
    ) -> str:
        """
        Ask the user for input via the command line with an inline dimmed placeholder.

        - If a default is provided, it is shown dimly as placeholder text.
        - The placeholder appears only when the input is empty.
        - Once the user starts typing, the placeholder disappears.
        - Pressing Enter without typing any input returns the default value.

        Args:
            session: The current session.
            description: The prompt text shown to the user.
            default: The default value to show as a placeholder.

        Returns:
            The user's input as a string, or the default if nothing was entered.
        """
        if not default:
            # No placeholder, just a regular prompt.
            return prompt(instruction).strip()

        # Define a style for the placeholder text.
        style = Style.from_dict(
            {"placeholder": "fg:#888888"}  # Dim color for the placeholder
        )

        # Create an input processor that shows the placeholder only when the buffer is empty.
        placeholder_processor = ConditionalProcessor(
            processor=BeforeInput(default, style="class:placeholder"),
            filter=Condition(lambda: not get_app().current_buffer.text),
        )

        # Use the input processor in the prompt.
        user_input = prompt(
            instruction,
            style=style,
            input_processors=[placeholder_processor],
        ).strip()

        # Return the user input or the default if no input was provided.
        return user_input or default


class MockInterface(UserInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Base mock provider that returns an empty string.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            An empty string.
        """
        return ""


class SingleTurnResponseMockInterface(MockInterface):
    def __init__(self, conversation_table: Dict[str, str]):
        self.conversation_table = conversation_table

    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Mock the user interaction by providing pre-defined responses based on the conversation history.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The pre-defined response based on the conversation history.
        """
        valid_messages = [
            x for x in session.messages if x.role != CONTROL_TEMPLATE_ROLE
        ]
        last_message = valid_messages[-1].content
        if last_message not in self.conversation_table:
            raise ValueError("No conversation found for " + last_message)
        return self.conversation_table[last_message]


class EchoMockInterface(MockInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Mock the user interaction by echoing the last message.

        Args:
            session: The current session of the conversation.
            description: The description of the input prompt.
            default: The default value for the input prompt.

        Returns:
            The last message as the user's input.
        """
        return session.get_last().content


class DefaultOrEchoMockInterface(MockInterface):
    def ask(
        self,
        session: Session,
        description: Optional[str] = None,
        default: Optional[str] = None,
    ) -> str:
        prompt = description if description else "Enter your message"
        if default:
            prompt += f" (default: {default})"
        prompt += ": "

        user_input = input(prompt)
        if not user_input and default:
            return default
        return user_input
