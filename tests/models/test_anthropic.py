import os
import sys
import unittest

from prompttrail.core import Message, Session
from prompttrail.core.errors import ParameterValidationError
from prompttrail.models.anthropic import (
    AnthropicClaudeModel,
    AnthropicClaudeModelConfiguration,
    AnthropicClaudeModelParameters,
)

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
from examples.agent import weather_forecast  # type: ignore # noqa: E402


# TODO: Add error handling test
class TestAnthropic(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.config = AnthropicClaudeModelConfiguration(api_key=self.api_key)
        self.use_model = "claude-3-haiku-20240307"
        self.parameters = AnthropicClaudeModelParameters(
            model_name=self.use_model, max_tokens=100, temperature=0
        )
        self.model = AnthropicClaudeModel(configuration=self.config)

    def test_model_list(self):
        model_list = self.model.list_models()
        self.assertIsInstance(model_list, list)
        self.assertIsInstance(model_list[0], str)
        self.assertIn(self.use_model, model_list)

    def test_model_send(self):
        # One message
        message = Message(
            content="This is automated test API call. Please answer the calculation 17*31.",
            sender="user",
        )
        session = Session(messages=[message])
        response = self.model.send(self.parameters, session)
        self.assertIsInstance(response, Message)
        self.assertIsInstance(response.content, str)
        self.assertIn("527", response.content)

        # All message types
        messages = [
            Message(content="You're a helpful assistant.", sender="system"),
            Message(
                content="Calculate 14+13", sender="user"
            ),  # Haiku won't answer difficult questions if the system message is present?
        ]
        session = Session(messages=messages)
        response = self.model.send(self.parameters, session)
        self.assertIsInstance(response, Message)
        self.assertIsInstance(response.content, str)
        self.assertIn("27", response.content)

        # malformed session
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                Session(
                    messages=[
                        Message(content="a", sender="system"),
                        Message(content="a", sender="system"),
                    ]
                ),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                Session(messages=[Message(content="", sender="User")]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                Session(messages=[Message(content="Hello", sender="User")]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                Session(messages=[Message(content="Hello", sender=None)]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(self.parameters, Session(messages=[]))


if __name__ == "__main__":
    unittest.main()
