import os
import sys
import unittest

from pydantic import ValidationError

from prompttrail.core import Message, Session
from prompttrail.core.cache import LRUCacheProvider
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.mocks import EchoMockProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(path)
from examples.agent import weather_forecast  # type: ignore # noqa: E402


# TODO: Add error handling test
class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.organization_id = os.environ.get("OPENAI_ORGANIZATION_ID", None)
        self.use_model = "gpt-3.5-turbo"
        self.config = OpenAIModelConfiguration(
            api_key=self.api_key, organization_id=self.organization_id
        )
        self.parameters = OpenAIModelParameters(
            model_name=self.use_model, max_tokens=100, temperature=0
        )
        self.model = OpenAIChatCompletionModel(configuration=self.config)

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
            Message(content="Calculate 129183712*1271606", sender="user"),
            Message(content="bc: The answer is 12696595579352", sender="system"),
        ]
        session = Session(messages=messages)
        response = self.model.send(self.parameters, session)
        self.assertIsInstance(response, Message)
        self.assertIsInstance(response.content, str)
        self.assertIn("12696595579352", response.content)

        # malformed session
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

    def test_streaming(self):
        # One message
        message = Message(
            content="This is automated test API call. Please answer the calculation 17*31.",
            sender="user",
        )
        session = Session(messages=[message])
        response = self.model.send_async(self.parameters, session)
        messages = list(response)
        self.assertTrue(
            all([isinstance(m, Message) for m in messages]) and len(messages) > 0
        )
        concat = "".join([m.content for m in messages])
        sender = messages[0].sender
        self.assertIn("527", concat)
        self.assertEqual(sender, "assistant")

    def test_function_calling(self):
        # Tools are already tested in test_tool.py
        # Here, we use the example from examples/agent/weather_forecast.py

        state = weather_forecast.runner.run(max_messages=10)
        messages = state.session_history.messages
        messages = [m for m in messages if m.sender != CONTROL_TEMPLATE_ROLE]
        senders = [m.sender for m in messages]
        # system, user, function call by assistant, function result by function, assistant
        self.assertEqual(
            senders, ["system", "user", "assistant", "function", "assistant"]
        )
        self.assertIn("Tokyo", messages[-1].content)

    def test_use_both_cache_and_mock(self):
        with self.assertRaises(ValidationError):
            OpenAIModelConfiguration(
                api_key="sk-xxx",
                mock_provider=EchoMockProvider(sender="assistant"),
                cache_provider=LRUCacheProvider(),
            )


if __name__ == "__main__":
    unittest.main()
