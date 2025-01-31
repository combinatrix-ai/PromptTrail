import os
import unittest

from pydantic import ValidationError

from examples.agent import weather_forecast
from prompttrail.core import Message, Session
from prompttrail.core.cache import LRUCacheProvider
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.mocks import EchoMockProvider
from prompttrail.models.openai import OpenAIConfig, OpenAIModel
from tests.models.test_utils import (
    run_basic_message_test,
    run_malformed_sessions_test,
    run_system_message_test,
)


class TestOpenAI(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.use_model = "gpt-4o-mini"
        self.config = OpenAIConfig(
            api_key=self.api_key,
            model_name=self.use_model,
            temperature=0.0,
            max_tokens=100,
        )
        self.model = OpenAIModel(configuration=self.config)

    def test_model_list(self):
        model_list = self.model.list_models()
        self.assertIsInstance(model_list, list)
        self.assertIsInstance(model_list[0], str)
        self.assertIn(self.use_model, model_list)

    def test_model_send(self):
        # Basic message handling
        run_basic_message_test(self.model, self.config, "527")

        # System message handling
        run_system_message_test(
            self.model,
            self.config,
            "27",
            user_message="Calculate 14+13",
        )

        # Test malformed sessions
        run_malformed_sessions_test(self.model, self.config, supports_tool_result=True)

    def test_streaming(self):
        # One message
        message = Message(
            content="This is automated test API call. Please answer the calculation 17*31.",
            role="user",
        )
        session = Session(messages=[message])
        response = self.model.send_async(session)
        messages = list(response)
        self.assertTrue(
            all([isinstance(m, Message) for m in messages]) and len(messages) > 0
        )
        concat = "".join([m.content for m in messages])
        role = messages[0].role
        self.assertIn("527", concat)
        self.assertEqual(role, "assistant")

    def test_function_calling(self):
        # Tools are already tested in test_tool.py
        # Here we use the example from examples/agent/weather_forecast.py

        session = weather_forecast.runner.run(max_messages=10)
        messages = session.messages
        messages = [m for m in messages if m.role != CONTROL_TEMPLATE_ROLE]
        roles = [m.role for m in messages]
        # system user assistant tool_result assistant
        self.assertEqual(
            roles, ["system", "user", "assistant", "tool_result", "assistant"]
        )
        self.assertIn("Tokyo", messages[-1].content)

    def test_use_both_cache_and_mock(self):
        with self.assertRaises(ValidationError):
            OpenAIConfig(
                api_key="sk-xxx",
                mock_provider=EchoMockProvider(role="assistant"),
                cache_provider=LRUCacheProvider(),
            )


if __name__ == "__main__":
    unittest.main()
