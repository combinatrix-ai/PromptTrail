import os
import unittest

from prompttrail.core import Message, Session
from prompttrail.core.errors import ParameterValidationError
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel
from tests.models.test_utils import (
    run_basic_message_test,
    run_malformed_sessions_test,
    run_system_message_test,
)


class TestAnthropic(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["ANTHROPIC_API_KEY"]
        self.use_model = "claude-3-5-haiku-latest"
        self.config = AnthropicConfig(
            api_key=self.api_key,
            model_name=self.use_model,
            max_tokens=100,
            temperature=0,
        )
        self.model = AnthropicModel(configuration=self.config)

    def test_model_send(self):
        # Basic message handling
        run_basic_message_test(self.model, self.config)

        # System message handling
        run_system_message_test(
            self.model,
            self.config,
            "27",
            user_message="Calculate 14+13",
        )

        # Test malformed sessions
        run_malformed_sessions_test(self.model, self.config, supports_tool_result=True)

        # Test empty messages (Anthropic-specific)
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                Session(messages=[Message(content="", role="user")]),  # empty message
            )


if __name__ == "__main__":
    unittest.main()
