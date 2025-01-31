import os
import unittest

from prompttrail.models.google import GoogleConfig, GoogleModel
from tests.models.test_utils import (
    run_basic_message_test,
    run_malformed_sessions_test,
    run_system_message_test,
)


class TestGoogleCloud(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
        self.config = GoogleConfig(
            api_key=self.api_key,
            model_name="models/gemini-1.5-flash",
            max_tokens=100,
            temperature=0,
        )
        self.model = GoogleModel(configuration=self.config)

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
        run_malformed_sessions_test(self.model, self.config, supports_tool_result=False)


if __name__ == "__main__":
    unittest.main()
