import os
import unittest

from prompttrail.core import TextMessage, TextSession
from prompttrail.error import ParameterValidationError
from prompttrail.provider.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)


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
        message = TextMessage(
            content="This is automated test API call. Please answer the calculation 17*31.",
            sender="user",
        )
        session = TextSession(messages=[message])
        response = self.model.send(self.parameters, session)
        self.assertIsInstance(response, TextMessage)
        self.assertIsInstance(response.content, str)
        self.assertIn("527", response.content)

        # All message types
        messages = [
            TextMessage(content="You're a helpful assistant.", sender="system"),
            TextMessage(content="Calculate 129183712*1271606", sender="user"),
            TextMessage(content="bc: The answer is 12696595579352", sender="system"),
        ]
        session = TextSession(messages=messages)
        response = self.model.send(self.parameters, session)
        self.assertIsInstance(response, TextMessage)
        self.assertIsInstance(response.content, str)
        self.assertIn("12696595579352", response.content)

        # malformed session
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                TextSession(messages=[TextMessage(content="", sender="User")]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                TextSession(messages=[TextMessage(content="Hello", sender="User")]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                TextSession(messages=[TextMessage(content="Hello", sender=None)]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(self.parameters, TextSession(messages=[]))


if __name__ == "__main__":
    unittest.main()
