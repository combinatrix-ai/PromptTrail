import os
import unittest

from prompttrail.core import Message, Session
from prompttrail.core.errors import ParameterValidationError
from prompttrail.models.google_cloud import (
    GoogleCloudChatModel,
    GoogleCloudChatModelConfiguration,
    GoogleCloudChatModelParameters,
)


class TestGoogleCloud(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
        self.configuration = GoogleCloudChatModelConfiguration(
            api_key=self.api_key,
        )
        self.use_model = "models/gemini-1.5-flash"
        self.parameters = GoogleCloudChatModelParameters(
            model_name=self.use_model,
            max_tokens=100,
            temperature=0,
        )
        self.models = GoogleCloudChatModel(configuration=self.configuration)

    def test_model_list(self):
        model_list = self.models.list_models()
        self.assertIsInstance(model_list, list)
        self.assertIsInstance(model_list[0], str)
        self.assertIn(self.use_model, model_list)

    def test_model_send(self):
        # One message
        message = Message(
            content="This is automated test API call. Please answer the calculation 17*31.",
            role="user",
        )
        session = Session(messages=[message])
        response = self.models.send(self.parameters, session)
        self.assertIsInstance(response, Message)
        self.assertIsInstance(response.content, str)
        self.assertIn("527", response.content)

        # All message types
        messages = [
            Message(
                content="You're a helpful assistant. When you see a response starting with 'bc:', it means it's a calculation result from a basic calculator tool. Please acknowledge and use that result.",
                role="system",
            ),
            Message(content="Calculate 129183712*1271606", role="user"),
            Message(content="bc: The answer is 12696595579352", role="system"),
        ]
        session = Session(messages=messages)
        response = self.models.send(self.parameters, session)
        self.assertIsInstance(response, Message)
        self.assertIsInstance(response.content, str)
        # TODO: Change problem. Sometimes the response has a comma, sometimes not.
        response_without_comma = response.content.replace(",", "")
        self.assertIn("12696595579352", response_without_comma)

        # malformed session
        with self.assertRaises(ParameterValidationError):
            response = self.models.send(
                self.parameters,
                Session(messages=[Message(content="", role="user")]),  # empty message
            )
        with self.assertRaises(ParameterValidationError):
            self.models.send(
                self.parameters,
                Session(messages=[Message(content="Hello", role=None)]),  # empty role
            )
        with self.assertRaises(ParameterValidationError):
            self.models.send(self.parameters, Session(messages=[]))  # empty session

        # On Google Cloud, role can be anything
        response = self.models.send(
            self.parameters,
            Session(messages=[Message(content="Hello", role="")]),
        )


if __name__ == "__main__":
    unittest.main()
