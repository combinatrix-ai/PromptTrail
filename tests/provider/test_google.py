import os
import unittest

from prompttrail.core import Message, Session
from prompttrail.error import ParameterValidationError
from prompttrail.provider.google_cloud import (
    GoogleCloudChatModel,
    GoogleCloudChatParameters,
    GoogleCloudConfiguration,
)


class TestGoogleCloud(unittest.TestCase):
    def setUp(self):
        self.api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
        self.configuration = GoogleCloudConfiguration(
            api_key=self.api_key,
        )
        self.use_model = "models/chat-bison-001"
        self.parameters = GoogleCloudChatParameters(
            model_name=self.use_model,
            max_output_tokens=100,
            temperature=0,
        )
        self.model = GoogleCloudChatModel(configuration=self.configuration)

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
            response = self.model.send(
                self.parameters,
                Session(messages=[Message(content="", sender="user")]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(
                self.parameters,
                Session(messages=[Message(content="Hello", sender=None)]),
            )
        with self.assertRaises(ParameterValidationError):
            self.model.send(self.parameters, Session(messages=[]))

        # On Google Cloud, sender can be anything
        response = self.model.send(
            self.parameters,
            Session(messages=[Message(content="Hello", sender="")]),
        )


if __name__ == "__main__":
    unittest.main()
