import unittest

from prompttrail.core import TextMessage, TextSession
from prompttrail.mock import OneTurnConversationMockProvider
from prompttrail.provider.openai import (
    OpenAIChatCompletionModelMock,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)


class TestOneTurnConversationMockProvider(unittest.TestCase):
    def setUp(self):
        self.conversation_table = {
            "Hello": "Hi",
            "How are you?": "I'm fine, thank you.",
        }
        self.sender = "user"
        self.mock_provider = OneTurnConversationMockProvider(
            self.conversation_table, self.sender
        )

    def test_call_with_known_message(self):
        session = TextSession(
            messages=[TextMessage(content="Hello", sender=self.sender)]
        )
        response = self.mock_provider.call(session)
        self.assertEqual(response.content, "Hi")

    def test_call_with_unknown_message(self):
        session = TextSession(
            messages=[TextMessage(content="Unknown message", sender=self.sender)]
        )
        with self.assertRaises(ValueError):
            self.mock_provider.call(session)


class TestOpenAIChatCompletionModelMock(unittest.TestCase):
    def setUp(self):
        self.conversation_table = {
            "Hello": "Hi",
            "How are you?": "I'm fine, thank you.",
        }
        self.sender = "user"
        self.mock_provider = OneTurnConversationMockProvider(
            self.conversation_table, self.sender
        )
        self.model = OpenAIChatCompletionModelMock(
            configuration=OpenAIModelConfiguration(
                api_key="",
            ),
            mock_provider=self.mock_provider,
        )
        self.parameters = OpenAIModelParameters(
            model_name="",
            max_tokens=1024,
        )

    def test_send_with_known_message(self):
        session = TextSession(
            messages=[TextMessage(content="Hello", sender=self.sender)]
        )
        response = self.model.send(parameters=self.parameters, session=session)
        self.assertEqual(response.content, "Hi")
        self.assertEqual(response.sender, self.sender)

    def test_send_with_unknown_message(self):
        session = TextSession(
            messages=[TextMessage(content="Unknown message", sender=self.sender)]
        )
        with self.assertRaises(ValueError):
            self.model.send(parameters=self.parameters, session=session)


if __name__ == "__main__":
    unittest.main()
