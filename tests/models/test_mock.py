import unittest

from prompttrail.core import Message, Session
from prompttrail.core.mocks import OneTurnConversationMockProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)


class TestOneTurnConversationMockProvider(unittest.TestCase):
    def setUp(self):
        self.first_sender = "user"
        self.second_sender = "assistant"
        self.conversation_table = {
            "Hello": Message(content="Hi", sender=self.second_sender),
            "How are you?": Message(
                content="I'm fine, thank you.", sender=self.second_sender
            ),
        }
        self.mock_provider = OneTurnConversationMockProvider(
            self.conversation_table, self.second_sender
        )

    def test_call_with_known_message(self):
        session = Session(messages=[Message(content="Hello", sender=self.first_sender)])
        response = self.mock_provider.call(session)
        self.assertEqual(response.content, "Hi")

    def test_call_with_unknown_message(self):
        session = Session(
            messages=[Message(content="Unknown message", sender=self.first_sender)]
        )
        with self.assertRaises(ValueError):
            self.mock_provider.call(session)


class TestOpenAIChatCompletionModelMock(unittest.TestCase):
    def setUp(self):
        self.first_sender = "user"
        self.second_sender = "assistant"
        self.conversation_table = {
            "Hello": Message(content="Hi", sender=self.second_sender),
            "How are you?": Message(
                content="I'm fine, thank you.", sender=self.second_sender
            ),
        }
        self.mock_provider = OneTurnConversationMockProvider(
            self.conversation_table, self.second_sender
        )
        self.models = OpenAIChatCompletionModel(
            configuration=OpenAIModelConfiguration(
                api_key="", mock_provider=self.mock_provider
            ),
        )
        self.parameters = OpenAIModelParameters(
            model_name="",
            max_tokens=1024,
        )

    def test_send_with_known_message(self):
        session = Session(messages=[Message(content="Hello", sender=self.first_sender)])
        response = self.models.send(parameters=self.parameters, session=session)
        self.assertEqual(response.content, "Hi")
        self.assertEqual(response.sender, self.second_sender)

    def test_send_with_unknown_message(self):
        session = Session(
            messages=[Message(content="Unknown message", sender=self.first_sender)]
        )
        with self.assertRaises(ValueError):
            self.models.send(parameters=self.parameters, session=session)

    def test_send_async_with_known_message_yield_all(self):
        session = Session(messages=[Message(content="Hello", sender=self.first_sender)])
        message_generator = self.models.send_async(
            parameters=self.parameters, session=session, yield_type="all"
        )
        messages = list(message_generator)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "H")
        self.assertEqual(messages[0].sender, self.second_sender)
        self.assertEqual(messages[1].content, "Hi")
        self.assertEqual(messages[1].sender, self.second_sender)

    def test_send_async_with_known_message_yield_new(self):
        session = Session(messages=[Message(content="Hello", sender=self.first_sender)])
        message_generator = self.models.send_async(
            parameters=self.parameters, session=session, yield_type="new"
        )
        messages = list(message_generator)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "H")
        self.assertEqual(messages[0].sender, self.second_sender)
        self.assertEqual(messages[1].content, "i")
        self.assertEqual(messages[1].sender, self.second_sender)

    def test_send_async_with_unknown_message(self):
        session = Session(
            messages=[Message(content="Unknown message", sender=self.first_sender)]
        )
        with self.assertRaises(ValueError):
            message_generator = self.models.send_async(
                parameters=self.parameters, session=session, yield_type="all"
            )
            # We need to call the generator to raise the error.
            list(message_generator)


if __name__ == "__main__":
    unittest.main()
