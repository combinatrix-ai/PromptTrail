import unittest

from prompttrail.core import Message, Session
from prompttrail.core.mocks import OneTurnConversationMockProvider
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


class TestOneTurnConversationMockProvider(unittest.TestCase):
    def setUp(self):
        self.conversation_table = {
            "Hello": Message(content="Hi", role="assistant"),
            "How are you?": Message(content="I'm fine, thank you.", role="user"),
        }
        self.mock_provider = OneTurnConversationMockProvider(self.conversation_table)

    def test_call_with_known_message(self):
        session = Session(messages=[Message(content="Hello", role="user")])
        response = self.mock_provider.call(session)
        self.assertEqual(response.content, "Hi")
        self.assertEqual(response.role, "assistant")

    def test_call_with_unknown_message(self):
        session = Session(messages=[Message(content="Unknown message", role="user")])
        with self.assertRaises(ValueError):
            self.mock_provider.call(session)


class TestOpenAIChatCompletionModelMock(unittest.TestCase):
    def setUp(self):
        self.conversation_table = {
            "Hello": Message(content="Hi", role="assistant"),
            "How are you?": Message(content="I'm fine, thank you.", role="user"),
        }
        self.mock_provider = OneTurnConversationMockProvider(self.conversation_table)
        config = OpenAIConfig(
            api_key="dummy",
            model_name="gpt-4o-mini",
            max_tokens=1024,
            mock_provider=self.mock_provider,
        )
        self.models = OpenAIModel(configuration=config)

    def test_send_with_known_message(self):
        session = Session(messages=[Message(content="Hello", role="user")])
        response = self.models.send(session=session)
        self.assertEqual(response.content, "Hi")
        self.assertEqual(response.role, "assistant")

    def test_send_with_unknown_message(self):
        session = Session(
            messages=[Message(content="Unknown message", role="assistant")]
        )
        with self.assertRaises(ValueError):
            self.models.send(session=session)

    def test_send_async_with_known_message_yield_all(self):
        session = Session(messages=[Message(content="Hello", role="user")])
        message_generator = self.models.send_async(session=session, yield_type="all")
        messages = list(message_generator)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "H")
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[1].content, "Hi")
        self.assertEqual(messages[1].role, "assistant")

    def test_send_async_with_known_message_yield_new(self):
        session = Session(messages=[Message(content="Hello", role="user")])
        message_generator = self.models.send_async(session=session, yield_type="new")
        messages = list(message_generator)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "H")
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[1].content, "i")
        self.assertEqual(messages[1].role, "assistant")

    def test_send_async_with_unknown_message(self):
        session = Session(messages=[Message(content="Unknown message", role="user")])
        with self.assertRaises(ValueError):
            message_generator = self.models.send_async(
                session=session, yield_type="all"
            )
            # We need to call the generator to raise the error.
            list(message_generator)


if __name__ == "__main__":
    unittest.main()
