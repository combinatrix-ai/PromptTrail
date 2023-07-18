import os
import sys
import unittest

from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))
from src.prompttrail.core import Configuration, Message, Model, TextMessage, TextSession


class TestCore(unittest.TestCase):
    def test_text_message_creation(self):
        message = TextMessage(content="Hello", sender="User")
        self.assertEqual(message.content, "Hello")
        self.assertEqual(message.sender, "User")

    def test_text_session_creation(self):
        message1 = TextMessage(content="Hello", sender="User")
        message2 = TextMessage(content="Hi", sender="Bot")
        session = TextSession(messages=[message1, message2])
        self.assertEqual(len(session.messages), 2)

        with self.assertRaises(ValidationError):
            message3 = Message(content="Bye", sender="User")
            session = TextSession(messages=[message1, message2, message3])  # type: ignore

    def test_model_implementation(self):
        with self.assertRaises(TypeError):
            _ = Model(configuration=Configuration())


if __name__ == "__main__":
    unittest.main()
