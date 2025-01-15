import unittest

from prompttrail.core import Configuration, Message, Model, Session


class TestCore(unittest.TestCase):
    def test_text_message_creation(self):
        message = Message(content="Hello", role="User")
        self.assertEqual(message.content, "Hello")
        self.assertEqual(message.role, "User")

    def test_text_session_creation(self):
        message1 = Message(content="Hello", role="User")
        message2 = Message(content="Hi", role="Bot")
        session = Session(messages=[message1, message2])
        self.assertEqual(len(session.messages), 2)

    def test_model_implementation(self):
        with self.assertRaises(TypeError):
            _ = Model(configuration=Configuration())


if __name__ == "__main__":
    unittest.main()
