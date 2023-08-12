import unittest

from prompttrail.core import Configuration, Message, Model, Session


class TestCore(unittest.TestCase):
    def test_text_message_creation(self):
        message = Message(content="Hello", sender="User")
        self.assertEqual(message.content, "Hello")
        self.assertEqual(message.sender, "User")

    def test_text_session_creation(self):
        message1 = Message(content="Hello", sender="User")
        message2 = Message(content="Hi", sender="Bot")
        session = Session(messages=[message1, message2])
        self.assertEqual(len(session.messages), 2)

    def test_model_implementation(self):
        with self.assertRaises(TypeError):
            _ = Model(configuration=Configuration())


if __name__ == "__main__":
    unittest.main()
