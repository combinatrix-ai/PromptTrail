import unittest
from typing import List

import pytest
from pydantic import ValidationError

from prompttrail.core import Message, Metadata, Session


class TestCore(unittest.TestCase):
    def test_text_message_creation(self):
        """Test text message creation."""
        message = Message(content="test", role="user")
        self.assertEqual(message.content, "test")
        self.assertEqual(message.role, "user")

        message = Message(content="test", role="assistant", metadata={"test": "value"})
        self.assertEqual(message.metadata["test"], "value")
        message = Message(
            content="test", role="assistant", metadata=Metadata({"test": "value"})
        )
        self.assertEqual(message.metadata["test"], "value")

    def test_text_session_creation(self) -> None:
        """Test text session creation."""
        messages: List[Message] = [
            Message(content="test", role="user"),
            Message(content="test", role="assistant"),
        ]
        session = Session(messages=messages)
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.messages[0].content, "test")
        self.assertEqual(session.messages[0].role, "user")
        self.assertEqual(session.messages[1].content, "test")
        self.assertEqual(session.messages[1].role, "assistant")

    def test_message_validation(self):
        """Test message validation."""
        # Test role is None
        with pytest.raises(ValidationError):
            Message(content="test", role=None)  # type: ignore

        # Test role is not in predefined roles
        with pytest.raises(ValidationError):
            Message(content="test", role="invalid_role")  # type: ignore

        # Test content is None
        with pytest.raises(ValidationError):
            Message(content=None, role="user")  # type: ignore

        # Test content is not string
        with pytest.raises(ValidationError):
            Message(content=123, role="user")  # type: ignore

        # Test empty content
        message = Message(content="", role="user")
        self.assertEqual(message.content, "")
        self.assertEqual(message.role, "user")

        # Test valid message creation with different roles
        roles = ["system", "user", "assistant", "tool_result", "control"]
        for role in roles:
            message = Message(content="test", role=role)  # type: ignore
            self.assertEqual(message.content, "test")
            self.assertEqual(message.role, role)

    def test_session_validation(self):
        """Test session validation."""
        # Test empty messages list
        session = Session(messages=[])
        self.assertEqual(len(session.messages), 0)

        # Test message append
        message = Message(content="test", role="user")
        session.append(message)
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0], message)

        # Test get_last with empty session
        empty_session = Session(messages=[])
        with pytest.raises(IndexError):
            empty_session.get_last()

        # Test get_last with non-empty session
        self.assertEqual(session.get_last(), message)

        # Test get_latest_metadata with empty session and empty messages
        empty_session = Session(messages=[], metadata={"test": "value"})
        self.assertEqual(empty_session.metadata, {"test": "value"})

        # Test get_latest_metadata with non-empty session
        message_with_metadata = Message(
            content="test", role="user", metadata={"test": "new_value"}
        )
        session = Session(messages=[message_with_metadata], metadata={"test": "value"})
        self.assertEqual(session.metadata, {"test": "value"})

        # Test get_latest_metadata with empty session metadata and message metadata
        message_with_metadata = Message(
            content="test", role="user", metadata={"test": "new_value"}
        )
        session = Session(messages=[message_with_metadata])
        self.assertEqual(session.metadata, {})

        # Test basic stack operations (Real stack operations are tested in tests for ControlTemplate)
        with pytest.raises(IndexError):
            session.pop_stack()
        with pytest.raises(IndexError):
            session.head_stack()

        # Test jump operations
        self.assertIsNone(session.get_jump())
        session.set_jump("test_id")
        self.assertEqual(session.get_jump(), "test_id")


if __name__ == "__main__":
    unittest.main()
