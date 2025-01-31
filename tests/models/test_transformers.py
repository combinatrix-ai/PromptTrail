from unittest.mock import MagicMock

import pytest

from prompttrail.core import Message, Session
from prompttrail.models.transformers import TransformersConfig, TransformersModel
from tests.models.test_utils import (
    run_basic_message_test,
    run_malformed_sessions_test,
    run_system_message_test,
)


@pytest.fixture
def mock_model():
    config = TransformersConfig(
        device="cpu", model_name="mock-model", temperature=0.0, max_tokens=100
    )
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "mock response"
    model.generate.return_value = [MagicMock()]
    return TransformersModel(
        configuration=config,
        model=model,
        tokenizer=tokenizer,
    )


def test_validate_session(mock_model):
    # Test malformed sessions
    run_malformed_sessions_test(
        mock_model, mock_model.configuration, supports_tool_result=False
    )


def test_basic_message(mock_model):
    # Basic message handling
    run_basic_message_test(mock_model, mock_model.configuration, "mock response")


def test_system_message(mock_model):
    # System message handling
    run_system_message_test(
        mock_model,
        mock_model.configuration,
        "mock response",
        user_message="Calculate 14+13",
    )


def test_session_to_text(mock_model):
    # Test session to text conversion
    session = Session(
        messages=[
            Message(content="You're a helpful assistant.", role="system"),
            Message(content="Hello", role="user"),
            Message(content="Hi!", role="assistant"),
        ]
    )
    text = mock_model._session_to_text(session)
    assert "system: You're a helpful assistant." in text
    assert "user: Hello" in text
    assert "assistant: Hi!" in text
