import pytest

from prompttrail.core import Message, Model, Parameters, Session
from prompttrail.core.errors import ParameterValidationError


def run_basic_message_test(
    model: Model,
    parameters: Parameters,
    expected_response: str = "527",
    message_content: str = "This is automated test API call. Please answer the calculation 17*31.",
):
    """Test basic message handling with a single user message."""
    message = Message(content=message_content, role="user")
    session = Session(messages=[message])
    response = model.send(parameters, session)
    assert isinstance(response, Message)
    assert isinstance(response.content, str)
    assert expected_response in response.content
    assert response.role == "assistant"


def run_system_message_test(
    model: Model,
    parameters: Parameters,
    expected_response: str = "27",
    system_message: str = "You're a helpful assistant.",
    user_message: str = "Calculate 14+13",
):
    """Test handling of system messages."""
    messages = [
        Message(content=system_message, role="system"),
        Message(content=user_message, role="user"),
    ]
    session = Session(messages=messages)
    response = model.send(parameters, session)
    assert isinstance(response, Message)
    assert isinstance(response.content, str)
    assert expected_response in response.content
    assert response.role == "assistant"
    expected_roles = ["system", "user", "assistant"]
    for i, message in enumerate(session.messages):
        assert message.role == expected_roles[i]


def run_malformed_sessions_test(
    model: Model,
    parameters: Parameters,
    supports_tool_result: bool = False,
):
    """Test handling of malformed sessions."""
    # Test empty session
    with pytest.raises(ParameterValidationError):
        model.send(parameters, Session(messages=[]))

    # Test multiple system messages
    with pytest.raises(ParameterValidationError):
        model.send(
            parameters,
            Session(
                messages=[
                    Message(content="a", role="system"),
                    Message(content="a", role="system"),
                ]
            ),
        )

    # Test system message not first
    with pytest.raises(ParameterValidationError):
        model.send(
            parameters,
            Session(
                messages=[
                    Message(content="Hello", role="user"),
                    Message(content="Hi!", role="assistant"),
                    Message(content="a", role="system"),
                ]
            ),
        )
