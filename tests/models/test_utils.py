import pytest

from prompttrail.core import Config, Message, Model, Session
from prompttrail.core.errors import ParameterValidationError


def run_basic_message_test(
    model: Model,
    config: Config,
    expected_response: str = "527",
    message_content: str = "This is automated test API call. Please answer the calculation 17*31.",
):
    """単一のユーザーメッセージの基本的なハンドリングをテストします。"""
    message = Message(content=message_content, role="user")
    session = Session(messages=[message])
    response = model.send(session)
    assert isinstance(response, Message)
    assert isinstance(response.content, str)
    assert expected_response in response.content
    assert response.role == "assistant"


def run_system_message_test(
    model: Model,
    config: Config,
    expected_response: str = "27",
    system_message: str = "You're a helpful assistant.",
    user_message: str = "Calculate 14+13",
):
    """システムメッセージのハンドリングをテストします。"""
    messages = [
        Message(content=system_message, role="system"),
        Message(content=user_message, role="user"),
    ]
    session = Session(messages=messages)
    response = model.send(session)
    assert isinstance(response, Message)
    assert isinstance(response.content, str)
    assert expected_response in response.content
    assert response.role == "assistant"
    expected_roles = ["system", "user", "assistant"]
    for i, message in enumerate(session.messages):
        assert message.role == expected_roles[i]


def run_malformed_sessions_test(
    model: Model,
    config: Config,
    supports_tool_result: bool = False,
):
    """不正な形式のセッションのハンドリングをテストします。"""
    # 空のセッションのテスト
    with pytest.raises(ParameterValidationError):
        model.send(Session(messages=[]))

    # 複数のシステムメッセージのテスト
    with pytest.raises(ParameterValidationError):
        model.send(
            Session(
                messages=[
                    Message(content="a", role="system"),
                    Message(content="a", role="system"),
                ]
            ),
        )

    # システムメッセージが最初でない場合のテスト
    with pytest.raises(ParameterValidationError):
        model.send(
            Session(
                messages=[
                    Message(content="Hello", role="user"),
                    Message(content="Hi!", role="assistant"),
                    Message(content="a", role="system"),
                ]
            ),
        )
