import logging
from typing import Generator, Optional
from unittest.mock import Mock

import pytest

from prompttrail.agent.subroutine.session_init_strategy import FilteredInheritStrategy
from prompttrail.agent.subroutine.squash_strategy import (
    FilterByRoleStrategy,
    LLMFilteringStrategy,
    LLMSummarizingStrategy,
)
from prompttrail.agent.templates import Template
from prompttrail.agent.tools import SubroutineTool
from prompttrail.core import Message, Session


class SimpleTemplate(Template):
    """Simple template for testing that echoes input with prefix"""

    def __init__(self, template_id: Optional[str] = None):
        super().__init__(template_id=template_id)

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        # Debug log session state
        logging.debug(f"SimpleTemplate._render session messages: {session.messages}")

        # Find user message
        for msg in reversed(session.messages):
            if msg.role == "user":
                yield Message(role="assistant", content=f"Echo: {msg.content}")
                return session

        # Handle case where no user message is found
        yield Message(role="assistant", content="No input message found")
        return session

    def create_stack(self, session: Session):
        return super().create_stack(session)


def test_basic_execution():
    """Test basic subroutine execution"""
    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    tool = SubroutineTool(
        name="echo",
        description="Echo input with prefix",
        template=SimpleTemplate(),
    )

    result = tool.execute(input="Hello")
    assert result.content == "Echo: Hello"
    assert len(result.metadata["messages"]) == 1
    assert result.metadata["messages"][0].content == "Echo: Hello"


def test_with_system_message():
    """Test execution with system message"""
    tool = SubroutineTool(
        name="echo",
        description="Echo input with prefix",
        template=SimpleTemplate(),
    )

    result = tool.execute(
        input="Hello",
        system_message="You are a helpful assistant",
    )
    assert result.content == "Echo: Hello"

    # Verify system message was included
    messages = result.metadata["messages"]
    assert len(messages) == 1
    assert messages[0].content == "Echo: Hello"


def test_argument_validation():
    """Test argument validation"""
    tool = SubroutineTool(
        name="echo",
        description="Echo input with prefix",
        template=SimpleTemplate(),
    )

    # Test missing required argument
    with pytest.raises(Exception):
        tool.execute()

    # Test invalid argument type
    with pytest.raises(Exception):
        tool.execute(input=123)  # type: ignore

    # Test unknown argument
    with pytest.raises(Exception):
        tool.execute(input="Hello", unknown="value")  # type: ignore


def test_with_custom_strategies():
    """Test execution with custom session and squash strategies"""
    tool = SubroutineTool(
        name="echo",
        description="Echo input with prefix",
        template=SimpleTemplate(),
        session_init_strategy=FilteredInheritStrategy(
            lambda msg: msg.role in ["system", "user"]
        ),
        squash_strategy=FilterByRoleStrategy(roles=["assistant"]),
    )

    result = tool.execute(
        input="Hello",
        system_message="You are a helpful assistant",
    )
    assert result.content == "Echo: Hello"

    # Verify message handling
    messages = result.metadata["messages"]
    assert len(messages) == 1
    assert all(msg.role == "assistant" for msg in messages)


def test_multiple_executions():
    """Test multiple executions of the same tool"""
    tool = SubroutineTool(
        name="echo",
        description="Echo input with prefix",
        template=SimpleTemplate(),
    )

    # First execution
    result1 = tool.execute(input="Hello")
    assert result1.content == "Echo: Hello"

    # Second execution
    result2 = tool.execute(input="World")
    assert result2.content == "Echo: World"

    # Verify each execution was independent
    assert result1.metadata["messages"] != result2.metadata["messages"]


def test_llm_filtering_strategy():
    # Set up mock
    mock_model = Mock()
    mock_model.send.return_value = Message(
        role="assistant", content="0,2"
    )  # Keep first and third messages

    # Test messages
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="What's the weather today?"),
        Message(role="assistant", content="It's sunny"),
    ]

    # Custom prompt
    prompt = """
Select important message indices from the conversation below (comma-separated):

{conversation}

Selected indices (0-based):
"""

    # Create strategy and test
    strategy = LLMFilteringStrategy(model=mock_model, prompt=prompt)
    result = strategy.squash(messages)

    # Assertions
    assert len(result) == 2
    assert result[0].content == "Hello"
    assert result[1].content == "What's the weather today?"

    # Verify mock calls
    mock_model.send.assert_called_once()
    call_args = mock_model.send.call_args[0][0].messages[0].content
    assert "conversation" in prompt
    assert "Hello" in call_args
    assert "What's the weather today?" in call_args


def test_llm_summarizing_strategy():
    # Set up mock
    mock_model = Mock()
    mock_model.send.return_value = Message(
        role="assistant", content="User greeted and asked about the weather"
    )

    # Test messages
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="What's the weather today?"),
        Message(role="assistant", content="It's sunny"),
    ]

    # Custom prompt
    prompt = """
Summarize the conversation below:

{conversation}

Summary:
"""

    # Create strategy and test
    strategy = LLMSummarizingStrategy(model=mock_model, prompt=prompt)
    result = strategy.squash(messages)

    # Assertions
    assert len(result) == 1
    assert result[0].role == "assistant"
    assert result[0].content == "User greeted and asked about the weather"

    # Verify mock calls
    mock_model.send.assert_called_once()
    call_args = mock_model.send.call_args[0][0].messages[0].content
    assert "conversation" in prompt
    assert "Hello" in call_args
    assert "What's the weather today?" in call_args


def test_llm_filtering_strategy_empty_messages():
    # Test empty message list
    mock_model = Mock()
    strategy = LLMFilteringStrategy(model=mock_model, prompt="")
    result = strategy.squash([])

    # Assertions
    assert len(result) == 0
    mock_model.send.assert_not_called()


def test_llm_summarizing_strategy_empty_messages():
    # Test empty message list
    mock_model = Mock()
    strategy = LLMSummarizingStrategy(model=mock_model, prompt="")
    result = strategy.squash([])

    # Assertions
    assert len(result) == 0
    mock_model.send.assert_not_called()
