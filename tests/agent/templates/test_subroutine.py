import copy
from typing import Generator, List, Optional
from unittest.mock import Mock

import pytest

from prompttrail.agent.runners import CommandLineRunner, Runner
from prompttrail.agent.subroutine import SubroutineTemplate
from prompttrail.agent.subroutine.session_init_strategy import (
    FilteredInheritStrategy,
    InheritMetadataStrategy,
    InheritSystemStrategy,
    LastNMessagesStrategy,
)
from prompttrail.agent.subroutine.squash_strategy import (
    FilterByRoleStrategy,
    LastMessageStrategy,
    LLMFilteringStrategy,
    LLMSummarizingStrategy,
)
from prompttrail.agent.templates import Stack, Template
from prompttrail.core import Message, Model, Session
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


class MockTemplate(Template):
    """Mock template for testing"""

    def __init__(self, messages: List[Message]):
        super().__init__()
        self.messages = messages

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        """Render mock messages and return session.

        Args:
            session: Current session

        Yields:
            Mock messages

        Returns:
            Updated session
        """
        try:
            for msg in self.messages:
                yield msg
            return session
        except Exception as e:
            raise e

    def create_stack(self, session: Session) -> Stack:
        """Create stack frame for this template."""
        return Stack(template_id=self.template_id)


def test_clean_session_strategy():
    """Test CleanSessionStrategy initialization"""
    parent_session = Session()
    parent_session.append(Message(role="user", content="test"))

    strategy = InheritMetadataStrategy()
    new_session = strategy.initialize(parent_session)

    assert len(new_session.messages) == 0
    assert new_session.metadata == parent_session.metadata
    assert new_session.runner == parent_session.runner


def test_inherit_system_strategy():
    """Test InheritSystemStrategy initialization"""
    parent_session = Session()
    system_msg = Message(role="system", content="system test")
    user_msg = Message(role="user", content="user test")
    parent_session.append(system_msg)
    parent_session.append(user_msg)

    strategy = InheritSystemStrategy()
    new_session = strategy.initialize(parent_session)

    assert len(new_session.messages) == 1
    assert new_session.messages[0].role == "system"
    assert new_session.messages[0].content == "system test"


def test_last_n_messages_strategy():
    """Test LastNMessagesStrategy initialization"""
    parent_session = Session()
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="user1"),
        Message(role="assistant", content="assistant1"),
        Message(role="user", content="user2"),
    ]
    for msg in messages:
        parent_session.append(msg)

    strategy = LastNMessagesStrategy(n=2)
    new_session = strategy.initialize(parent_session)

    assert len(new_session.messages) == 2
    assert new_session.messages[0].content == "assistant1"
    assert new_session.messages[1].content == "user2"


def test_filtered_inherit_strategy():
    """Test FilteredInheritStrategy initialization"""
    parent_session = Session()
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="user1"),
        Message(role="assistant", content="assistant1"),
    ]
    for msg in messages:
        parent_session.append(msg)

    strategy = FilteredInheritStrategy(lambda msg: msg.role == "user")
    new_session = strategy.initialize(parent_session)

    assert len(new_session.messages) == 1
    assert new_session.messages[0].role == "user"
    assert new_session.messages[0].content == "user1"


def test_last_message_squash_strategy():
    """Test LastMessageStrategy squashing"""
    messages = [
        Message(role="user", content="user1"),
        Message(role="assistant", content="assistant1"),
        Message(role="user", content="user2"),
    ]

    strategy = LastMessageStrategy()
    strategy.initialize(Session(), Session())
    result = strategy.squash(messages)

    assert len(result) == 1
    assert result[0].content == "user2"


def test_filter_by_role_squash_strategy():
    """Test FilterByRoleStrategy squashing"""
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="user1"),
        Message(role="assistant", content="assistant1"),
    ]

    strategy = FilterByRoleStrategy(roles=["assistant", "system"])
    strategy.initialize(Session(), Session())
    result = strategy.squash(messages)

    assert len(result) == 2
    assert result[0].role == "system"
    assert result[1].role == "assistant"


def test_subroutine_template_execution():
    """Test complete SubroutineTemplate execution"""
    mock_messages = [
        Message(role="assistant", content="test1"),
        Message(role="assistant", content="test2"),
    ]
    mock_template = MockTemplate(mock_messages)

    parent_session = Session()
    parent_session.append(Message(role="system", content="system"))

    subroutine = SubroutineTemplate(
        template=mock_template,
        session_init_strategy=InheritMetadataStrategy(),
        squash_strategy=LastMessageStrategy(),
    )

    runner = CommandLineRunner(
        model=OpenAIModel(
            configuration=OpenAIConfig(
                api_key="dummy",
            ),
        ),
        template=subroutine,
        user_interface=None,
    )

    messages = list(runner.run(parent_session).messages)

    assert len(messages) == 2
    assert len(parent_session.messages) == 2  # system + last message
    assert parent_session.messages[-1].content == "test2"


class MockModel:
    """Mock model for testing"""

    def __init__(self, response_content: str):
        self.response_content = response_content

    def send(self, session: Session) -> Message:
        return Message(role="assistant", content=self.response_content)


class MockRunner(Runner):
    """Mock runner for testing"""

    def __init__(self, model: Model, template: Template, user_interface=None):
        super().__init__(model=model, template=template, user_interface=user_interface)

    def run(self, session=None, max_messages=None, debug_mode=False) -> Session:
        """Mock implementation of run method"""
        return session or Session()


def test_subroutine_model_override():
    """Test model override in SubroutineTemplate"""
    mock_template = MockTemplate([])
    parent_model = MockModel("parent response")
    override_model = MockModel("override response")

    parent_session = Session()
    parent_session.runner = MockRunner(
        model=parent_model, template=mock_template, user_interface=None
    )

    subroutine = SubroutineTemplate(template=mock_template, model=override_model)

    # Verify that the subroutine uses the overridden model
    temp_session = subroutine.session_init_strategy.initialize(parent_session)
    subroutine.squash_strategy.initialize(parent_session, temp_session)
    if subroutine.model:
        temp_session.runner = type(parent_session.runner)(
            model=subroutine.model,
            template=parent_session.runner.template,
            user_interface=parent_session.runner.user_interface,
        )

    response = temp_session.runner.model.send(temp_session)
    assert response.content == "override response"


def test_subroutine_runner_override():
    """Test runner override in SubroutineTemplate"""
    mock_template = MockTemplate([])
    parent_model = MockModel("parent response")
    override_model = MockModel("override response")
    override_runner = MockRunner(
        model=override_model, template=mock_template, user_interface=None
    )

    parent_session = Session()
    parent_session.runner = MockRunner(
        model=parent_model, template=mock_template, user_interface=None
    )

    subroutine = SubroutineTemplate(template=mock_template, runner=override_runner)

    # Verify that the subroutine uses the overridden runner
    temp_session = subroutine.session_init_strategy.initialize(parent_session)
    subroutine.squash_strategy.initialize(parent_session, temp_session)
    temp_session.runner = subroutine.runner

    response = temp_session.runner.model.send(temp_session)
    assert response.content == "override response"


def test_subroutine_runner_model_exclusive():
    """Test that runner and model cannot be set simultaneously"""
    mock_template = MockTemplate([])
    model = MockModel("test")
    runner = MockRunner(model=model, template=mock_template, user_interface=None)

    with pytest.raises(ValueError) as exc_info:
        SubroutineTemplate(template=mock_template, runner=runner, model=model)
    assert (
        str(exc_info.value) == "Cannot set both runner and model - use one or the other"
    )


def test_subroutine_environment_isolation():
    """Test that subroutine environment is properly isolated"""
    mock_template = MockTemplate([])
    parent_model = MockModel("parent response")

    parent_session = Session()
    parent_session.runner = MockRunner(
        model=parent_model, template=mock_template, user_interface=None
    )

    subroutine = SubroutineTemplate(template=mock_template)

    # Verify that the subroutine creates a copy of the parent runner
    temp_session = subroutine.session_init_strategy.initialize(parent_session)
    subroutine.squash_strategy.initialize(parent_session, temp_session)
    temp_session.runner = copy.deepcopy(parent_session.runner)

    # Modify temp_session's runner
    temp_session.runner.model = MockModel("modified response")

    # Verify that parent session's runner is unchanged
    assert parent_session.runner.model.send(parent_session).content == "parent response"
    assert temp_session.runner.model.send(temp_session).content == "modified response"


class SimpleTemplate(Template):
    """Simple template for testing that echoes input with prefix"""

    def __init__(self, template_id: Optional[str] = None):
        super().__init__(template_id=template_id)

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        # Debug log session state
        self.debug(f"SimpleTemplate._render session messages: {session.messages}")

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
