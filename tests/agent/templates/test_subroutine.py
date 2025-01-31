from typing import Generator, List

from prompttrail.agent.subroutine import SubroutineTemplate
from prompttrail.agent.subroutine.session_init_strategy import (
    CleanSessionStrategy,
    FilteredInheritStrategy,
    InheritSystemStrategy,
    LastNMessagesStrategy,
)
from prompttrail.agent.subroutine.squash_strategy import (
    FilterByRoleStrategy,
    LastMessageStrategy,
)
from prompttrail.agent.templates import Stack, Template
from prompttrail.core import Message, Session


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

    strategy = CleanSessionStrategy()
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
        session_init_strategy=CleanSessionStrategy(),
        squash_strategy=LastMessageStrategy(),
    )

    messages = list(subroutine.render(parent_session))

    assert len(messages) == 2
    assert len(parent_session.messages) == 2  # system + last message
    assert parent_session.messages[-1].content == "test2"
