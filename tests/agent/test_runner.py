# simple meta templates
from prompttrail.agent import Session
from prompttrail.agent.hooks import BooleanHook, TransformHook
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    BreakTemplate,
    EndTemplate,
    IfTemplate,
    LinearTemplate,
    LoopTemplate,
    MessageTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.core.mocks import EchoMockProvider
from prompttrail.models.openai import OpenAIConfiguration, OpenAIModel, OpenAIParam

# Run various templates

# TODO: Add tests for all templates

# Echo mock model
echo_mock_model = OpenAIModel(
    configuration=OpenAIConfiguration(
        api_key="", mock_provider=EchoMockProvider(role="assistant")
    ),
)
parameters = OpenAIParam(
    model_name="gpt-4o-mini",
)


def test_linear_template():
    template = LinearTemplate(
        templates=[
            SystemTemplate(content="Repeat what the user said."),
            UserTemplate(content="Lazy fox jumps over the brown dog."),
            AssistantTemplate(),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    session = runner.run(max_messages=10)

    assert len(session.messages) == 3
    assert session.messages[0].content == "Repeat what the user said."
    assert session.messages[1].content == "Lazy fox jumps over the brown dog."
    assert session.messages[2].content == "Lazy fox jumps over the brown dog."


def test_if_template():
    template = LinearTemplate(
        templates=[
            MessageTemplate(
                content="{{ content }}",
                role="system",
            ),
            IfTemplate(
                true_template=MessageTemplate(
                    content="True",
                    role="assistant",
                ),
                false_template=MessageTemplate(
                    content="False",
                    role="assistant",
                ),
                condition=BooleanHook(
                    lambda session: session.messages[-1].content == "TRUE"
                ),
            ),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    session = Session(initial_metadata={"content": "TRUE"})
    session = runner.run(session=session, max_messages=10)

    assert len(session.messages) == 2  # system message + True/False message
    assert session.messages[0].content == "TRUE"
    assert session.messages[1].content == "True"

    session = Session(initial_metadata={"content": "FALSE"})
    session = runner.run(session=session, max_messages=10)
    print(session.messages)
    assert len(session.messages) == 2
    assert session.messages[0].content == "FALSE"
    assert session.messages[1].content == "False"


def test_loop_template():
    loop_count = 0

    def update_loop_count(session: Session) -> Session:
        nonlocal loop_count
        loop_count += 1
        for message in session.messages:
            message.metadata["loop_count"] = loop_count
        return session

    def mock_exit_condition(session: Session) -> bool:
        # For the purpose of the test, let's exit after 3 iterations
        # This is evaluated after the child templates are evaluated
        return loop_count >= 3

    template = LoopTemplate(
        templates=[
            MessageTemplate(
                role="assistant",
                # loop_count is 1,2,3... as before_transform is called before the message is rendered
                content="This is loop iteration {{ loop_count }}.",
                before_transform=[TransformHook(update_loop_count)],
            )
        ],
        exit_condition=BooleanHook(condition=mock_exit_condition),
    )

    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        template=template,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
    )
    session = Session(initial_metadata={"loop_count": 1})
    session = runner.run(session=session, max_messages=10)

    # Check if it looped 3 times
    assert len(session.messages) == 3  # 3 loop iterations

    # Check if the generated messages are as expected
    for idx, message in enumerate(session.messages, start=1):
        assert message.content == f"This is loop iteration {idx}."


# Test nested LoopTemplates
def test_nested_loop_template():
    outer_loop_count = 0
    inner_loop_count = 0

    def update_outer_loop_count(session: Session) -> Session:
        nonlocal inner_loop_count
        nonlocal outer_loop_count
        inner_loop_count = 0
        outer_loop_count += 1
        for message in session.messages:
            message.metadata["inner_loop_count"] = 0
            message.metadata["outer_loop_count"] = outer_loop_count
        return session

    def update_inner_loop_count(session: Session) -> Session:
        nonlocal inner_loop_count
        inner_loop_count += 1
        for message in session.messages:
            message.metadata["inner_loop_count"] = inner_loop_count
        return session

    def mock_outer_exit_condition(session: Session) -> bool:
        nonlocal outer_loop_count
        return outer_loop_count >= 3

    def mock_inner_exit_condition(session: Session) -> bool:
        nonlocal inner_loop_count
        return inner_loop_count >= 2

    template = LoopTemplate(
        templates=[
            MessageTemplate(
                role="assistant",
                content="Outer Loop: {{ outer_loop_count }}",
                before_transform=[TransformHook(update_outer_loop_count)],
            ),
            LoopTemplate(
                templates=[
                    MessageTemplate(
                        role="assistant",
                        content="  Inner Loop: {{ inner_loop_count }}",
                        before_transform=[TransformHook(update_inner_loop_count)],
                    )
                ],
                exit_condition=BooleanHook(condition=mock_inner_exit_condition),
            ),
        ],
        exit_condition=BooleanHook(condition=mock_outer_exit_condition),
    )

    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        template=template,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
    )
    session = Session(initial_metadata={"outer_loop_count": 1, "inner_loop_count": 1})
    session = runner.run(session=session, max_messages=10)

    # Validate the generated messages
    expected_messages = [
        "Outer Loop: 1",
        "  Inner Loop: 1",
        "  Inner Loop: 2",
        "Outer Loop: 2",
        "  Inner Loop: 1",
        "  Inner Loop: 2",
        "Outer Loop: 3",
        # For LoopTemplate, this behaiviour is correct because the exit_condition is evaluated after each child templates are evaluated
        # TODO: This should be changed?
        # "  Inner Loop: 1",
        # "  Inner Loop: 2",
    ]
    for idx, message in enumerate(session.messages):
        assert message.content == expected_messages[idx]


def test_end_template():
    template = LinearTemplate(
        templates=[
            MessageTemplate(
                content="{{ content }}",
                role="system",
            ),
            IfTemplate(
                true_template=MessageTemplate(
                    content="True",
                    role="assistant",
                ),
                false_template=EndTemplate(),
                condition=BooleanHook(
                    lambda session: session.messages[-1].content == "TRUE"
                ),
            ),
            MessageTemplate(
                role="assistant",
                content="This is rendered if the previous message was TRUE",
            ),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    session = Session(initial_metadata={"content": "TRUE"})
    session = runner.run(session=session, max_messages=10)

    assert len(session.messages) == 3  # system message + True message + final message
    assert session.messages[0].content == "TRUE"
    assert session.messages[1].content == "True"
    assert (
        session.messages[2].content
        == "This is rendered if the previous message was TRUE"
    )

    session = Session(initial_metadata={"content": "FALSE"})
    session = runner.run(session=session, max_messages=10)
    assert len(session.messages) == 1
    assert session.messages[0].content == "FALSE"


def test_break_template():
    template = LinearTemplate(
        templates=[
            MessageTemplate(
                content="{{ content }}",
                role="system",
            ),
            IfTemplate(
                true_template=MessageTemplate(
                    content="True",
                    role="assistant",
                ),
                false_template=BreakTemplate(),
                condition=BooleanHook(
                    lambda session: session.messages[-1].content == "TRUE"
                ),
            ),
            MessageTemplate(
                role="assistant",
                content="This is rendered if the previous message was TRUE",
            ),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    session = Session(initial_metadata={"content": "TRUE"})
    session = runner.run(session=session, max_messages=10)

    assert len(session.messages) == 3  # system message + True message + final message
    assert session.messages[0].content == "TRUE"
    assert session.messages[1].content == "True"
    assert (
        session.messages[2].content
        == "This is rendered if the previous message was TRUE"
    )

    session = Session(initial_metadata={"content": "FALSE"})
    session = runner.run(session=session, max_messages=10)
    assert len(session.messages) == 1
    assert session.messages[0].content == "FALSE"


def test_system_template():
    """Test SystemTemplate with static content."""
    template = LinearTemplate(
        templates=[
            SystemTemplate(
                content="You are a helpful assistant.",
            ),
            MessageTemplate(
                content="Hello!",
                role="user",
            ),
            AssistantTemplate(),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    session = runner.run(max_messages=10)

    assert len(session.messages) == 3
    assert session.messages[0].role == "system"
    assert session.messages[0].content == "You are a helpful assistant."
    assert session.messages[1].role == "user"
    assert session.messages[1].content == "Hello!"
    assert session.messages[2].role == "assistant"
    assert session.messages[2].content == "Hello!"


def test_user_template():
    """Test UserTemplate in both static and interactive modes."""
    # Test static mode
    static_template = LinearTemplate(
        templates=[
            UserTemplate(
                content="Hello, assistant!",
            ),
            AssistantTemplate(),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=static_template,
    )
    session = runner.run(session=Session(), max_messages=10)

    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hello, assistant!"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Hello, assistant!"

    # Test interactive mode
    interactive_template = LinearTemplate(
        templates=[
            SystemTemplate(
                content="You are a helpful assistant.",
            ),
            UserTemplate(
                description="Enter your message:",
                default="Hello from user!",
            ),
            AssistantTemplate(),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=interactive_template,
    )
    session = runner.run(max_messages=10)

    assert len(session.messages) == 3
    assert session.messages[0].role == "system"
    assert session.messages[0].content == "You are a helpful assistant."
    assert session.messages[1].role == "user"
    # Because we are using the echo mock provider, the user's message is echoed back
    assert session.messages[1].content == "You are a helpful assistant."
    assert session.messages[2].role == "assistant"
    assert session.messages[2].content == "You are a helpful assistant."


def test_assistant_template():
    """Test AssistantTemplate in both static and generate modes."""
    # Test static mode
    static_template = LinearTemplate(
        templates=[
            MessageTemplate(
                content="Hello!",
                role="user",
            ),
            AssistantTemplate(
                content="I'm here to help!",
            ),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=static_template,
    )
    session = runner.run(max_messages=10)

    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hello!"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "I'm here to help!"

    # Test generate mode
    generate_template = LinearTemplate(
        templates=[
            MessageTemplate(
                content="Hello!",
                role="user",
            ),
            AssistantTemplate(),  # No content = generate mode
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=generate_template,
    )
    session = runner.run(max_messages=10)

    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hello!"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Hello!"  # Echo mock returns user's message


def test_nested_loop_with_break_template():
    outer_loop_counter = 0
    inner_loop_counter = 0

    def update_outer_loop_counter(session: Session) -> Session:
        nonlocal outer_loop_counter
        outer_loop_counter += 1
        metadata = session.get_latest_metadata()
        metadata["outer_loop_counter"] = outer_loop_counter
        metadata["inner_loop_counter"] = 0  # Reset inner loop counter
        # reset inner loop counter
        nonlocal inner_loop_counter
        inner_loop_counter = 0
        return session

    def update_inner_loop_counter(session: Session) -> Session:
        nonlocal inner_loop_counter
        inner_loop_counter = (inner_loop_counter + 1) % 5  # Reset after 5 iterations
        metadata = session.get_latest_metadata()
        metadata["inner_loop_counter"] = inner_loop_counter
        return session

    inner_loop_template = LoopTemplate(
        templates=[
            IfTemplate(
                before_transform=[TransformHook(update_inner_loop_counter)],
                true_template=BreakTemplate(),
                false_template=MessageTemplate(
                    role="assistant",
                    content="Inner loop iteration {{ inner_loop_counter }} of outer iteration {{ outer_loop_counter }}",
                ),
                condition=BooleanHook(
                    lambda session: session.get_latest_metadata()["inner_loop_counter"]
                    > 2
                ),
            ),
        ],
    )

    outer_loop_template = LoopTemplate(
        templates=[
            MessageTemplate(
                before_transform=[TransformHook(update_outer_loop_counter)],
                role="assistant",
                content="Starting outer loop iteration {{ outer_loop_counter }}",
            ),
            inner_loop_template,
        ],
        exit_condition=BooleanHook(
            condition=lambda session: session.get_latest_metadata()[
                "outer_loop_counter"
            ]
            >= 3
        ),
    )

    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=outer_loop_template,
    )

    # We'll break the inner loop on the 3rd iteration
    session = Session(
        initial_metadata={"outer_loop_counter": 1, "inner_loop_counter": 1}
    )
    session = runner.run(
        session=session,
        max_messages=10,
        debug_mode=True,
    )
    # Expecting the inner loop to break on the 3rd iteration for each outer loop iteration
    expected_messages = [
        "Starting outer loop iteration 1",
        "Inner loop iteration 1 of outer iteration 1",
        "Inner loop iteration 2 of outer iteration 1",
        "Starting outer loop iteration 2",
        "Inner loop iteration 1 of outer iteration 2",
        "Inner loop iteration 2 of outer iteration 2",
        "Starting outer loop iteration 3",
        "Inner loop iteration 1 of outer iteration 3",
        "Inner loop iteration 2 of outer iteration 3",
    ]
    for idx, message in enumerate(session.messages):
        assert message.content == expected_messages[idx]
