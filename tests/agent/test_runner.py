# simple meta templates
from prompttrail.agent import State
from prompttrail.agent.hooks import BooleanHook, TransformHook
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    BreakTemplate,
    EndTemplate,
    IfTemplate,
    LinearTemplate,
    LoopTemplate,
    MessageTemplate,
)
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate,
    OpenAISystemTemplate,
)
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.core.mocks import EchoMockProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

# Run various templates

# TODO: Add tests for all templates

# Echo mock model
echo_mock_model = OpenAIChatCompletionModel(
    configuration=OpenAIModelConfiguration(
        api_key="", mock_provider=EchoMockProvider(sender="assistant")
    ),
)
parameters = OpenAIModelParameters(
    model_name="gpt-3.5-turbo",
)


def test_linear_template():
    template = LinearTemplate(
        templates=[
            OpenAISystemTemplate(content="Repeat what the user said."),
            MessageTemplate(
                content="Lazy fox jumps over the brown dog.",
                role="user",
            ),
            OpenAIGenerateTemplate(
                role="assistant",
            ),
        ]
    )
    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    state = runner.run(max_messages=10)

    assert len(state.session_history.messages) == 3
    assert state.session_history.messages[0].content == "Repeat what the user said."
    assert (
        state.session_history.messages[1].content
        == "Lazy fox jumps over the brown dog."
    )
    assert (
        state.session_history.messages[2].content
        == "Lazy fox jumps over the brown dog."
    )


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
                    lambda state: state.session_history.messages[-1].content == "TRUE"
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
    state = runner.run(state=State(data={"content": "TRUE"}), max_messages=10)

    assert len(state.session_history.messages) == 2
    assert state.session_history.messages[0].content == "TRUE"
    assert state.session_history.messages[1].content == "True"

    state = runner.run(state=State(data={"content": "FALSE"}), max_messages=10)
    print(state.session_history.messages)
    assert len(state.session_history.messages) == 2
    assert state.session_history.messages[0].content == "FALSE"
    assert state.session_history.messages[1].content == "False"


def test_loop_template():
    loop_count = 0

    def update_loop_count(state: State) -> State:
        nonlocal loop_count
        loop_count += 1
        state.data["loop_count"] = (
            state.data["loop_count"] + 1 if "loop_count" in state.data else 1
        )
        return state

    def mock_exit_condition(state: State) -> bool:
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
    state = runner.run(max_messages=10)

    # Check if it looped 3 times
    assert len(state.session_history.messages) == 3

    # Check if the generated messages are as expected
    for idx, message in enumerate(state.session_history.messages, start=1):
        assert message.content == f"This is loop iteration {idx}."


# Test nested LoopTemplates
def test_nested_loop_template():
    outer_loop_count = 0
    inner_loop_count = 0

    def update_outer_loop_count(state: State) -> State:
        nonlocal inner_loop_count
        nonlocal outer_loop_count
        inner_loop_count = 0
        state.data["inner_loop_count"] = 0
        outer_loop_count += 1
        state.data["outer_loop_count"] = (
            state.data["outer_loop_count"] + 1
            if "outer_loop_count" in state.data
            else 1
        )
        return state

    def update_inner_loop_count(state: State) -> State:
        nonlocal inner_loop_count
        inner_loop_count += 1
        state.data["inner_loop_count"] = (
            state.data["inner_loop_count"] + 1
            if "inner_loop_count" in state.data
            else 1
        )
        return state

    def mock_outer_exit_condition(state: State) -> bool:
        nonlocal outer_loop_count
        return outer_loop_count >= 3

    def mock_inner_exit_condition(state: State) -> bool:
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
    state = runner.run(max_messages=10)

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
    for idx, message in enumerate(state.session_history.messages):
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
                    lambda state: state.session_history.messages[-1].content == "TRUE"
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
    state = runner.run(state=State(data={"content": "TRUE"}), max_messages=10)

    assert len(state.session_history.messages) == 3
    assert state.session_history.messages[0].content == "TRUE"
    assert state.session_history.messages[1].content == "True"
    assert (
        state.session_history.messages[2].content
        == "This is rendered if the previous message was TRUE"
    )

    state = runner.run(state=State(data={"content": "FALSE"}), max_messages=10)
    assert len(state.session_history.messages) == 1
    assert state.session_history.messages[0].content == "FALSE"


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
                    lambda state: state.session_history.messages[-1].content == "TRUE"
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
    state = runner.run(state=State(data={"content": "TRUE"}), max_messages=10)

    assert len(state.session_history.messages) == 3
    assert state.session_history.messages[0].content == "TRUE"
    assert state.session_history.messages[1].content == "True"
    assert (
        state.session_history.messages[2].content
        == "This is rendered if the previous message was TRUE"
    )

    state = runner.run(state=State(data={"content": "FALSE"}), max_messages=10)
    assert len(state.session_history.messages) == 1
    assert state.session_history.messages[0].content == "FALSE"


def test_nested_loop_with_break_template():
    outer_loop_counter = 0
    inner_loop_counter = 0

    def update_outer_loop_counter(state: State) -> State:
        nonlocal outer_loop_counter
        outer_loop_counter += 1
        state.data["outer_loop_counter"] = outer_loop_counter
        # reset inner loop counter
        nonlocal inner_loop_counter
        inner_loop_counter = 0
        return state

    def update_inner_loop_counter(state: State) -> State:
        nonlocal inner_loop_counter
        inner_loop_counter = (inner_loop_counter + 1) % 5  # Reset after 5 iterations
        state.data["inner_loop_counter"] = inner_loop_counter
        return state

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
                    lambda state: state.data["inner_loop_counter"] > 2
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
            condition=lambda state: state.data["outer_loop_counter"] >= 3
        ),
    )

    runner = CommandLineRunner(
        model=echo_mock_model,
        parameters=parameters,
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=outer_loop_template,
    )

    # We'll break the inner loop on the 3rd iteration
    state = runner.run(
        state=State(data={"outer_loop_counter": 0, "inner_loop_counter": 0}),
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
    for idx, message in enumerate(state.session_history.messages):
        assert message.content == expected_messages[idx]
