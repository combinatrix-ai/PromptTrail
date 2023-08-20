# simple meta templates
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.template import LinearTemplate, MessageTemplate
from prompttrail.agent.template.openai import (
    OpenAIGenerateTemplate,
    OpenAISystemTemplate,
)
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.mock import EchoMockProvider
from prompttrail.provider.openai import (
    OpenAIChatCompletionModelMock,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

# Run various templates

# TODO: Add tests for all templates


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
        model=OpenAIChatCompletionModelMock(
            configuration=OpenAIModelConfiguration(
                api_key="",
            ),
            mock_provider=EchoMockProvider(sender="assistant"),
        ),
        parameters=OpenAIModelParameters(
            model_name="gpt-3.5-turbo",
        ),
        user_interaction_provider=EchoUserInteractionTextMockProvider(),
        template=template,
    )
    state = runner.run(max_messages=10)

    print(state.session_history.messages)
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
