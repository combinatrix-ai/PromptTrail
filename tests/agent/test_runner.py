# simple meta templates
from prompttrail.agent.hook.core import BooleanHook, CountUpHook
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.template import LinearTemplate, LoopTemplate, MessageTemplate
from prompttrail.agent.template import OpenAIGenerateTemplate as GenerateTemplate
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.mock import EchoMockProvider
from prompttrail.provider.openai import (
    OpenAIChatCompletionModelMock,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

template = LinearTemplate(
    templates=[
        LoopTemplate(
            templates=[
                MessageTemplate(
                    content="You must repeat what the user said.", role="system"
                ),
                LoopTemplate(
                    templates=[
                        MessageTemplate(
                            role="user",
                            content="TEST",
                        ),
                        assistant_reply := GenerateTemplate(
                            role="assistant", after_transform=[CountUpHook()]
                        ),
                    ],
                    exit_condition=BooleanHook(
                        lambda flow_state: len(flow_state.session_history.messages) > 10
                    ),
                ),
            ],
            exit_condition=BooleanHook(
                lambda flow_state: flow_state.data[assistant_reply.template_id] >= 2
                if assistant_reply.template_id in flow_state.data
                else False
            ),
        )
    ]
)


def test_runner():
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
        templates=[template],
    )
    flow_state = runner.run(max_messages=100)
    print(flow_state)
