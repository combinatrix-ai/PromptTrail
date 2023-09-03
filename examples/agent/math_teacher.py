import logging
import os

from prompttrail.agent.hooks import BooleanHook
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    LinearTemplate,
    LoopTemplate,
    MessageTemplate,
    UserInputTextTemplate,
)
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate as GenerateTemplate,
)
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

logging.basicConfig(level=logging.INFO)

template = LinearTemplate(
    [
        MessageTemplate(
            role="system",
            content="You're a math teacher bot.",
        ),
        LoopTemplate(
            [
                UserInputTextTemplate(
                    role="user",
                    description="Let's ask a question to AI:",
                    default="Why can't you divide a number by zero?",
                ),
                GenerateTemplate(
                    role="assistant",
                ),
                MessageTemplate(role="assistant", content="Are you satisfied?"),
                UserInputTextTemplate(
                    role="user",
                    description="Input:",
                    default="Yes.",
                ),
                # Let the LLM decide whether to end the conversation or not
                MessageTemplate(
                    role="assistant",
                    content="The user has stated their feedback."
                    + "If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`.",
                ),
                check_end := GenerateTemplate(
                    role="assistant",
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda state: (
                    "END" == state.get_last_message().content.strip()
                )
            ),
        ),
    ],
)


runner = CommandLineRunner(
    model=OpenAIChatCompletionModel(
        configuration=OpenAIModelConfiguration(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
    ),
    parameters=OpenAIModelParameters(model_name="gpt-4"),
    template=template,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

runner.run()
