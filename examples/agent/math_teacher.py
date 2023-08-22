import logging
import os

from prompttrail.agent.hook.core import BooleanHook
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.template.control import LinearTemplate, LoopTemplate
from prompttrail.agent.template.core import MessageTemplate, UserInputTextTemplate
from prompttrail.agent.template.openai import OpenAIGenerateTemplate as GenerateTemplate
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.provider.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

logging.basicConfig(level=logging.INFO)

template = LinearTemplate(
    [
        MessageTemplate(
            role="system",
            content="You're a math teacher. You're teaching a student how to solve equations.",
        ),
        LoopTemplate(
            [
                UserInputTextTemplate(
                    role="user",
                    description="Let's ask question to AI:",
                    default="Why can't you divide a number by zero?",
                ),
                GenerateTemplate(
                    role="assistant",
                ),
                MessageTemplate(role="assistant", content="Are you satisfied?"),
                UserInputTextTemplate(
                    role="user",
                    description="Input:",
                    default="Explain more.",
                ),
                # Let the LLM decide whether to end the conversation or not
                MessageTemplate(
                    role="assistant",
                    content="""
                    The user has stated their feedback.
                    If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`.
                    """,
                ),
                check_end := GenerateTemplate(
                    role="assistant",
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda state: ("END" in state.get_last_message().content)
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
