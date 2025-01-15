import logging
import os

from prompttrail.agent.hooks import BooleanHook
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    LoopTemplate,
    MessageTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import OpenAIConfiguration, OpenAIModel, OpenAIParam

logging.basicConfig(level=logging.INFO)

template = LinearTemplate(
    [
        SystemTemplate(
            content="You're a math teacher bot.",
        ),
        LoopTemplate(
            [
                UserTemplate(
                    description="Let's ask a question to AI:",
                    default="Why can't you divide a number by zero?",
                ),
                AssistantTemplate(),
                MessageTemplate(role="assistant", content="Are you satisfied?"),
                UserTemplate(
                    description="Input:",
                    default="Yes.",
                ),
                # Let the LLM decide whether to end the conversation or not
                MessageTemplate(
                    role="assistant",
                    content="The user has stated their feedback."
                    + "If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`.",
                ),
                check_end := AssistantTemplate(),
            ],
            exit_condition=BooleanHook(
                condition=lambda session: ("END" == session.get_last().content.strip())
            ),
        ),
    ],
)


runner = CommandLineRunner(
    model=OpenAIModel(
        configuration=OpenAIConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
    ),
    parameters=OpenAIParam(model_name="gpt-4o-mini"),
    template=template,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

runner.run()
