# This script is actually used to housekeep the markdown files in this repository.
# See https://github.com/combinatrix-ai/PromptTrail/pull/3 for what it does.

import os
import sys

from prompttrail.agent.core import State
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.template import LinearTemplate
from prompttrail.agent.template import OpenAIGenerateTemplate as GenerateTemplate
from prompttrail.agent.template import OpenAIMessageTemplate as MessageTemplate
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.provider.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI proofreader that help user to fix markdown.
You're given markdown content by the user.
You only emit the corrected markdown. No explanation, comments, or anything else is needed.
Do not remove > in the code section, which represents the prompt.
""",
            role="system",
        ),
        MessageTemplate(
            content="{{content}}",
            role="user",
        ),
        GenerateTemplate(role="assistant"),
    ],
)

configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=8000
)
model = OpenAIChatCompletionModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    templates=[templates],
    user_interaction_provider=UserInteractionTextCLIProvider(),
)


def main(
    load_file: str,
):
    # show all log levels
    import logging

    logging.basicConfig(level=logging.DEBUG)

    load_file_content = open(load_file, "r")
    initial_state = State(
        data={
            "content": load_file_content.read(),
        }
    )
    state = runner.run(state=initial_state)
    last_message = state.get_last_message()
    message = last_message.content
    print(message)
    if len(sys.argv) > 2:
        # add EOF if not exists
        if message[-1] != "\n":
            message += "\n"
        save_file_io = open(load_file, "w")
        save_file_io.write(last_message.content)
        save_file_io.close()


if __name__ == "__main__":
    main(load_file="README.md")
    main(load_file="CONTRIBUTING.md")
