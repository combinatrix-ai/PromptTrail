# flake8: noqa: E402
import logging
import os
import sys

sys.path.append(os.path.abspath("."))

from examples.dogfooding.dogfooding_tools import load_all_important_files
from prompttrail.agent.hooks import ResetDataHook
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    LoopTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel, AnthropicParam

logging.basicConfig(level=logging.ERROR)

templates = LinearTemplate(
    templates=[
        SystemTemplate(
            content="""
You're given source code and test scripts and documents for a library, PromptTrail as below:
{{code}}
Discuss the question with user. User is the author of this library, who want to improve the design, implementation, and documentation of the library.
""",
            after_transform=ResetDataHook(),
        ),
        LoopTemplate(
            [
                UserTemplate(
                    description="Input: ",
                ),
                AssistantTemplate(),
            ]
        ),
    ],
)

configuration = AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
parameter = AnthropicParam(
    model_name="claude-3-5-sonnet-latest",
    temperature=1,
    max_tokens=4096,
)
model = AnthropicModel(configuration=configuration)

content = load_all_important_files()

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

initial_session = Session(metadata={"code": content})
runner.run(session=initial_session)
