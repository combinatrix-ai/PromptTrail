# flake8: noqa: E402

import os
import sys

from prompttrail.agent.templates.anthropic import AnthropicToolingTemplate

sys.path.append(os.path.abspath("."))

import logging

from examples.dogfooding.dogfooding_tools import (
    CreateOrOverwriteFile,
    EditFile,
    ExecuteCommand,
    ReadFile,
    ReadImportantFiles,
    TreeDirectory,
    disable_noisy_loggers,
)
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate, LoopTemplate, UserTemplate
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel, AnthropicParam

logging.basicConfig(level=logging.DEBUG)
disable_noisy_loggers()

tools_to_use = [
    ReadImportantFiles(),
    ExecuteCommand(),
    ReadFile(),
    TreeDirectory(),
    CreateOrOverwriteFile(),
    EditFile(),
]

templates = LinearTemplate(
    templates=[
        UserTemplate(
            content="""Please help me improve my LLM library, PromptTrail. With the tools you provided, you can execute any linux command.""",
        ),
        LoopTemplate(
            [
                UserTemplate(
                    description="Input:",
                ),
                AnthropicToolingTemplate(tools=tools_to_use),
            ]
        ),
    ],
)

configuration = AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
parameter = AnthropicParam(
    model_name="claude-3-5-sonnet-latest",
    temperature=1,
    max_tokens=4096,
    tools=tools_to_use,
)
model = AnthropicModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

initial_session = Session()
runner.run(session=initial_session)
