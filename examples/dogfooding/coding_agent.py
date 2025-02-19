# flake8: noqa: E402

import os
import sys

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.session_transformers import ResetMetadata
from prompttrail.agent.templates import (
    LinearTemplate,
    LoopTemplate,
    ToolingTemplate,
    UserTemplate,
)
from prompttrail.agent.tools.builtin import (
    CreateOrOverwriteFile,
    EditFile,
    ExecuteCommand,
    ReadFile,
    TreeDirectory,
)
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel

sys.path.append(os.path.abspath("."))


from examples.dogfooding.dogfooding_tools import (
    ReadImportantFiles,
)

tools_to_use = [
    ReadImportantFiles(),
    ExecuteCommand(),
    ReadFile(),
    TreeDirectory(),
    CreateOrOverwriteFile(),
    EditFile(),
    # RunTest(),
]

templates = LinearTemplate(
    [
        UserTemplate(
            content="""
Please help me improve my LLM library, PromptTrail. With the tools you provided, you can execute any linux command.

Rules:
# Ask users before edit
- You must show the edits you want to make before actual edit.

# Use tools to do action
- You must use the tools provided to do any action.

# Run tests after finish edit
- You must run RunTest tool if you finished your work.

# Important Rules for Commits
- Always prefix commit message titles with '[coding-agent]'
- Add a note at the end of commit message: "Note: This commit was automatically generated by coding_agent.py using Claude"

# Project Rules:
{{clinerules}}
""",
            after_transform=ResetMetadata(),
        ),
        LoopTemplate(
            [
                UserTemplate(
                    description="Input:",
                ),
                ToolingTemplate(tools=tools_to_use),
            ]
        ),
    ],
)

configuration = AnthropicConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model_name="claude-3-5-sonnet-latest",
    temperature=1,
    max_tokens=4096,
    tools=tools_to_use,
)
model = AnthropicModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    template=templates,
    user_interface=CLIInterface(),
)

initial_session = Session(
    metadata={
        # read .clinerules file
        "clinerules": open(".clinerules").read(),
    }
)
runner.run(session=initial_session)
