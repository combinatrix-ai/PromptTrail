# fmt: off
# flake8: noqa
# This is a snippet for the README.md. So, no formatting is needed.

import os

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import *
from prompttrail.agent.tools.builtin import *
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel

# Build your agent with intuitive DSL
templates = LinearTemplate([
    # Dynamic message generation with Jinja2 templating
    SystemTemplate(content="You're a smart coding agent! Type END if you want to end conversation. Follow rules: {{clinerules}}"),
    # Complex control flow with loops and conditionals
    LoopTemplate([
        # Easy user interaction handling
        UserTemplate(description="Input: "),
        # Powerful built-in tools for file and system operations
        ToolingTemplate(tools=[ExecuteCommand(),ReadFile(),CreateOrOverwriteFile(),EditFile()])],
        exit_condition=lambda session: session.messages[-1].content == "END")])

# Various models are supported with the same API
model = AnthropicModel(
    configuration=AnthropicConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name="claude-3-5-sonnet-latest",
        max_tokens=4096,
        tools=[ExecuteCommand(), ReadFile(), CreateOrOverwriteFile(), EditFile()]))

# Metadata is a powerful way to pass, store, and retrieve information in your agent
initial_session = Session(metadata={"clinerules": open(".clinerules").read()})

# Run your agent easily
runner = CommandLineRunner(model=model, template=templates, user_interface=CLIInterface())
runner.run(session=initial_session)
