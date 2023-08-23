import os
from typing import List, Optional

from prompttrail.agent import State
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.template import LinearTemplate
from prompttrail.agent.template import OpenAIGenerateTemplate as GenerateTemplate
from prompttrail.agent.template import OpenAIMessageTemplate as MessageTemplate
from prompttrail.provider.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

# Setup LLM model
# Don't forget to set OPENAI_API_KEY environment variable
configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=8000
)
model = OpenAIChatCompletionModel(configuration=configuration)

# Define templates
templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI proofreader that helps users fix markdown.
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

# Define runner
runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    templates=[templates],
)

# Prepare markdown to be proofread
markdown = """
# PromptTrail

PromptTrail is a library to build a text generation agent with LLMs.
"""

# Run the agent
result = runner.run(
    state=State(
        data={"content": markdown},
    ),
)

# Get the corrected markdown
corrected_markdown = result.session_history.messages[-1].content
print(corrected_markdown)
