import glob
import os

from tqdm import tqdm

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate as GenerateTemplate,
)
from prompttrail.agent.templates.openai import OpenAIMessageTemplate as MessageTemplate
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.core import Message, Session
from prompttrail.models.anthropic import (
    AnthropicClaudeModel,
    AnthropicClaudeModelConfiguration,
    AnthropicClaudeModelParameters,
)

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're given examples and test scripts and documents for a library, PromptTrail.
Write a quickstart document for user to coding on this library. The document should be written in markdown format.
Show plenty of self-contained code examples with comments.

{{code}}
""",
            role="user",
        ),
        GenerateTemplate(role="assistant"),
    ],
)

configuration = AnthropicClaudeModelConfiguration(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
parameter = AnthropicClaudeModelParameters(
    model_name="claude-3-5-sonnet-20241022",
    temperature=1,
    max_tokens=8192,
)
model = AnthropicClaudeModel(configuration=configuration)

# load all files in examples and tests with its name in text
text = ""
for file in tqdm(list(glob.glob("examples/**/*.py", recursive=True))):
    text += f"Example filename: {file}\n"
    text += f"```python\n{open(file, 'r').read()}\n```\n"

for file in tqdm(list(glob.glob("tests/**/*.py", recursive=True))):
    text += f"Test filename: {file}\n"
    text += f"```python\n{open(file, 'r').read()}\n```\n"

# add README.md content
text += f"```README\n{open('README.md', 'r').read()}\n```\n"

# add docs *.md content
for file in tqdm(list(glob.glob("docs/*.md", recursive=False))):
    text += f"Docs filename: {file}\n"
    text += f"```markdown\n{open(file, 'r').read()}\n```\n"


runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

initial_session = Session()
initial_session.append(Message(content="", metadata={"code": text}))
session = runner.run(session=initial_session)
last_message = session.get_last_message().content

print(last_message)
