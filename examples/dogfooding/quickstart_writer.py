import glob
import os

from tqdm import tqdm

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import AssistantTemplate, LinearTemplate, UserTemplate
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel

templates = LinearTemplate(
    [
        UserTemplate(
            content="""
You're given examples and test scripts and documents for a library, PromptTrail.
Write a quickstart document for user to coding on this library. The document should be written in markdown format.
Show plenty of self-contained code examples with comments.

{{code}}
""",
        ),
        AssistantTemplate(),
    ],
)

configuration = AnthropicConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model_name="claude-3-5-sonnet-latest",
    temperature=1,
    max_tokens=8192,
)
model = AnthropicModel(configuration=configuration)

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
    template=templates,
    user_interface=CLIInterface(),
)

initial_session = Session(metadata={"code": text})
session = runner.run(session=initial_session)
last_message = session.get_last_message().content

print(last_message)
