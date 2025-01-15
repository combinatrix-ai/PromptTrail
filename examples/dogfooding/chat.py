import glob
import os

from tqdm import tqdm

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    GenerateTemplate,
    LinearTemplate,
    LoopTemplate,
    MessageTemplate,
    UserInputTextTemplate,
)
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.core import Session
from prompttrail.models.anthropic import (
    AnthropicClaudeModel,
    AnthropicClaudeModelConfiguration,
    AnthropicClaudeModelParameters,
)

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're given source code and test scripts and documents for a library, PromptTrail as below:
{{code}}
Discuss the question with user. User is the author of this library, who want to improve the design, implementation, and documentation of the library.
""",
            role="user",
        ),
        LoopTemplate(
            [
                UserInputTextTemplate(
                    role="user",
                    description="Input:",
                ),
                GenerateTemplate(role="assistant"),
            ]
        ),
    ],
)

configuration = AnthropicClaudeModelConfiguration(
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
parameter = AnthropicClaudeModelParameters(
    model_name="claude-3-sonnet-20240229",
    temperature=1,
    max_tokens=4096,
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

initial_session = Session(initial_metadata={"code": text})
runner.run(session=initial_session)
