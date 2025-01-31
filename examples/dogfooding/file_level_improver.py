# flake8: noqa: E402

import glob
import json
import os
import sys

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.session_transformers import ResetMetadata
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.tools.builtin import CreateOrOverwriteFile, ReadFile
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel

sys.path.append(os.path.abspath("."))

from examples.dogfooding.dogfooding_tools import ReadImportantFiles, RunTest

configuration = AnthropicConfig(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model_name="claude-3-5-sonnet-latest",
    temperature=1,
    max_tokens=4096,
)
model = AnthropicModel(configuration=configuration)


paths = list(glob.glob("src/**/*.py", recursive=True))

for path in paths:
    templates = LinearTemplate(
        [
            SystemTemplate(
                content="""
    Please help the user improve a LLM library, PromptTrail. 
    You're tasked with improving the conciseness and readability of the codebase.

    Rules:
    - Don't change the behavior of the code.
    - Don't remove comments. You can add comments if necessary. Write proper docstrings.

    You're firstly given the content of all files first. 
    After that, you will be given the path of the path of the file and content of the file.

    You answer should be the content of the file after the improvement surrounded by python markdown tag.
    If you don't want to make any changes, just return the original content.

    Example Input:
    core/__init__.py

    from * import *

    Good Output (Only emit the content of the file):
    ``python
    from * import *
    ```

    Bad Output (Don't include your comment):
    Unspecific import statement is not generally recommended. However, it's okay in this case.
    ```python
    from * import *
    ```

    Bad Output (Don't include your comment):
    Unspecific import statement is not generally recommended. Therefore, it's better to import specific modules.
    ```python
    import models
    imoprt utils
    ```
    
    Files:
    {{important_files}}
    """,
                after_transform=ResetMetadata("important_files"),
            ),
            UserTemplate(
                content="{{content}}",
            ),
            AssistantTemplate(),
        ]
    )
    runner = CommandLineRunner(
        model=model,
        template=templates,
        user_interface=CLIInterface(),
    )

    initial_session = Session(
        metadata={
            "important_files": ReadImportantFiles().execute().content["result"],
            # all python files in src directory
            "content": "filepath: " + path + "\n\n" + open(path).read(),
        }
    )

    backup_content = json.loads(ReadFile().execute(path=path).content)["content"]
    session = runner.run(session=initial_session)
    # write the content to the file
    content = session.messages[-1].content.strip()

    # if content is markdown content, remove the markdown tags
    if content.startswith("```") and content.endswith("```"):
        # remove first line
        content = content[content.find("\n") + 1 :].strip()
        # remove last line
        content = content[: content.rfind("\n")].strip()

    if content.strip() == backup_content.strip():
        print("No changes made to the file.")
        continue
    print("Writing the new content to the file.")
    CreateOrOverwriteFile().execute(path=path, content=content)
    # run test
    test_result = RunTest().execute()
    print(test_result)
    if not json.loads(test_result.content)["status"] == "success":
        print("Test failed. Restoring the file.")
        CreateOrOverwriteFile().execute(path=path, content=backup_content)
