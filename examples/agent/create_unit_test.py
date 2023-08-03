import os
import sys
from typing import List, Optional

import click
from prompttrail.agent.core import FlowState
from prompttrail.agent.template import LinearTemplate
from prompttrail.provider.openai import (
    OpenAIChatCompletionModel,
    OpenAIMessageTemplate as MessageTemplate,
    OpenAIGenerateTemplate as GenerateTemplate,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)
from prompttrail.agent.runner import CommandLineRunner

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI assistant that help user to create a unit test for given code.
Your output is written to the file and will be executed by the user. Therefore, you only emit the test code.
You emit the whole file content. You must use unittest module.
Your input looks like this:

* related files:
** file1.py

import xxxx
...

** file2.py

import xxxx
...

* File to write test:
import file1, file2

class A(object):
...

* desciption:

write a test for class A only
...

Your output looks like this:

import unittest
import file1, file2

class test_A(unittest.TestCase):
...


Again, dont forget you only emit the test code. No explanation is needed. But can add comments in code.
""",
            role="system",
        ),
        MessageTemplate(
            content="""
* related files:
{% for filename, file in context_files.items() %}
** {{filename}}
{{file}}
{% endfor %}

* File to write test:

{{code}}

* description:

{{description}}

""",
            role="user",
        ),
        GenerateTemplate(role="assistant"),
    ],
)

configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=5000
)
model = OpenAIChatCompletionModel(configuration=configuration)

runner = CommandLineRunner(model=model, parameters=parameter, templates=[templates])


@click.command()
@click.option("--load_file", type=click.Path(exists=True))
@click.option("--save_file", type=click.Path(exists=False))
@click.option("--context_files", type=click.Path(exists=True), multiple=True)
@click.option("--description", type=str, default=None)
def main(
    load_file: str,
    save_file: str,
    context_files: List[str],
    description: Optional[str] = None,
):
    # show all log levels
    import logging

    logging.basicConfig(level=logging.DEBUG)

    load_file_content = open(load_file, "r")
    context_file_contents = {x: open(x, mode="r").read() for x in context_files}
    initial_state = FlowState(
        data={
            "code": load_file_content.read(),
            "context_files": context_file_contents,
            "description": description,
        }
    )
    flow_state = runner.run(flow_state=initial_state)
    last_message = flow_state.get_last_message()
    print(
        last_message.content.content
    )  # TODO: maybe bug? (last_message is not message?)
    if len(sys.argv) > 2:
        save_file_io = open(save_file, "w")
        save_file_io.write(last_message.content.content)
        save_file_io.close()


if __name__ == "__main__":
    main()

# ./.venv/bin/python -m examples.agent.create_unit_test \
# --load_file src/prompttrail/mock.py --context_files src/prompttrail/core.py \
# --save_file tests/test_mock.py