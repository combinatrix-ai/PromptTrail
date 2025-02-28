# This example take a python script path and proofread the comments in it.

import os

import click

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.openai import OpenAIConfig, OpenAIModel

templates = LinearTemplate(
    [
        SystemTemplate(
            content="""
You're an AI proofreader that help user to fix markdown.
You're given python script content by the user.
You must fix the missspellings in the comments.
You only emit the corrected python script. No explanation is needed.
""",
        ),
        UserTemplate(
            content="{{content}}",
        ),
        AssistantTemplate(),
    ],
)

configuration = OpenAIConfig(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=8000,
)
model = OpenAIModel(configuration=configuration)

runner = CommandLineRunner(
    model=model,
    template=templates,
    user_interface=CLIInterface(),
)


@click.command()
@click.option("--load_file", type=click.Path(exists=True))
def main(
    load_file: str,
):
    # show all log levels
    import logging

    logging.basicConfig(level=logging.INFO)

    load_file_content = open(load_file, "r")
    initial_session = Session(
        metadata={
            "content": load_file_content.read(),
        }
    )
    session = runner.run(session=initial_session)
    last_message = session.get_last_message()
    message = last_message.content
    print(message)
    # add EOF if not exists
    if message[-1] != "\n":
        message += "\n"
    save_file_io = open(load_file, "w")
    save_file_io.write(last_message.content)
    save_file_io.close()


if __name__ == "__main__":
    main()
