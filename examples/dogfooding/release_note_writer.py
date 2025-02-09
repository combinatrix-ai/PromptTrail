import os
from typing import Optional

import click
import requests

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates._control import LinearTemplate, LoopTemplate
from prompttrail.agent.templates._core import (
    AssistantTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.google import GoogleConfig, GoogleModel


def get_patch_url(last_tag: str, current_tag: str = "main") -> str:
    return f"https://github.com/combinatrix-ai/PromptTrail/compare/{last_tag}...{current_tag}.patch"


@click.command()
@click.option("--last-tag", required=True, help="The last tag")
@click.option("--current-tag", default="main", help="The current tag")
@click.option("--non-interactive", is_flag=True, help="Disable interactive mode")
@click.option("--output", default=None, help="Output file")
def write_release_note(
    last_tag: str, current_tag: str, non_interactive: bool, output: Optional[str]
):
    patch_url = get_patch_url(last_tag, current_tag)
    print(patch_url)
    response = requests.get(patch_url).text

    if not response:
        print("No changes found between the tags")
        raise click.Abort("No changes found between the tags")

    if non_interactive:
        templates = LinearTemplate(
            [
                SystemTemplate(
                    """
                    You're given a patch file. Please write the release note based on the changes in the patch file.
                    This is automated API call to generate release note. Therefore, only emit the content of the release markdown.
                    You don't need ```markdown``` code block.
                    """
                ),
                # The patch can contain {{ }} which is a jinja template syntax. We need to disable jinja for this task
                # And I don't want to pass it as Metadata, as it's too long and mess the terminal
                UserTemplate(response, disable_jinja=True),
                AssistantTemplate(),
            ]
        )
    else:
        templates = LinearTemplate(
            [
                SystemTemplate(
                    "You're given a patch file. Please write the release note based on the changes in the patch file."
                ),
                # The patch can contain {{ }} which is a jinja template syntax. We need to disable jinja for this task
                # And I don't want to pass it as Metadata, as it's too long and mess the terminal
                UserTemplate(response, disable_jinja=True),
                LoopTemplate(
                    [AssistantTemplate(), UserTemplate(description="Input: ")]
                ),
            ]
        )

    model = GoogleModel(
        configuration=GoogleConfig(
            # As changes are long, gemini is best suited for this task
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            api_key=os.environ["GOOGLE_CLOUD_API_KEY"],
        )
    )

    runner = CommandLineRunner(
        model=model, template=templates, user_interface=CLIInterface()
    )

    session = runner.run(session=Session(), debug_mode=True)

    if non_interactive:
        release_note = session.messages[-1].content
        if output:
            with open(output, "w") as f:
                f.write(release_note)
        print("Release note:")
        print(session.messages[-1].content)


if __name__ == "__main__":
    write_release_note()
