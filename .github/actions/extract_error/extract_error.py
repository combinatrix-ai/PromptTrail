import argparse
import csv
import os
from io import StringIO

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate
from prompttrail.agent.templates.openai import (
    OpenAIGenerateTemplate,
    OpenAIMessageTemplate,
)
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)


def extract(error_message: str):
    template = LinearTemplate(
        [
            OpenAIMessageTemplate(
                role="system",
                content="""
You're an AI assistant that helps software engineer to understand test error message.
Your input is error message that emitted by test codes.
Your output is the summary csv of the error message.
Each line in the summary csv descributes each error in the error message.
the summary csv first column indicates where the error occurred, second column is summary of error.
You emit ONLY the summary csv. No explanation is needed.
""",
            ),
            OpenAIMessageTemplate(
                role="user",
                content=error_message,
            ),
            OpenAIGenerateTemplate(role="assistant"),
        ]
    )
    runner = CommandLineRunner(
        model=OpenAIChatCompletionModel(
            configuration=OpenAIModelConfiguration(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
        ),
        parameters=OpenAIModelParameters(model_name="gpt-3.5-turbo-0301"),
        template=template,
        user_interaction_provider=UserInteractionTextCLIProvider(),
    )
    state = runner.run()
    last_message_content = state.get_last_message().content
    # remove response message header
    csv_str = "\n".join(last_message_content.splitlines()[2:])
    with StringIO() as f:
        f.write(csv_str)
        f.seek(0)
        csv_content = list(csv.reader(f))
        content = "\n".join([f"| {row[0]} | {row[1]} |" for row in csv_content])
    return f"""# Error Summaries

| where | summary |
| ----- | ------- |
{content}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("error_filepath")
    parser.add_argument("output_filepath")
    args = parser.parse_args()
    with open(args.error_filepath) as f:
        error_message = f.read()
    with open(args.output_filepath, "w") as f:
        f.write(extract(error_message))


if __name__ == "__main__":
    main()
