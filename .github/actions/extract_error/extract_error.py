import argparse
import os
from prompttrail.agent.templates import LinearTemplate
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.models.openai import OpenAIChatCompletionModel, OpenAIModelConfiguration, OpenAIModelParameters
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider


def extract(error_message: str):
    return f"""# Error Summary

TODO: impl summarization of error_message by PromptTail.
error_mesasge len: {len(error_message)}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('error_filepath') 
    parser.add_argument('output_filepath') 
    args = parser.parse_args()
    with open(args.error_filepath) as f:
        error_message = f.read()
    with open(args.output_filepath, 'w') as f:
        f.write(extract(error_message))


if __name__ == "__main__":
    main()
