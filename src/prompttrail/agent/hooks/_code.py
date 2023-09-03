import logging
import re

from prompttrail.agent import State
from prompttrail.agent.hooks._core import TransformHook
from prompttrail.core.utils import hook_logger


class ExtractMarkdownCodeBlockHook(TransformHook):
    def __init__(self, key: str, lang: str):
        """
        Extract a code block from a markdown content.

        Args:
            key (str): The key to store the extracted code block.
            lang (str): The language of the code block.
        """
        self.key = key
        self.lang = lang

    def hook(self, state: State) -> State:
        """
        Extract the code block from the last message in the state.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.
        """
        markdown = state.get_last_message().content
        match = re.search(r"```" + self.lang + r"\n(.+?)```", markdown, re.DOTALL)
        if match:
            code_block = match.group(1)
        else:
            code_block = None
        state.data[self.key] = code_block
        return state


class EvaluatePythonCodeHook(TransformHook):
    def __init__(self, key: str, code: str):
        """
        Evaluate a Python code block and store the result.

        Args:
            key (str): The key to store the evaluated result.
            code (str): The key of the code block to evaluate.
        """
        self.key = key
        self.code_key = code

    def hook(self, state: State) -> State:
        """
        Evaluate the Python code block and store the result in the state.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.
        """
        python_segment = state.data[self.code_key]
        if python_segment is None:
            hook_logger(
                self,
                state,
                f"No code block found for key {self.key}",
                level=logging.WARNING,
            )
            return state
        lines = python_segment.splitlines()
        if len(lines) > 0:
            leading_spaces = [len(line) - len(line.lstrip()) for line in lines]
            if len(set(leading_spaces)) == 1:
                python_segment = "\n".join(
                    [line[leading_spaces[0] :] for line in lines]
                )
        try:
            answer = eval(python_segment)
        except Exception as e:
            hook_logger(
                self,
                state,
                f"Failed to evaluate python code: {python_segment}",
                level=logging.WARNING,
            )
            raise e
        state.data[self.key] = answer
        return state
