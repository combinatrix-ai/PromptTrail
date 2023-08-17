import logging
import re

from prompttrail.agent.core import State
from prompttrail.util import hook_logger

from .core import TransformHook


class ExtractMarkdownCodeBlockHook(TransformHook):
    def __init__(self, key: str, lang: str):
        self.key = key
        self.lang = lang

    def hook(self, state: State) -> State:
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
        self.key = key
        self.code_key = code

    def hook(self, state: State) -> State:
        python_segment = state.data[self.code_key]
        if python_segment is None:
            # TODO: Hook must know which template it is in, to let user know which template is failing.
            hook_logger(
                self,
                state,
                f"No code block found for key {self.key}",
                level=logging.WARNING,
            )
            return state
        # rewrite python segment
        # remove leading spaces if all lines have the same number of leading spaces
        lines = python_segment.splitlines()
        if len(lines) > 0:
            leading_spaces = [len(line) - len(line.lstrip()) for line in lines]
            if len(set(leading_spaces)) == 1:
                python_segment = "\n".join(
                    [line[leading_spaces[0] :] for line in lines]
                )
        try:
            answer = eval(python_segment)  # TODO: security check
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
