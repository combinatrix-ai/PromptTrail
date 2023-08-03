import logging
import re

from prompttrail.flow.core import FlowState
from prompttrail.util import hook_logger

from .core import TransformHook


class ExtractMarkdownCodeBlockHook(TransformHook):
    def __init__(self, key: str, lang: str):
        self.key = key
        self.lang = lang

    def hook(self, flow_state: FlowState) -> FlowState:
        markdown = flow_state.get_last_message().content
        match = re.search(r"```" + self.lang + r"\n(.+?)```", markdown, re.DOTALL)
        if match:
            code_block = match.group(1)
        else:
            code_block = None
        flow_state.data[self.key] = code_block
        return flow_state


class EvaluatePythonCodeHook(TransformHook):
    def __init__(self, key: str, code: str):
        self.key = key
        self.code_key = code

    def hook(self, flow_state: FlowState) -> FlowState:
        python_segment = flow_state.data[self.code_key]
        if python_segment is None:
            # TODO: Hook must know which template it is in, to let user know which template is failing.
            hook_logger(
                self,
                flow_state,
                f"No code block found for key {self.key}",
                level=logging.WARNING,
            )
            return flow_state
        answer = eval(python_segment)  # TODO: security check
        flow_state.data[self.key] = answer
        return flow_state
