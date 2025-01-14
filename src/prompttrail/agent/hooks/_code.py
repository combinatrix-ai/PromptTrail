import logging
import re

from prompttrail.agent import Session
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

    def hook(self, session: Session) -> Session:
        """
        Extract the code block from the last message in the session.

        Args:
            session (Session): The current session.

        Returns:
            Session: The updated session.
        """
        markdown = session.get_last().content
        match = re.search(r"```" + self.lang + r"\n(.+?)```", markdown, re.DOTALL)
        if match:
            code_block = match.group(1)
        else:
            code_block = None
        metadata = session.get_latest_metadata()
        metadata[self.key] = code_block
        return session


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

    def hook(self, session: Session) -> Session:
        """
        Evaluate the Python code block and store the result in the session.

        Args:
            session (Session): The current session.

        Returns:
            Session: The updated session.
        """
        metadata = session.get_latest_metadata()
        if self.code_key not in metadata:
            raise KeyError(f"Code key {self.code_key} not found in metadata")
        python_segment = metadata[self.code_key]
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
                session,
                f"Failed to evaluate python code: {python_segment}",
                level=logging.WARNING,
            )
            raise e
        metadata[self.key] = answer
        return session
