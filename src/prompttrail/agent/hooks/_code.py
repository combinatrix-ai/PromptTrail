import re

from prompttrail.agent import Session
from prompttrail.agent.hooks._core import TransformHook


class ExtractMarkdownCodeBlockHook(TransformHook):
    """A hook that extracts code blocks from markdown content."""

    def __init__(self, key: str, lang: str):
        """Initialize the hook.

        Args:
            key: Key to store the extracted code block in metadata
            lang: Programming language of the code block to extract
        """
        self.key = key
        self.lang = lang

    def hook(self, session: Session) -> Session:
        """Extract code block from last message content.

        Args:
            session: Current conversation session

        Returns:
            Updated session with extracted code stored in metadata[key]
        """
        markdown = session.get_last().content
        pattern = f"```{self.lang}\n(.+?)```"
        match = re.search(pattern, markdown, re.DOTALL)
        code_block = match.group(1) if match else None
        session.get_latest_metadata()[self.key] = code_block
        return session


class EvaluatePythonCodeHook(TransformHook):
    """A hook that evaluates Python code blocks."""

    def __init__(self, key: str, code: str):
        """Initialize the hook.

        Args:
            key: Key to store evaluation result in metadata
            code: Key of code block to evaluate from metadata
        """
        self.key = key
        self.code_key = code

    def hook(self, session: Session) -> Session:
        """Evaluate Python code from metadata and store result.

        Args:
            session: Current conversation session

        Returns:
            Updated session with evaluation result stored in metadata[key]

        Raises:
            KeyError: If code_key is not found in metadata
        """
        metadata = session.get_latest_metadata()
        if self.code_key not in metadata:
            raise KeyError(f"Code key {self.code_key} not found in metadata")

        python_segment = metadata[self.code_key]

        # Normalize indentation if all lines have same leading spaces
        lines = python_segment.splitlines()
        if lines:
            leading_spaces = [len(line) - len(line.lstrip()) for line in lines]
            if len(set(leading_spaces)) == 1:
                python_segment = "\n".join(line[leading_spaces[0] :] for line in lines)

        try:
            answer = eval(python_segment)
        except Exception as e:
            self.error(f"Failed to evaluate python code: {python_segment}")
            raise e

        metadata[self.key] = answer
        return session
