import re
from dataclasses import dataclass
from typing import List

from prompttrail.core import Metadata, Session

from ._core import MetadataTransformer


@dataclass
class CodeBlock:
    """A code block extracted from markdown."""

    lang: str
    code: str


def extract_code_blocks(markdown: str) -> List[CodeBlock]:
    """Extract code blocks from markdown content.

    Args:
        markdown: Markdown content to extract code blocks from

    Returns:
        List of extracted code blocks
    """
    pattern = r"```(\w+)\n(.*?)```"
    matches = re.finditer(pattern, markdown, re.DOTALL)
    return [CodeBlock(lang=m.group(1), code=m.group(2).strip()) for m in matches]


class ExtractMarkdownCodeBlockHook(MetadataTransformer):
    """A hook that extracts code blocks from markdown content."""

    def __init__(self, key: str, lang: str):
        """Initialize the hook.

        Args:
            key: Key to store the extracted code block in metadata
            lang: Programming language of the code block to extract
        """
        super().__init__()
        self.key = key
        self.lang = lang

    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        """Extract code block from last message content.

        Args:
            metadata: Current metadata
            session: Current conversation session

        Returns:
            Updated session with extracted code stored in metadata[key]
        """
        if not session.messages:
            raise KeyError("No messages in session")

        message = session.get_last()
        code_blocks = extract_code_blocks(message.content)

        if not code_blocks:
            self.warning("No code blocks found in message content: %s", message.content)
            metadata[self.key] = None
            return metadata

        if self.lang:
            code_blocks = [block for block in code_blocks if block.lang == self.lang]
            if not code_blocks:
                metadata[self.key] = None
                return metadata

        metadata[self.key] = code_blocks[0].code
        return metadata


class EvaluatePythonCodeHook(MetadataTransformer):
    """A hook that evaluates Python code blocks."""

    def __init__(self, key: str, code: str):
        """Initialize the hook.

        Args:
            key: Key to store evaluation result in metadata
            code: Key of code block to evaluate from metadata
        """
        self.key = key
        self.code_key = code

    def process_metadata(self, metadata: Metadata, session: Session) -> Metadata:
        """Evaluate Python code from metadata and store result.

        Args:
            metadata: Current metadata
            session: Current conversation session

        Returns:
            Updated session with evaluation result stored in metadata[key]

        Raises:
            KeyError: If code_key is not found in metadata
        """
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
        return metadata
