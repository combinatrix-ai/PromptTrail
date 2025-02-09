import unittest

from prompttrail.agent.session_transformers import (
    DangerouslyEvaluatePythonCode,
    ExtractMarkdownCodeBlock,
)
from prompttrail.core import Message, Session


class TestExtractMarkdownCodeBlockHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        session.append(
            Message(content="```python\nprint('Hello, World!')```", role="assistant")
        )
        hook = ExtractMarkdownCodeBlock("code", "python")
        session = hook.process(session)
        self.assertEqual(session.metadata["code"], "print('Hello, World!')")

    def test_hook_no_code_block(self):
        session = Session()
        session.append(Message(content="This  is a regular message", role="assistant"))
        hook = ExtractMarkdownCodeBlock("code", "python")
        session = hook.process(session)
        self.assertIsNone(session.metadata["code"])


class TestEvaluatePythonCodeHook(unittest.TestCase):
    def test_hook(self):
        session = Session(metadata={"code": "print('Hello, World!')"})
        session.append(Message(content="blah", role="assistant"))
        hook = DangerouslyEvaluatePythonCode("answer", "code")
        session = hook.process(session)
        self.assertEqual(session.metadata["answer"], None)

    def test_hook_no_code_block(self):
        session = Session(metadata={})
        session.append(Message(content="blah", role="assistant"))
        hook = DangerouslyEvaluatePythonCode("answer", "code")
        with self.assertRaises(KeyError):
            session = hook.process(session)


if __name__ == "__main__":
    unittest.main()
