import unittest

from prompttrail.agent import Session
from prompttrail.agent.hooks import EvaluatePythonCodeHook, ExtractMarkdownCodeBlockHook
from prompttrail.core import Message


class TestExtractMarkdownCodeBlockHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        session.append(
            Message(content="```python\nprint('Hello, World!')```", role="assistant")
        )
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        session = hook.hook(session)
        self.assertEqual(session.metadata["code"], "print('Hello, World!')")

    def test_hook_no_code_block(self):
        session = Session()
        session.append(Message(content="This  is a regular message", role="assistant"))
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        session = hook.hook(session)
        self.assertIsNone(session.metadata["code"])


class TestEvaluatePythonCodeHook(unittest.TestCase):
    def test_hook(self):
        session = Session(metadata={"code": "print('Hello, World!')"})
        session.append(Message(content="blah", role="assistant"))
        hook = EvaluatePythonCodeHook("answer", "code")
        session = hook.hook(session)
        self.assertEqual(session.metadata["answer"], None)

    def test_hook_no_code_block(self):
        session = Session(metadata={})
        session.append(Message(content="blah", role="assistant"))
        hook = EvaluatePythonCodeHook("answer", "code")
        with self.assertRaises(KeyError):
            session = hook.hook(session)


if __name__ == "__main__":
    unittest.main()
