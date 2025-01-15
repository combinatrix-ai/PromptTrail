import unittest

from prompttrail.agent import Session
from prompttrail.agent.hooks import EvaluatePythonCodeHook, ExtractMarkdownCodeBlockHook
from prompttrail.core import Message


class TestExtractMarkdownCodeBlockHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        session.append(
            Message(content="```python\nprint('Hello, World!')```", sender="assistant")
        )
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        session = hook.hook(session)
        self.assertEqual(
            session.get_latest_metadata()["code"], "print('Hello, World!')"
        )

    def test_hook_no_code_block(self):
        session = Session()
        session.append(
            Message(content="This  is a regular message", sender="assistant")
        )
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        session = hook.hook(session)
        self.assertIsNone(session.get_latest_metadata()["code"])


class TestEvaluatePythonCodeHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        session.append(
            Message(content="blah", metadata={"code": "print('Hello, World!')"})
        )
        hook = EvaluatePythonCodeHook("answer", "code")
        session = hook.hook(session)
        self.assertEqual(session.get_latest_metadata()["answer"], None)

    def test_hook_no_code_block(self):
        session = Session()
        session.append(Message(content="blah", metadata={}))
        hook = EvaluatePythonCodeHook("answer", "code")
        with self.assertRaises(KeyError):
            session = hook.hook(session)


if __name__ == "__main__":
    unittest.main()
