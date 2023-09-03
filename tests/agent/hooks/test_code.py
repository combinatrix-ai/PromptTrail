import unittest

from prompttrail.agent import State
from prompttrail.agent.hooks import EvaluatePythonCodeHook, ExtractMarkdownCodeBlockHook
from prompttrail.core import Message


class TestExtractMarkdownCodeBlockHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        state.session_history.messages.append(Message(content="```python\nprint('Hello, World!')```", sender="assistant"))  # type: ignore
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        state = hook.hook(state)
        self.assertEqual(state.data["code"], "print('Hello, World!')")

    def test_hook_no_code_block(self):
        state = State()
        state.session_history.messages.append(Message(content="This  is a regular message", sender="assistant"))  # type: ignore
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        state = hook.hook(state)
        self.assertIsNone(state.data["code"])


class TestEvaluatePythonCodeHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        state.data["code"] = "print('Hello, World!')"
        hook = EvaluatePythonCodeHook("answer", "code")
        state = hook.hook(state)
        self.assertEqual(state.data["answer"], None)

    def test_hook_no_code_block(self):
        state = State()
        hook = EvaluatePythonCodeHook("answer", "code")
        with self.assertRaises(KeyError):
            # TODO: Hook should raise error?
            state = hook.hook(state)


if __name__ == "__main__":
    unittest.main()
