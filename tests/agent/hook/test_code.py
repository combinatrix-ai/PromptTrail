import unittest

from prompttrail.agent.core import FlowState
from prompttrail.agent.hook.code import (
    EvaluatePythonCodeHook,
    ExtractMarkdownCodeBlockHook,
)
from prompttrail.core import Message


class TestExtractMarkdownCodeBlockHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        flow_state.session_history.messages.append(Message(content="```python\nprint('Hello, World!')```", sender="assistant"))  # type: ignore
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        flow_state = hook.hook(flow_state)
        self.assertEqual(flow_state.data["code"], "print('Hello, World!')")

    def test_hook_no_code_block(self):
        flow_state = FlowState()
        flow_state.session_history.messages.append(Message(content="This  is a regular message", sender="assistant"))  # type: ignore
        hook = ExtractMarkdownCodeBlockHook("code", "python")
        flow_state = hook.hook(flow_state)
        self.assertIsNone(flow_state.data["code"])


class TestEvaluatePythonCodeHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        flow_state.data["code"] = "print('Hello, World!')"
        hook = EvaluatePythonCodeHook("answer", "code")
        flow_state = hook.hook(flow_state)
        self.assertEqual(flow_state.data["answer"], None)

    def test_hook_no_code_block(self):
        flow_state = FlowState()
        hook = EvaluatePythonCodeHook("answer", "code")
        flow_state = hook.hook(flow_state)
        self.assertEqual(flow_state.data["answer"], None)


if __name__ == "__main__":
    unittest.main()
