import unittest
import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional

from prompttrail.agent.core import FlowState
from prompttrail.agent.hook.core import AskUserHook, BooleanHook, GenerateChatHook, Hook, IfJumpHook, JumpHook, TransformHook
from prompttrail.agent.template import TemplateId

if TYPE_CHECKING:
    from prompttrail.agent.template import TemplateId

logger = logging.getLogger(__name__)


class TestHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        hook = Hook()
        with self.assertRaises(NotImplementedError):
            hook.hook(flow_state)


class TestTransformHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        transform_hook = TransformHook(lambda x: x)
        result = transform_hook.hook(flow_state)
        self.assertEqual(result, flow_state)


class TestBooleanHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        boolean_hook = BooleanHook(lambda x: True)
        result = boolean_hook.hook(flow_state)
        self.assertTrue(result)


class TestJumpHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        jump_hook = JumpHook(lambda x: None)
        with self.assertRaises(NotImplementedError):
            jump_hook.hook(flow_state)


class TestIfJumpHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        true_template = TemplateId()
        false_template = TemplateId()
        if_jump_hook = IfJumpHook(lambda x: True, true_template, false_template)
        result = if_jump_hook.hook(flow_state)
        self.assertEqual(result, true_template)


class TestAskUserHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        key = "test_key"
        description = "test_description"
        default = "test_default"
        ask_user_hook = AskUserHook(key, description, default)
        with self.assertRaises(ValueError):
            ask_user_hook.hook(flow_state)


class TestGenerateChatHook(unittest.TestCase):
    def test_hook(self):
        flow_state = FlowState()
        key = "test_key"
        generate_chat_hook = GenerateChatHook(key)
        with self.assertRaises(ValueError):
            generate_chat_hook.hook(flow_state)


if __name__ == "__main__":
    unittest.main()