import logging
import unittest

from prompttrail.agent.core import State
from prompttrail.agent.hook.core import (
    BooleanHook,
    GenerateChatHook,
    Hook,
    IfJumpHook,
    JumpHook,
    TransformHook,
)
from prompttrail.agent.template import TemplateId

logger = logging.getLogger(__name__)


class TestHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        hook = Hook()
        with self.assertRaises(NotImplementedError):
            hook.hook(state)


class TestTransformHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        transform_hook = TransformHook(lambda x: x)
        result = transform_hook.hook(state)
        self.assertEqual(result, state)


class TestBooleanHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        boolean_hook = BooleanHook(lambda x: True)
        result = boolean_hook.hook(state)
        self.assertTrue(result)


class TestJumpHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        jump_hook = JumpHook(lambda x: None)
        with self.assertRaises(NotImplementedError):
            jump_hook.hook(state)


class TestIfJumpHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        true_template = TemplateId()
        false_template = TemplateId()
        if_jump_hook = IfJumpHook(lambda x: True, true_template, false_template)
        result = if_jump_hook.hook(state)
        self.assertEqual(result, true_template)


# AskUserHook uses input() which is not testable for now!
# class TestAskUserHook(unittest.TestCase):
#     def test_hook(self):
#         state = State()
#         key = "test_key"
#         description = "test_description"
#         default = "test_default"
#         ask_user_hook = AskUserHook(key, description, default)
#         with self.assertRaises(ValueError):
#             ask_user_hook.hook(state)


class TestGenerateChatHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        key = "test_key"
        generate_chat_hook = GenerateChatHook(key)
        with self.assertRaises(ValueError):
            generate_chat_hook.hook(state)


if __name__ == "__main__":
    unittest.main()
