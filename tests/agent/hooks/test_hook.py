import logging
import unittest

from prompttrail.agent import State
from prompttrail.agent.hooks import BooleanHook, GenerateChatHook, Hook, TransformHook

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


class TestGenerateChatHook(unittest.TestCase):
    def test_hook(self):
        state = State()
        key = "test_key"
        generate_chat_hook = GenerateChatHook(key)
        with self.assertRaises(ValueError):
            generate_chat_hook.hook(state)


if __name__ == "__main__":
    unittest.main()
