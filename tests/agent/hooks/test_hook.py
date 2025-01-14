import logging
import unittest

from prompttrail.agent import Session
from prompttrail.agent.hooks import BooleanHook, GenerateChatHook, Hook, TransformHook

logger = logging.getLogger(__name__)


class TestHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        hook = Hook()
        with self.assertRaises(NotImplementedError):
            hook.hook(session)


class TestTransformHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        transform_hook = TransformHook(lambda x: x)
        result = transform_hook.hook(session)
        self.assertEqual(result, session)


class TestBooleanHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        boolean_hook = BooleanHook(lambda x: True)
        result = boolean_hook.hook(session)
        self.assertTrue(result)


class TestGenerateChatHook(unittest.TestCase):
    def test_hook(self):
        session = Session()
        key = "test_key"
        generate_chat_hook = GenerateChatHook(key)
        with self.assertRaises(ValueError):
            generate_chat_hook.hook(session)


if __name__ == "__main__":
    unittest.main()
