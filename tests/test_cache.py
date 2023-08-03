import unittest

from prompttrail.cache import LRUCacheProvider
from prompttrail.core import Message, Parameters, Session


class TestLRUCacheProvider(unittest.TestCase):
    def test_add(self):
        cache_provider = LRUCacheProvider(3)
        session = Session()
        message = Message(content="Test message", sender="user")
        cache_provider.add(session, message)
        self.assertEqual(cache_provider.cache[session], message)

    def test_search(self):
        cache_provider = LRUCacheProvider(3)
        session = Session()
        message_1 = Message(content="Test message", sender="user")
        message_2 = Message(content="Test message", sender="user")
        cache_provider.add(session, message_1)
        result = cache_provider.search(Parameters(), session)
        self.assertEqual(result, message_2)


if __name__ == "__main__":
    unittest.main()
