import unittest

from prompttrail.core import Message, Parameters, Session
from prompttrail.core.cache import LRUCacheProvider
from prompttrail.models.openai import OpenAIConfig, OpenAIModel, OpenAIParam


class TestLRUCacheProvider(unittest.TestCase):
    def test_add(self):
        cache_provider = LRUCacheProvider(3)
        session = Session()
        message = Message(content="Test message", role="user")
        cache_provider.add(session, message)
        self.assertEqual(cache_provider.cache[session], message)

    def test_search(self):
        cache_provider = LRUCacheProvider(3)
        session = Session()
        message_1 = Message(content="Test message", role="user")
        message_2 = Message(content="Test message", role="user")
        cache_provider.add(session, message_1)
        result = cache_provider.search(Parameters(), session)
        self.assertEqual(result, message_2)

    def search_invalid_session(self):
        cache_provider = LRUCacheProvider(3)
        session = Session()
        message = Message(content="Test message", role="user")
        cache_provider.add(session, message)
        result = cache_provider.search(Parameters(), Session())
        self.assertIsNone(result)

    def test_cache_in_models(self):
        # API key is invalid.
        api_key = ""
        cache_provider = LRUCacheProvider(1)
        message_in = Message(content="Hey", role="user")
        message_out = Message(content="HeyHey", role="user")
        session = Session(messages=[message_in])
        cache_provider.add(session, message_out)
        config = OpenAIConfig(api_key=api_key, cache_provider=cache_provider)
        parameters = OpenAIParam(
            model_name="gpt-4o-mini", max_tokens=1000, temperature=0
        )
        model = OpenAIModel(configuration=config)
        # But, cache is called, so no error is raised.
        message = model.send(parameters=parameters, session=session)
        # The message is the same as the one in the cache.
        self.assertEqual(message.content, message_out.content)


if __name__ == "__main__":
    unittest.main()
