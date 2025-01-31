import unittest

from prompttrail.core import Config, Message, Session
from prompttrail.core.cache import LRUCacheProvider
from prompttrail.core.mocks import OneTurnConversationMockProvider
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


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
        result = cache_provider.search(Config(model_name="test"), session)
        self.assertEqual(result, message_2)

    def search_invalid_session(self):
        cache_provider = LRUCacheProvider(3)
        session = Session()
        message = Message(content="Test message", role="user")
        cache_provider.add(session, message)
        result = cache_provider.search(Config(model_name="test"), Session())
        self.assertIsNone(result)

    def test_cache_in_models(self):
        """Test model behavior with cache provider"""
        message_in = Message(content="Hey", role="user")
        message_out = Message(content="HeyHey", role="user")
        session = Session(messages=[message_in])

        cache_provider = LRUCacheProvider(1)
        cache_provider.add(session, message_out)

        config = OpenAIConfig(
            api_key="dummy",
            model_name="gpt-4o-mini",
            max_tokens=1000,
            temperature=0,
            cache_provider=cache_provider,
        )
        model = OpenAIModel(configuration=config)

        message = model.send(session=session)
        self.assertEqual(message.content, message_out.content)

    def test_mock_in_models(self):
        """Test model behavior with mock provider"""
        message_in = Message(content="Hey", role="user")
        session = Session(messages=[message_in])

        mock_provider = OneTurnConversationMockProvider(
            {"Hey": Message(content="MockResponse", role="assistant")}
        )

        config = OpenAIConfig(
            api_key="dummy",
            model_name="gpt-4o-mini",
            max_tokens=1000,
            temperature=0,
            mock_provider=mock_provider,
        )
        model = OpenAIModel(configuration=config)

        message = model.send(session=session)
        self.assertEqual(message.content, "MockResponse")


if __name__ == "__main__":
    unittest.main()
