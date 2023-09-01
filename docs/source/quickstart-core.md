# Quickstart for `core`

`core` module is unified with other modules.
Therefore, you should read other quickstart sections first.

(cache)=
## Cache

LLM APIs are sometimes expensive. PromptTrail has a built-in cache mechanism to reduce the number of API calls.

Cache is implemented as a `CacheProvider`. You can pass a `CacheProvider` to `Configuration` object.

Let's see an example:

```python
from prompttrail.core.cache import LRUCacheProvider
from prompttrail.models.openai import OpenAIModelConfiguration

config = OpenAIModelConfiguration(
    api_key=api_key,
    # Just pass a cache provider to the configuration
    cache_provider=LRUCacheProvider()
)
```

Just pass the config to the model and call `send` method as usual. If you pass the same session, you will get the cached message.

Internally, `CacheProvider` has two methods:
- `add(session: Session, message: Message)`: add a new session and message pair to the cache
- `search(session: Session) -> Message`: get a message from the cache

Therefore, `CacheProvider` is basically a simple key-value store.

You don't need to implement `CacheProvider` by yourself. PromptTrail has a built-in `LRUCacheProvider` which is a simple LRU cache. If you want a custom implementation, you can inherit from `CacheProvider` and implement the methods.

