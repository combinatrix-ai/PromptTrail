# `core`: Developer Tools for LLM

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
model = OpenAIChatCompletionModel(config)
```

We passed a `LRUCacheProvider` to the configuration. And the configuration is passed to the model.
Now, the model will cache the messages.
Just call `send` method as usual. If you pass the same `session` and `parameter`, you will get the cached message.

```python
from prompttrail.core import Session, Message
from prompttrail.models.openai import OpenAIModelParameters
parameter = OpenAIModelParameters(max_tokens=100, model_name="gpt-3.5-turbo")
session = Session(
    messages = [Message(content="Hello, I'm a human.", sender="user")]
)
# This time, the model calls the API
message_1 = model.send(parameters=parameters, session=session)
# This time, the model returns the cached message
message_2 = model.send(parameters=parameters, session=session)
```

Internally, `CacheProvider` has two methods:
- `add(session: Session, message: Message)`: add a new session and message pair to the cache
- `search(session: Session) -> Message`: get a message from the cache

Therefore, `CacheProvider` is basically a simple key-value store.

You don't need to implement `CacheProvider` by yourself. PromptTrail has a built-in `LRUCacheProvider` which is a simple LRU cache. If you want a custom implementation, you can inherit from `CacheProvider` and implement the methods.

(mock)=
## Mock

Say, you successfully build something using LLM. But, how can you test it?
One way is to call the LLM API, of course. But, it may be costly, slow, and even non-deterministic (e.g., [GPT-3.5 and 4 is non-deterministic even if `temperature` is set to `0`](https://152334h.github.io/blog/non-determinism-in-gpt-4/)).


You may want to mock the LLM API. PromptTrail has a built-in mock mechanism.

Mock is implemented as a `MockProvider`. You can pass a `MockProvider` to `Configuration` object like `CacheProvider`.

Let's see an example of `OneTurnConversationMockProvider`, which returns a message based on the last message of the session:

```python
from prompttrail.core import Message
from prompttrail.core.mock import OneTurnConversationMockProvider
from prompttrail.models.openai import OpenAIModelConfiguration
# First, you need to define a table to define how the mock return response based on last message
sender = "assistant"
conversation_table = {
    "Hello": Message(content="Hi", sender=sender),
}
# Then, you can pass the table to the mock provider and pass the mock provider to the configuration
config = OpenAIModelConfiguration(
    api_key=api_key,
    mock_provider= OneTurnConversationMockProvider(conversation_table)
)
model = OpenAIChatCompletionModel(config)
```

Let's call the model:
```python
from prompttrail.core import Session, Message
from prompttrail.models.openai import OpenAIModelParameters
parameter = OpenAIModelParameters(max_tokens=100, model_name="gpt-3.5-turbo")
session = Session(
    messages = [Message(content="Hello", sender="user")]
)
message = model.send(parameters=parameters, session=session)
assert message.content == "Hi"
```

As the session's last message is "Hello", the mock provider returns "Hi" based on the table.

There're other mock providers:

- `EchoMockProvider`: returns the same message as the last message of the session
- `FunctionMockProvider`: returns a message based on a function you defined

The usage of `MockProvider` is very similar to `CacheProvider`. You can pass a `MockProvider` to `Configuration` object like `CacheProvider`.

Actually, the role of `MockProvider` is very similar to that of `CacheProvider`. Because both return a `Message` object based on `Session` object. There're two differences:

- `CacheProvider` checks the `Parameter` because LLM may return different messages for the same session with different parameters.
- `CacheProvider` can return `None` if the message is not in the cache. `MockProvider` always returns a `Message` object.


```{Note}
`MockProvider` and `CacheProvider` cannot be used at the same time.
```

<!-- TODO: Add debug and logging section -->