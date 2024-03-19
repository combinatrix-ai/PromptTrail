(models)=
# `models`: Unified Interface to LLM API

If you want to just use LLM models, you can use `models` module.
Let's see how to use it.

```{Note}
If you build complex conversation flow, you may want to use `agents` module.
See [agents](#agents) section for more details.
```

## Make an API Call

PromptTrail implement many LLM models under `models`.
Let's call OpenAI's GPT-3 model. (You need to set `API_KEY` environment variable with your OpenAI API key.)

```python
import os
from prompttrail.core import Session, Message
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters
)

api_key = os.environ["OPENAI_API_KEY"]
config = OpenAIModelConfiguration(api_key=api_key)
parameters = OpenAIModelParameters(model_name="gpt-3.5-turbo", max_tokens=100, temperature=0)
model = OpenAIChatCompletionModel(configuration=config)
session = Session(
  messages=[
    Message(content="Hey", sender="user"),
  ]
)
model.send(parameters=parameters, session=session)
```

You can see the response from the model like this:

```python
Message(content="Hello! How can I assist you today?", sender="assistant")
```

Yay! You have successfully called OpenAI's GPT-3.5 model.

## Core Concepts

Some new concepts are introduced in the example above. Let's see them in detail.
You can skip the following sections if you're already familiar with other LLM libraries.

### Message

```python
Message(content="Hello! How can I assist you today?", sender="assistant", data={})
```

Message represents a single message in a conversation.
It has the following attributes:

- `content: str`: the content of the message (text)
- `sender: str`: the sender of the message
  - OpenAI's API expect one of `system`, `user`, `assistant` as the sender.
  - Other providers have different rules.
- `data`: additional data for the message (used for advanced features)

### Session

```python
session = Session(
  messages=[
    Message(content="Hey", sender="user"),
  ]
)
```

Session represents a conversation.
Session is just a collection of messages.

```{Note}
If you want to use non-chat models as traditional language models, you can just pass a session with a single message.
```

### Model

```python
message = model.send(parameters=parameters, session=session)
```

We call the interface to access LLM models `Model`.

Conversation with LLM is a simple process:

- Pass a `Session` to the LLM
- Get a new `Message` from the LLM

Therefore, `Model` is simple. You only need to remeber one method:

- `send(session: Session) -> Message`: send a session to the model and get the response

### Configuration

```python
config = OpenAIModelConfiguration(api_key=api_key)
...
model = OpenAIChatCompletionModel(configuration=config)
```

On initialization of the model, you need to pass a `Configuration` object.
Each provider has different configuration parameters.
Things won't change course of the conversation (e.g. API key) are passed here.

```{Note}
`CacheProvider` is also passed as a configuration parameter. PromptTrail has a built-in cache  mechanism to reduce the number of API calls. See `Cache` section for more details.
```

### Parameters

```python
parameters = OpenAIModelParameters(model_name="gpt-3.5-turbo", max_tokens=100, temperature=0)
...
message = model.send(parameters=parameters, session=session)
```

Different from `Configuration`, `Parameters` are passed on every API call.
Things that can change course of the conversation (e.g. model name, temperature) are passed here.
For example, you may want to GPT-3.5 for the first message, and if the conversation is not going well, you may want to switch to GPT-4.

## Try different Model

If you want to call Google's Palm model, you can do it by changing some lines.

```python
import os
from prompttrail.core import Session, Message
from prompttrail.models.google_cloud import (
    GoogleCloudChatModel, # Change Model
    GoogleCloudChatModelParameters, # Change Parameters
    GoogleCloudChatModelConfiguration, # Change Configuration
)

api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
config = GoogleCloudChatModelConfiguration(api_key=api_key)
# Change model name, of course Google's model name is different from OpenAI's
parameters = GoogleCloudChatModelParameters(model_name="models/chat-bison-001", max_tokens=100, temperature=0)
model = GoogleCloudChatModel(configuration=config)
session = Session(
  messages=[
    Message(content="Hey", sender="user"),
  ]
)
message = model.send(parameters=parameters, session=session)
```

You will get the following response:

```python
Message(content='Hey there! How can I help you today?', sender='1', data={})
```

You may notice the sender system is different from OpenAI's!
We're successfully using Google's model!

The code is almost the same as the OpenAI example. Just change the `Model`, `Configuration` and `Parameters` to Google's.

You will get plenty type hints for every model and parameters. So you may not need to view documentation for every provider.

```{Note}
PromptTrail is fully typed. Therefore we recommend you to write code with VSCode or PyCharm.
docstring is also available for almost every class and method.
We want users to be able to write code without viewing documentation.
```

Anthropic's Claude is also available:

```python
import os
from prompttrail.core import Session, Message
from prompttrail.models.anthropic import (
    AnthropicClaudeModel, # Change Model
    AnthropicClaudeModelParameters, # Change Parameters
    AnthropicClaudeModelConfiguration, # Change Configuration
)

api_key = os.environ["ANTHROPIC_API_KEY"]
config = AnthropicClaudeModelConfiguration(api_key=api_key)
parameters = AnthropicClaudeModelParameters(model_name="claude-3-haiku-20240307", max_tokens=100, temperature=0)
model = AnthropicClaudeModel(configuration=config)
session = Session(
  messages=[
    Message(content="Hey", sender="user"),
  ]
)
message = model.send(parameters=parameters, session=session)
```

You will get the following response:

```python
Message(content='Hello! How can I assist you today?', sender='assistant', data={})
```

## Stream Output

```{Caution}
Streaming output is currently only supported for OpenAI's API.
```

If you want streaming output, you can use the `send_async` method if the provider offers the feature.

```python
message_generator = model.send_async(parameters=parameters, session=session)
for message in message_generator:
    print(message.content, sep="", flush=True)
```

This will print the following:

```shell
Hello! How can # text is incrementally typed
```

`send_async` returns a generator that yields `Message` objects.
