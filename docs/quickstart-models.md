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
Let's call OpenAI's GPT models. (You need to set `OPENAI_API_KEY` environment variable with your API key.)

```python
import os
from prompttrail.core import Session, Message
from prompttrail.models.openai import OpenAIModel, OpenAIConfig

api_key = os.environ["OPENAI_API_KEY"]
config = OpenAIConfig(
    api_key=api_key,
    model_name="gpt-4o-mini",
    max_tokens=100,
    temperature=0
)
model = OpenAIModel(configuration=config)
session = Session(
  messages=[
    Message(content="Hey", role="user"),
  ]
)
model.send(session=session)
```

You can see the response from the model like this:

```python
Message(content="Hello! How can I assist you today?", role="assistant")
```

Yay! You have successfully called an OpenAI GPT model.

## Core Concepts

Some new concepts are introduced in the example above. Let's see them in detail.
You can skip the following sections if you're already familiar with other LLM libraries.

### Message

```python
Message(content="Hello! How can I assist you today?", role="assistant", metadata={})
```

Message represents a single message in a conversation.
It has the following attributes:

- `content: str`: the content of the message (text)
- `role: str`: the role of the message
  - OpenAI's API expect one of `system`, `user`, `assistant` as the role.
  - Other providers have different rules.
- `metadata`: additional metadata for the message (used for templates, hooks, and other features)

### Session

```python
session = Session(
  messages=[
    Message(content="Hey", role="user"),
  ]
)
```

Session represents a conversation.
Session is just a collection of messages.

```{Note}
If you want to use non-chat models as traditional language models, you can just pass a session with a single message.
```

### Model and Configuration

```python
config = OpenAIConfig(api_key=api_key, model_name="gpt-4o-mini")
model = OpenAIModel(configuration=config)
message = model.send(session=session)
```

We call the interface to access LLM models `Model`.

Conversation with LLM is a simple process:

- Pass a `Session` to the LLM
- Get a new `Message` from the LLM

Each provider has its own configuration class that inherits from `Config`. This configuration includes:
- Static settings (e.g., API keys, organization IDs)
- Model parameters (e.g., model name, temperature, max tokens)
- Optional providers (cache provider, mock provider)

```{Note}
`CacheProvider` can be passed as a configuration parameter. PromptTrail has a built-in cache mechanism to reduce the number of API calls. See `Cache` section for more details.
```

## Try different API

### Google

If you want to call Google's Gemini model, you can do it by changing some lines.

```python
import os
from prompttrail.core import Session, Message
from prompttrail.models.google import GoogleModel, GoogleConfig

api_key = os.environ["GOOGLE_CLOUD_API_KEY"]
config = GoogleConfig(
    api_key=api_key,
    model_name="models/gemini-1.5-flash",
    max_tokens=100,
    temperature=0
)
model = GoogleModel(configuration=config)
session = Session(
  messages=[
    Message(content="Hey", role="user"),
  ]
)
message = model.send(session=session)
```

You will get the following response:

```python
Message(content='Hey there! How can I help you today?', role='1', metadata={})
```

You may notice the role system is different from OpenAI's!
We're successfully using Google's model!

The code is almost the same as the OpenAI example. Just change the `Model` and `Config` to Google's.

You will get plenty type hints for every model and configuration. So you may not need to view documentation for every provider.

```{Note}
PromptTrail is fully typed. Therefore we recommend you to write code with VSCode or PyCharm.
docstring is also available for almost every class and method.
We want users to be able to write code without viewing documentation.
```

### Anthropic

Anthropic's Claude is also available:

```python
import os
from prompttrail.core import Session, Message
from prompttrail.models.anthropic import AnthropicModel, AnthropicConfig

api_key = os.environ["ANTHROPIC_API_KEY"]
config = AnthropicConfig(
    api_key=api_key,
    model_name="claude-3-5-haiku-latest",
    max_tokens=100,
    temperature=0
)
model = AnthropicModel(configuration=config)
session = Session(
  messages=[
    Message(content="Hey", role="user"),
  ]
)
message = model.send(session=session)
```

You will get the following response:

```python
Message(content='Hello! How can I assist you today?', role='assistant', metadata={})
```

## Try local LLMs

One of the key features of PromptTrail is the ability to run LLMs locally using the Transformers library.
This allows you to use any model from the HuggingFace Hub without relying on external APIs.

First, install the transformers library. Please refer to the [Transformers installation guide](https://huggingface.co/docs/transformers/installation) for detailed instructions.

Then you can use it like this:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompttrail.models.transformers import TransformersModel, TransformersConfig
from prompttrail.core import Session, Message

# Load model and tokenizer from HuggingFace Hub
model_name = "facebook/opt-125m"  # You can use any model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create TransformersModel instance
llm = TransformersModel(
    configuration=TransformersConfig(
        device="cuda",  # Use "cpu" if you don't have GPU
        model_name=model_name,
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2
    ),
    model=model,
    tokenizer=tokenizer
)

session = Session(
    messages=[
        Message(content="What is machine learning?", role="user")
    ]
)

# Generate response
response = llm.send(session=session)
```

The TransformersConfig supports various generation parameters:
- `temperature`: Controls randomness in generation (default: 1.0)
- `max_tokens`: Maximum number of tokens to generate (default: 1024)
- `top_p`: Nucleus sampling parameter (default: 1.0)
- `top_k`: Top-k sampling parameter (optional)
- `repetition_penalty`: Penalizes repeated tokens (default: 1.0)

## Stream Output

Streaming output is supported for OpenAI's API and local Transformers models.
If you want streaming output, you can use the `send_async` method if the provider offers the feature.

```python
message_generator = model.send_async(session=session)
for message in message_generator:
    print(message.content, sep="", flush=True)
```

This will print the following:

```shell
Hello! How can # text is incrementally typed
```

`send_async` returns a generator that yields `Message` objects.

For Transformers models, you can use streaming in the same way:

```python
for partial_response in llm.send_async(session=session):
    print(partial_response.content, end="", flush=True)
