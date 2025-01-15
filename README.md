# PromptTrail

PromptTrail is a lightweight library to help you build something with LLM.

PromptTrail provide:

<p align="center">
  <img src="https://github.com/combinatrix-ai/PromptTrail/assets/1284876/dd766b44-165e-4c55-98a3-f009334bbc1c" width="640px">
  <br>
  A unified interface to various LLMs
</p>

<p align="center">
  <img src="https://github.com/combinatrix-ai/PromptTrail/assets/1284876/ef50b481-1ef5-4807-b9c4-6e2ef32d5641" width="640px">
  <br>
  A simple and intuituve DSL for "Agent as Code"
</p>

And various "Developer Tools" to help you build LLM applications.

- [PromptTrail](#prompttrail)
  - [Qucikstart](#qucikstart)
  - [Installation](#installation)
  - [What PromptTrail can do?](#what-prompttrail-can-do)
  - [Examples](#examples)
    - [LLM API Call](#llm-api-call)
    - [Developer Tools](#developer-tools)
    - [Agent as Code](#agent-as-code)
    - [Tooling](#tooling)
  - [Next](#next)
  - [License](#license)
  - [Contributing](#contributing)
  - [Q\&A](#qa)
    - [Why bother yet another LLM library?](#why-bother-yet-another-llm-library)
  - [Showcase](#showcase)

## Qucikstart

- See [Quickstart](https://prompttrail.readthedocs.io/en/latest/quickstart.html) for more details.

## Installation

```bash
pip install prompttrail
```

or

```bash
git clone https://github.com/combinatrix-ai/PromptTrail.git
cd PromptTrail
pip install -e .
```

## What PromptTrail can do?

- PromptTrail offers the following features:
  - [Unified interface to various LLMs](#llm-api-call)
    - OpenAI
    - Google Gemini
    - Anthropic Claude
    - Local LLMs (via Transformers)
  - [Developer Tools for prompt programming](#developer-tools)
    - Mocking LLMs for testing
    - [TODO] Logging
    - [TODO] Debugging
  - [Everything you need to do "Agent as Code"](#agent-as-code)
    - Template
    - Runner
    - Hooks
    - Calling other APIs other than LLMs (Tooling)
      - Function Calling
      - Built-in Tools
        - [TODO] Code Execution
        - [TODO] Vector Search

## Examples

You can find more examples in [examples](examples) directory.

### LLM API Call

This is the simplest example of how to use PromptTrail as a thin wrapper around LLMs of various providers.

```python
> import os
> from src.prompttrail.core import Session, Message
> from src.prompttrail.models.openai import OpenAIModel, OpenAIConfiguration, OpenAIParam
>
> api_key = os.environ["OPENAI_API_KEY"]
> config = OpenAIConfiguration(api_key=api_key)
> parameters = OpenAIParam(model_name="gpt-3.5-turbo", max_tokens=100, temperature=0)
> model = OpenAIModel(configuration=config)
> session = Session(
>   messages=[
>     Message(content="Hey", role="user"),
>   ]
> )
> message = model.send(parameters=parameters, session=session)

Message(content="Hello! How can I assist you today?", role="assistant")
```

If you want streaming output, you can use the `send_async` method if the provider offers the feature.

```python
> message_generator = model.send_async(parameters=parameters, session=session)
> for message in message_generator:
>     print(message.content, sep="", flush=True)

Hello! How can # text is incrementally typed
```

### Developer Tools

We provide various tools for developers to build LLM applications.
For example, you can mock LLMs for testing.

```python
> # Change model class to mock model class
> model = OpenAIChatCompletionModelMock(configuration=config)
> # and just call the setup method to set up the mock provider
> model.setup(
>     mock_provider=OneTurnConversationMockProvider(
>         conversation_table={
>             "1+1": "1215973652716",
>         },
>         role="assistant",
>     )
> )
> session = Session(
>     messages=[
>         Message(content="1+1", role="user"),
>     ]
> )
> message = model.send(parameters=parameters, session=session)
> print(message)

TextMessage(content="1215973652716", role="assistant")
```

### Agent as Code

You can write a simple agent like below. Without reading the documentation, you can understand what this agent does!

```python
template = LinearTemplate(
    [
        MessageTemplate(
            role="system",
            content="You're a math teacher bot.",
        ),
        LoopTemplate(
            [
                UserTemplate(
                    description="Let's ask a question to AI:",
                    default="Why can't you divide a number by zero?",
                ),
                AssistantTemplate(),  # Generates response using LLM
                MessageTemplate(role="assistant", content="Are you satisfied?"),
                UserTemplate(
                    description="Input:",
                    default="Yes.",
                ),
                # Let the LLM decide whether to end the conversation or not
                MessageTemplate(
                    role="assistant",
                    content="The user has stated their feedback."
                    + "If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`."
                ),
                check_end := AssistantTemplate(),  # Generates END or RETRY response
            ],
            exit_condition=BooleanHook(
                condition=lambda session: ("END" == session.get_last_message().content.strip())
            ),
        ),
    ],
)

runner = CommandLineRunner(
    model=OpenAIModel(
        configuration=OpenAIConfiguration(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
    ),
    parameters=OpenAIParam(model_name="gpt-4"),
    template=template,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

runner.run()
```

You can talk with the agent on your console like below:

````console
===== Start =====
From: 📝 system
message:  You're a math teacher bot.
=================
Let's ask a question to AI:
From: 👤 user
message:  Why can't you divide a number by zero?
=================
From: 🤖 assistant
message:  Dividing a number by zero is undefined in mathematics. Here's why:

Let's say we have a division operation a/b. This operation asks the question: "how many times does b fit into a?" If b is zero, the question becomes "how many times does zero fit into a?", and the answer is undefined because zero can fit into a an infinite number of times.

Moreover, if we look at the operation from the perspective of multiplication (since division is the inverse of multiplication), a/b=c means that b*c=a. If b is zero, there's no possible value for c that would satisfy the equation, because zero times any number is always zero, not a.

So, due to these reasons, division by zero is undefined in mathematics.
=================
From: 🤖 assistant
message:  Are you satisfied?
=================
Input:
From: 👤 user
message:  Yes.
=================
From: 🤖 assistant
message:  The user has stated their feedback.If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`.
=================
From: 🤖 assistant
message:  END
=================
====== End ======
````

Go to [examples](examples) directory for more examples.

### Tooling

PromptTrail provides a powerful tool system for function calling that handles all the complexity of:

- Type-safe argument validation
- Automatic documentation generation from type hints
- Function calling API formatting and execution
- Result parsing and conversion

You can define your own tools using TypedDict for structured data and type annotations for safety:

```python
from typing import Literal
from typing_extensions import TypedDict
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult

# Define the structure of tool output
class WeatherData(TypedDict):
    """Weather data structure"""
    temperature: float
    weather: Literal["sunny", "rainy", "cloudy", "snowy"]

# Define the result wrapper
class WeatherForecastResult(ToolResult):
    """Weather forecast result"""
    content: WeatherData

# Implement the tool
class WeatherForecastTool(Tool):
    """Weather forecast tool"""
    name: str = "get_weather_forecast"
    description: str = "Get the current weather in a given location"
    
    # Define arguments with type safety
    arguments: Dict[str, ToolArgument[Any]] = {
        "location": ToolArgument[str](
            name="location",
            description="The location to get the weather forecast",
            value_type=str,
            required=True
        ),
        "unit": ToolArgument[str](
            name="unit",
            description="Temperature unit (celsius or fahrenheit)",
            value_type=str,
            required=False
        )
    }

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the weather forecast tool"""
        # Implement real API call here
        return WeatherForecastResult(
            content={
                "temperature": 20.5,
                "weather": "sunny"
            }
        )

# Use the tool in a template
template = LinearTemplate(
    templates=[
        MessageTemplate(
            content="You are a helpful weather assistant that provides weather forecasts.",
            role="system"
        ),
        MessageTemplate(
            role="user",
            content="What's the weather in Tokyo?",
        ),
        OpenAIGenerateWithFunctionCallingTemplate(
            role="assistant",
            functions=[WeatherForecastTool()],
        ),
    ]
)
```

The conversation will be like below:

```console
===== Start =====
From: 📝 system
message:  You are a helpful weather assistant that provides weather forecasts.
=================
From: 👤 user
message:  What's the weather in Tokyo?
=================
From: 🤖 assistant
data:  {'function_call': {'name': 'get_weather_forecast', 'arguments': {'location': 'Tokyo'}}}
=================
From: 🧮 function
message:  {"content": {"temperature": 20.5, "weather": "sunny"}}
=================
From: 🤖 assistant
message:  The weather in Tokyo is currently sunny with a temperature of 20.5°C.
=================
====== End ======
```

See [documentation)](https://prompttrail.readthedocs.org/en/quickstart-agents.html#tool-function-calling) for more information.


## Next

- [ ] Provide a way to export / import sessions
- [ ] Better error messages that help debugging
- [ ] More default tools
  - [ ] Vector Search Integration
  - [ ] Code Execution
- [ ] toml input/output for templates
- [ ] repository for templates
- [ ] job queue and server
- [ ] asynchronous execution (more complex runner)
File an issue if you have any requests!

## License

- PromptTrail is licensed under the MIT License.

## Contributing

- Contributions are more than welcome!
- See [CONTRIBUTING](CONTRIBUTING.md) for more details.

## Q&A

### Why bother yet another LLM library?

- PromptTrail is designed to be lightweight and easy to use.
- Manipulating LLM is actually not that complicated, but LLM libraries are getting more and more complex to embrace more features.
- PromptTrail aims to provide a simple interface for LLMs and let developers implement their own features.

## Showcase

- If you build something with PromptTrail, please share it with us via Issues or Discussions!
