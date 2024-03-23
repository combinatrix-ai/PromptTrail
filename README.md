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
    - Google Cloud (Palm)
      - [TODO] Gemini
    - Anthropic Claude
    - [TODO] Local LLMs
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
> from src.prompttrail.models.openai import OpenAIChatCompletionModel, OpenAIModelConfiguration, OpenAIModelParameters
> 
> api_key = os.environ["OPENAI_API_KEY"]
> config = OpenAIModelConfiguration(api_key=api_key)
> parameters = OpenAIModelParameters(model_name="gpt-3.5-turbo", max_tokens=100, temperature=0)
> model = OpenAIChatCompletionModel(configuration=config)
> session = Session(
>   messages=[
>     Message(content="Hey", sender="user"),
>   ]
> )
> message = model.send(parameters=parameters, session=session)

Message(content="Hello! How can I assist you today?", sender="assistant")
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
>         sender="assistant",
>     )
> )
> session = Session(
>     messages=[
>         Message(content="1+1", sender="user"),
>     ]
> )
> message = model.send(parameters=parameters, session=session)
> print(message)

TextMessage(content="1215973652716", sender="assistant")
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
                UserInputTextTemplate(
                    role="user",
                    description="Let's ask a question to AI:",
                    default="Why can't you divide a number by zero?",
                ),
                GenerateTemplate(
                    role="assistant",
                ),
                MessageTemplate(role="assistant", content="Are you satisfied?"),
                UserInputTextTemplate(
                    role="user",
                    description="Input:",
                    default="Yes.",
                ),
                # Let the LLM decide whether to end the conversation or not
                MessageTemplate(
                    role="assistant",
                    content="The user has stated their feedback."
                    + "If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`."
                ),
                check_end := GenerateTemplate(
                    role="assistant",
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda state: ("END" == state.get_last_message().content.strip())
            ),
        ),
    ],
)

runner = CommandLineRunner(
    model=OpenAIChatCompletionModel(
        configuration=OpenAIModelConfiguration(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
    ),
    parameters=OpenAIModelParameters(model_name="gpt-4"),
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

You can use function calling!
In function calling, you need to give LLM instructions to use the tool.
Then, LLM give you the tool arguments and you need to give the result back to LLM.
Therefore, you need:

- giving explanation by the way LLM can understand
- handling of multiple turn conversations
- validation of tool arguments given by LLM
- executing the function and return the result to LLM

PromptTrail handles all of these for you.
You can define your own Tools to call and use them in your templates.
Inherit `Tool`, `ToolArgument`, `ToolResult` and add type annotations.
PromptTrail will automatically generate descriptions for LLM and let the LLM use the tool.
Execution and validation is also handled by PromptTrail.
Let's see a simple weather forecast tool as example:

```python
class Place(ToolArgument):
    description: str = "The location to get the weather forecast"
    value: str

class TemperatureUnitEnum(enum.Enum):
    Celsius = "Celsius"
    Fahrenheit = "Fahrenheit"

class TemperatureUnit(ToolArgument):
    description: str = "The unit of temperature"
    value: Optional[TemperatureUnitEnum]

class WeatherForecastResult(ToolResult):
    temperature: int
    weather: str

    def show(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "weather": self.weather}

class WeatherForecastTool(Tool):
    name = "get_weather_forecast"
    description = "Get the current weather in a given location and date"
    argument_types = [Place, TemperatureUnit]
    result_type = WeatherForecastResult

    def _call(self, args: Sequence[ToolArgument], state: State) -> ToolResult:
        # Implement real API call here
        return WeatherForecastResult(temperature=0, weather="sunny")

template = LinearTemplate(
    templates=[
        MessageTemplate(
            role="system",
            content="You're an AI weather forecast assistant that help your users to find the weather forecast.",
        ),
        MessageTemplate(
            role="user",
            content="What's the weather in Tokyo tomorrow?",
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
message:  You're an AI weather forecast assistant that help your users to find the weather forecast.
=================
From: 👤 user
message:  What's the weather in Tokyo tomorrow?
=================
From: 🤖 assistant
data:  {'function_call': {'name': 'get_weather_forecast', 'arguments': {'place': 'Tokyo', 'temperatureunit': 'Celsius'}}}
=================
From: 🧮 function
message:  {"temperature": 0, "weather": "sunny"}
=================
From: 🤖 assistant
message:  The weather in Tokyo tomorrow is expected to be sunny with a temperature of 0 degrees Celsius.
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
- [ ] Local LLMs

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
