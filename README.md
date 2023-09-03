# PromptTrail

PromptTrail is a lightweight library to interact with LLM.

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
    - [Before the first release](#before-the-first-release)
    - [Big Features](#big-features)
  - [License](#license)
  - [Contributing](#contributing)
  - [Q\&A](#qa)
    - [Why bother yet another LLM library?](#why-bother-yet-another-llm-library)
    - [Environment Variables](#environment-variables)
    - [Module Architecture](#module-architecture)

## Qucikstart

- See [Documentation](https://prompttrail.readthedocs.io/en/latest/) for more details.

## Installation

```bash
git clone https://github.com/combinatrix-ai/PromptTrail.git
cd PromptTrail
pip install -e .
```

When we release the first version, we will publish this package to PyPI.

## What PromptTrail can do?

- PromptTrail offers the following features:
  - [Unified interface to various LLMs](#llm-api-call)
    - OpenAI
    - Google Cloud
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
            content="You're a math teacher. You're teaching a student how to solve equations.",
        ),
        LoopTemplate(
            [
                UserInputTextTemplate(
                    role="user",
                    description="Let's ask question to AI:",
                    default="Why can't you divide a number by zero?",
                ),
                GenerateTemplate(
                    role="assistant",
                ),
                MessageTemplate(role="assistant", content="Are you satisfied?"),
                UserInputTextTemplate(
                    role="user",
                    description="Input:",
                    default="Explain more.",
                ),
                # Let the LLM decide whether to end the conversation or not
                MessageTemplate(
                    role="assistant",
                    content="""
                    The user has stated their feedback.
                    If you think the user is satisfied, you must answer `END`. Otherwise, you must answer `RETRY`.
                    """,
                ),
                check_end := GenerateTemplate(
                    role="assistant",
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda state: ("END" in state.get_last_message().content)
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
message:
You're a helpful assistant to solve Fermi Problem.... (omitted)
=================
Input: How many elephants in Japan?
From: 👤 user
message:  How many elephants in Japan?
=================
INFO:prompttrail.agent.templates.core:Generating content with OpenAIChatCompletionModel...
From: 🤖 assistant
message:  Thoughts:
- Elephants are not native to Japan, so the only elephants in Japan would be in zoos.
- According to the Japan Association of Zoos and Aquariums, there are 89 zoos in Japan.
- Not all zoos have elephants, but let's assume that half of them do.
- A large zoo might have up to 5 elephants, but smaller zoos might only have 1 or 2. Let's estimate an average of 3 elephants per zoo.

Equation to be calculated:
- Total Number of Elephants in Japan = Number of Zoos in Japan * Rate of Zoos having elephants * Average Number of Elephants Per Zoo

Calculation:
```python
89 * 0.5 * 3
```
=================
From: 🤖 assistant
message:  The answer is 133.5 . Satisfied?
=================
Input: Yes, I'm satisfied.
From: 👤 user
message:  Yes, I'm satisfied.
=================
From: 🤖 assistant
message:  The user has stated their feedback. If you think the user is satisified, you must answer `END`. Otherwise, you must answer `RETRY`.
=================
INFO:prompttrail.agent.templates.core:Generating content with OpenAIChatCompletionModel...
From: 🤖 assistant
message:  END
=================
====== End ======
````

### Tooling

You can use function calling! See [documentation)](src/prompttrail/agent/README.md#`agent.tool`\(Function Calling\)) for more information

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

## Next

### Before the first release

- [ ] Examples
- [ ] Documentation
- [x] Runner
  - [x] Comprehensive test
  - [x] Sophisticated CLI experience for intuitive demo
- [ ] Vector Search Integration
- [ ] Better error messages that help debugging
- [x] Caching of API call
- [x] Function Calling

### Big Features

- [ ] Provide a way to export / import sessions
- [ ] toml input/output for templates
- [ ] repository for templates
- [ ] job queue and server
- [ ] asynchronous execution (more complex runner)
- [ ] Genral Tooling
- [ ] Local LLMs

## License

- This project is licensed under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license).
  - See [LICENCE](LICENCE.md) for more details.

## Contributing

- Contributions are welcome!
- If you build something with PromptTrail, please share it with us via Issues or Discussions!
- See [CONTRIBUTING](CONTRIBUTING.md) for more details.

## Q&A

### Why bother yet another LLM library?

- PromptTrail is designed to be lightweight and easy to use.
- Manipulating LLM is actually not that complicated, but LLM libraries are getting more and more complex to embrace more features.
- PromptTrail aims to provide a simple interface for LLMs and let developers implement their own features.

### Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI API
- `GOOGLE_CLOUD_API_KEY`: API key for Google Cloud API

### Module Architecture

- core: Base classes such as message, session etc...
- provider: Unified interface to various LLMs
  - openai: OpenAI API
  - stream: OpenAI API with streaming output
  - google: Google Cloud API
  - mock:   Mock of API for testing
- agent
  - runner:   Runner execute agent in various media (CLI, API, etc...) based on Templates with Hooks
  - template: Template for agents, let you write complex agent in a simple way
  - hook:     Pytorch Lightning style hook for agents, allowing you to customize agents based on your needs
    - core:   Basic hooks
    - code:   Hooks for code related tasks
  - tool:     Tooling for agents incl. function calling
  - user_interaction: Unified interface to user interaction
    - console: Console-based user interaction
    - mock:    Mock of user interaction for testing

Your typical workflow is as follows:

- Create a template using control flow templates (Looptemplate, Iftemplate etc..) and message templates
- Run them in your CLI with CLIRunner and test it.
- If you want to use it in your application, use APIRunner!
  - See the examples for server side usage.
- Mock your agent with MockProvider and MockUserInteraction let them automatically test on your CI.
