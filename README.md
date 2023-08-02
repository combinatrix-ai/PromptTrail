# PromptTrail

PromptTrail is a lightweight library to interact with LLM.

## Why bother yet another LLM library?

- PromptTrail is designed to be lightweight and easy to use.
- Manipulating LLM is actually not that complicated, but LLM libraries are getting more and more complex to embrace more features.
- PromptTrail aims to provide simple interface for LLMs, and let developers to implement their own features.

## What PromptTrail can do?

- PromptTrail offers below features:
  - Thin layer of abstraction for LLMs you can intuitively understand
    - Message
    - Session
    - Model
  - Unified interface to various LLMs)
    - OpenAI
    - Google Cloud
    - [TODO] Local LLMs
  - Tools for basic prompt programming
    - Mocking LLMs for testing
    - [TODO] Logging
    - [TODO] Debugging
  - Everything you need to do "Agent as Code"
    - Template
    - Runner
    - Agent
    - Hooks
      - PytorchLightning-like hook-based agent definition is supported
    - [TODO] Calling other APIs other than LLMs (Tooling)
      - [TODO] Vector Search
    - [TODO] Multiple Conversation Flow
      - [TODO] Concurrent Execution
    - [TODO] Unified interface to build/parse LLM input/output and agent for function calling.
      - [TODO] QueryBuilder
      - [TODO] OutputParser

## Example

## LLM API Call

```python
> import os
> from src.prompttrail.core import Session, Message
> from src.prompttrail.providers.openai import OpenAIChatCompletionModel, OpenAIModelConfiguration, OpenAIModelParameters
> 
> api_key = os.environ["OPENAI_API_KEY"]
> config = OpenAIModelConfiguration(api_key=api_key)
> parameters = OpenAIModelParameters(model_name="gpt-3.5-turbo", max_tokens=100, temperature=0)
> model = OpenAIChatCompletionModel(configuration=config)
> session = Session(
>   messages=[
>     TextMessage(content="Hey", sender="user"),
>   ]
> )
> message = model.send(parameters=parameters, session=session)

TextMessage(content="Hello! How can I assist you today?", sender="assistant")
```



If you want streaming output, you can use `send_async` method if the provider offers the feature.
 ```python
> message_genrator = model.send_async(parameters=parameters, session=session)
> for message in message_genrator:
>     print(message.content)
```

If you want to mock LLM, you can use various mock models: 
```python
> model = OpenAIChatCompletionModelMock(configuration=config)
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
> # Mocking API provider and return message based on last message using the table
> message = model.send(parameters=parameters, session=session)
> print(message)

> TextMessage(content="1215973652716", sender="assistant")
```
## Agent

You can write a simple agent like this:

```python
flow_template = LinearTemplate(
    [
        MessageTemplate(
            role="system",
            content="""
            You're a math teacher. You're teaching a student how to solve equations. 
            """,
        ),
        MessageTemplate(
            role="user",
            before_transform=[
                AskUserHook(
                    key="prompt",
                    description="Input:",
                    default="Tell me about yourself:",
                )
            ],
            content="""
            {% prompt %}
            """,
        ),
        LoopTemplate(
            [
                MessageTemplate(
                    role="user",
                    before_transform=[
                        AskUserHook(
                            key="prompt",
                            description="Input:",
                            default="Why you can't divide a number by zero?",
                        )
                    ],
                    content="""
                    {% prompt %}
                    """,
                ),
                MessageTemplate(
                    role="assistant",
                    before_transform=[GenerateChatHook(key="generated_text")],
                    content="""
                    {% generated_text %}

                    Are you satisfied?
                    """,
                ),
                MessageTemplate(
                    role="user",
                    before_transform=[
                        AskUserHook(
                            key="feedback", description="Input:", default="Explain more."
                        )
                    ],
                    content="""
                    {% feedback %}
                    """,
                ),
                MessageTemplate(
                    role="assistant",
                    content="""
                    The user has stated their feedback. If you think the user is satisified, you must answer `END`. Otherwise, you must answer `RETRY`.
                    """,
                ),
                check_end := MessageTemplate(
                    role="assistant",
                    before_transform=[GenerateChatHook(key="generated_text")],
                    content="""
                    {% generated_text %}
                    """,
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda flow_state: (
                    flow_state.get_current_template().id == check_end.id
                    and "END" in flow_state.get_last_message().content
                )
            ),
        ),
    ],
)


runner = FlowRunner(
    model=OpenAIChatCompletionModel(
        configuration=OpenAIModelConfiguration(
            api_key=os.environ.get("OPENAI_API_KEY", "")
        )
    ),
    parameters=OpenAIModelParameters(model_name="gpt-3.5-turbo"),
    templates=[flow_template],
)

runner.run()
```


## Design Principles

- If you know what is LLM, you must be able to use PromptTrail.
- Agent (Flow) as Code
  - Agent that can be written in one place by code
    - Hook-based agent definition like PyTorch Lightning
- Provide an easy way to debug prompt program
  - Turn-based execution
  - Record everything for later inspection
  - Easy to read error messages with template id, hook name etc... is included
- Intuitive and explicit (but sometimes convention)
  - Everything evolves fast here. You can't be sure what is right now. So explicit is better than implicit. Code is better than document.
    - No hidden templates and configurations
    - Every parameter should be passed explicitly and be able to understood by types
      - Easy to work with on VSCode and JetBrains IDEs
  - Everything must be clear by class inhetitance and types. I don't want to read docs.
    - Unified access to templates, parameters of agents
    - Hook-based agent definition
    - More default values


## Next
- [ ] yaml input/output for templates
- [ ] Offer repository for templates
- [ ] job queue and server
- [ ] asynchronous execution (more complex runner)

## License

- This project is licensed under the [Elastic License 2.0](https://www.elastic.co/licensing/elastic-license).
  - See [LICENSE](LICENSE) for more details.

## Contributing

- Contributions are welcome!
- See [CONTRIBUTING](CONTRIBUTING.md) for more details.
