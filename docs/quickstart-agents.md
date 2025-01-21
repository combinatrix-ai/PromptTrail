(agents)=
# `agents`: Agent as Code

```{Note}
If you just want to call LLM API. See [models](#models) section first!
```

An agent is defined as an executable control flow of a text generation session using LLMs, tools, and other functions.
PromptTrail offers a simple and intuitive DSL to build agents with code.

We call this `Agent as Code`.

```{Note}
You can explore the `core` module to learn how you can mock, cache and debug your agent.
```

You can use the agent via CLI, API, etc. Therefore, you can build a chatbot on Agent, but you can also build any application that requires multiple-step text generation. If you're only building applications with single-turn text generation, you just need to use prompttrail.models, which allows you to use LLMs with a simple API.

## Introduction

### Template

`agent.templates` offers a simple DSL to define the conversation flow.

You can write how you would like to interact with the user, LLM, and functions in it.

Example: A simple proofreading agent

This is actually used in this repository to housekeep README.md, etc.

See [examples/dogfooding/fix_markdown.py](https://github.com/combinatrix-ai/PromptTrail/blob/main/examples/dogfooding/fix_markdown.py) for the actual code.

```python
from prompttrail.agent.templates import LinearTemplate, AssistantTemplate, MessageTemplate, SystemTemplate, UserTemplate

templates = LinearTemplate(
    templates=[
        SystemTemplate(
            content="""
You're an AI proofreader that helps users fix markdown.
You're given markdown content by the user.
You only emit the corrected markdown. No explanation, comments, or anything else is needed.
Do not remove > in the code section, which represents the prompt.
""",
        ),
        UserTemplate(
            content="{{content}}",
        ),
        AssistantTemplate(),  # Generates response using LLM
    ],
)
```

The template above is an example of a very simple agent.

`LinearTemplate` is a template that runs templates in order. So, let's see child templates.

The first `SystemTemplate` is a convenience template that automatically sets the role to "system" following OpenAI's convention. It's used to tell LLM what they are.

In this agent, markdown is passed to LLM and LLM returns the corrected markdown.

The second `UserTemplate` is a template that takes the user's input. `{{content}}` is a placeholder that will be replaced by `runner`. This template automatically sets the role to "user".

This is where the actual markdown is passed. As some of you may have noticed, this is `Jinja2` template syntax. We use Jinja to dynamically generate templates.

Finally, `AssistantTemplate` is a template that generates content using LLM. When no content is provided, it automatically calls the LLM to generate a response. This template automatically sets the role to "assistant".

OK. You may grasp what's going on here. Let's run this agent.

### Runner

`agent.runner` is a library to run the conversation defined in `agent.templates`.

We have defined how the conversation should go in `agent.templates`.

Then, we need to define how the conversation is actually carried out. You need to pass the following arguments:

- How the agent interacts with LLM?: Model & Parameter
- How the agent interacts with the user?: UserInteractionProvider

In this example, we don't have any user interaction. If you want to see more about user interaction, see [examples/agent/fermi_problem.py](examples/agent/fermi_problem.py).

Let's run the agent above on CLI. Use OpenAI's gpt-4o-mini. The user is interacted with CLI.

```python
import os
from prompttrail.core import Session
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIModel,
    OpenAIConfiguration,
    OpenAIParam,
)

# Setup LLM model
# Don't forget to set OPENAI_API_KEY environment variable
configuration = OpenAIConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIParam(
    model_name="gpt-4o-mini", temperature=0.0, max_tokens=8000
)
model = OpenAIModel(configuration=configuration)

# Define runner
runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    templates=[templates],
    user_interaction_provider=UserInteractionTextCLIProvider(),
)
```

OK, we are ready to run the agent. Let's run it!

We need to prepare the markdown file to be proofread.

```python
markdown = """
# PromptTrail

PromptTrail is a library to build a text generation agent with LLMs.
"""
```

Then, we need to pass the markdown to the agent.

The point here is `session`. `session` is a state that is passed to the templates. In this example, we pass the markdown to the template.

`session.initial_metadata` is passed to the Jinja2 processor and impute the template. Each message also has its own metadata that can be accessed via `message.metadata`.

You can also update the metadata with LLM outputs, function results, etc. See [examples/agent/fermi_problem.py] for an example.

Finally, run the agent!

```python
result = runner.run(
    session=Session(
        initial_metadata={"content": markdown},
    ),
)
```

You will see the following output on your terminal.

```python
StatefulMessage(
  'content': """\nYou're an AI proofreader that helps users fix markdown.\nYou're given markdown content by the user.\nYou only emit the corrected markdown. No explanation, comments, or anything else is needed.\nDo not remove > in the code section, which represents the prompt.""",
  'role': 'system',
),
StatefulMessage(
  'content': """\n# PromptTrail\n\nPromptTrail is a library to build a text generation agent with LLMs.""",
  'role': 'user',
),
StatefulMessage(
  'content': """'# PromptTrail\n\nPromptTrail is a library to build a text generation agent with LLMs.""",
  'role': 'assistant',
)
```

Pretty simple, right?

What we want is the last message, which is the corrected markdown.

Now we saved the output of the runner in `result`, which is the final `state`.

We can extract the conversation as follows:

```python
result.messages
```

We just need the last message:

```python
corrected_markdown = result.messages[-1].content
print(corrected_markdown)
```

The result is (may vary depending on the LLM):

```markdown
# PromptTrail

PromptTrail is a library to build a text generation agent with LLMs.
```

Great! We have built our first agent!

Here we have reviewed the core concepts of `prompttrail.agent`.

You may start using `prompttrail.agent` to build your own agent now!

## Hooks

Hooks are used to enhance the template.

Let's see an excerpt from [examples/agent/fermi_problem.py]:

```python
AssistantTemplate(
    after_transform=[
        ExtractMarkdownCodeBlockHook(
            key="python_segment", lang="python"
        ),
        EvaluatePythonCodeHook(key="answer", code="python_segment"),
    ]
),
```

This template orders the LLM to generate text, extract a Python code block from the generated text, and evaluate the code.

`after_transform` is called after the LLM generates text. We passed `ExtractMarkdownCodeBlockHook` and `EvaluatePythonCodeHook`.

Let's see what they do.

`ExtractMarkdownCodeBlockHook` extracts a code block of the language specified by `lang` from the generated text and stores it in the message's metadata under the key `"python_segment"`.

`EvaluatePythonCodeHook` evaluates the code stored in the message's metadata under the key `"python_segment"` and stores the result under the key `"answer"`.

As a convention, `key` is used to represent the key in the message's metadata to store the result of the hook.

`gather_feedback` and `first.template_id` are template ids, the unique identifier of the template passed to the runner. `template_id` can be set at instantiation like:

```python
GenerateTemplate(
    role="assistant",
    template_id="gather_feedback",
    ...
)
```

However, you can omit it. In that case, `template_id` is automatically generated. You can get it by `template.template_id`.

Anyway, we omitted the templates in this example, so we will not explain it here. See [examples/agent/fermi_problem.py] for more details.

As there are `after_transform` and there are `before_transform`. They are called before the `rendering` of the template.

The order of hooks is:

- `before_transform`
- (rendering)
- `after_transform`

## Rendering

`rendering` is a process to create a message from a template.

Every template has a `render` method.

For `MessageTemplate`, it simply renders the template with the message's metadata via Jinja2 and returns the result as a message.

For `GenerateTemplate`, it calls the LLM and returns the result as a message.

For `InputTemplate`, it asks for user input using `user_interaction_provider` and returns the result as a message.

You can also add your own template. See [template.py] for more details.

## Session

`Session` is a state that is passed to the templates and holds the state of the conversation.

If you're going to build an application with `prompttrail.agent`, you just need the following:

- `Session.initial_metadata`
  - This is a Python dictionary you can use to pass initial data to the templates.
  - Each message also has its own metadata that can be accessed via `message.metadata`.
  - If a key is not found in the metadata, an error will be raised unless you specify a `default` in the template or hooks.

- `Session.messages`
  - This is a list of messages in the conversation.
  - Each message has `content`, `role`, and `metadata`.
  - You can access the latest metadata using `get_latest_metadata()`.

```python
class Session(BaseModel):
    """A session represents a conversation between a user and a model, or API etc..."""

    messages: List[Message] = Field(default_factory=list)
    initial_metadata: Dict[str, Any] = Field(default_factory=dict)
    runner: Optional["Runner"] = Field(default=None, exclude=True)
    debug_mode: bool = Field(default=False)
    stack: List["Stack"] = Field(default_factory=list)
    jump_to_id: Optional[str] = Field(default=None)

    def get_latest_metadata(self) -> Dict[str, Any]:
        """Get metadata from the last message or initial metadata if no messages exist."""
        if not self.messages:
            return self.initial_metadata.copy()
        return self.messages[-1].metadata
```

Other attributes can also be accessed:

- `runner`: You can access the runner itself. If you want to search templates passed to the runner, you can use `session.runner.search_template`.

- `stack`: You can access the template stack. This is used for template control flow.

- `jump_to_id`: You can set this to jump to another template. For example, this is used by `IfJumpHook` to jump to another template.

## Tool (Function Calling)

`agent.tool` is a set of tools that can be used by LLMs, especially OpenAI's function calling feature.

Using `agent.tool`, functions called by LLMs can be written with a unified interface.
The tool system provides type safety and automatic documentation generation from type annotations.

Let's see an example from [examples/agent/weather_forecast.py]:

```python
from typing import Literal
from typing_extensions import TypedDict
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult

# First, define the structure of the tool's output using TypedDict
class WeatherData(TypedDict):
    """Weather data structure"""
    temperature: float
    weather: Literal["sunny", "rainy", "cloudy", "snowy"]

# Then define the result class that wraps the output structure
class WeatherForecastResult(ToolResult):
    """Weather forecast result
    
    This class defines the structure of the data returned by the weather forecast tool.
    The content field contains:
    - temperature: The temperature in the specified unit (float)
    - weather: The weather condition (one of: sunny, rainy, cloudy, snowy)
    """
    content: WeatherData

# Finally, implement the tool itself
class WeatherForecastTool(Tool):
    """Weather forecast tool
    
    This tool simulates getting weather forecast data for a location.
    """
    name: str = "get_weather_forecast"
    description: str = "Get the current weather in a given location"
    
    # Define the tool's arguments using ToolArgument
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
        """Execute the weather forecast tool
        
        This is where the actual weather data fetching would happen.
        For this example, we return mock data.
        """
        return WeatherForecastResult(
            content={
                "temperature": 20.5,
                "weather": "sunny"
            }
        )
```
This tool definition is automatically converted to OpenAI's function calling format:

```json
{
   "name": "get_weather_forecast",
   "description": "Get the current weather in a given location",
   "parameters": {
      "type": "object",
      "properties": {
         "location": {
            "type": "string",
            "description": "The location to get the weather forecast"
         },
         "unit": {
            "type": "string",
            "description": "Temperature unit (celsius or fahrenheit)"
         }
      },
      "required": ["location"]
   }
}
```

Then, you can use this tool with OpenAI's function calling feature through the `OpenAIGenerateWithFunctionCallingTemplate`:

```python
from prompttrail.agent.templates import (
    LinearTemplate,
    OpenAIMessageTemplate,
    OpenAIGenerateWithFunctionCallingTemplate,
)

template = LinearTemplate(
    templates=[
        MessageTemplate(
            content="You are a helpful weather assistant that provides weather forecasts.",
            role="system"
        ),
        OpenAIMessageTemplate(
            role="user",
            content="What's the weather in Tokyo?",
        ),
        # This template handles the function calling flow:
        # 1. Sends the function definition to OpenAI
        # 2. Gets back which function to call with what arguments
        # 3. Executes the function with those arguments
        # 4. Sends the result back to OpenAI for final response
        OpenAIGenerateWithFunctionCallingTemplate(
            role="assistant",
            functions=[WeatherForecastTool()],
        ),
    ]
)
```

The tool system handles all the complexity of function calling for you:

- Type-safe argument validation
- Automatic documentation generation from type hints and docstrings
- Function calling API formatting and execution
- Result parsing and conversion

This allows you to focus on implementing the actual tool functionality rather than dealing with API integration details.
Isn't it great?
