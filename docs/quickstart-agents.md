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
    [
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
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.models.openai import (
    OpenAIModel,
    OpenAIConfiguration,
    OpenAIParam,
)

# Setup LLM model
# Don't forget to set OPENAI_API_KEY environment variable
config = OpenAIConfig(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=8000
)
model = OpenAIModel(configuration=config)

# Define runner
runner = CommandLineRunner(
    model=model,
    template=templates,
    user_interface=CLIInterface(),
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

`session.metadata` is passed to the Jinja2 processor and impute the template. Each message also has its own metadata that can be accessed via `message.metadata`. The metadata is managed by a dedicated `Metadata` class that provides dictionary-like operations with type safety.

You can also update the metadata with LLM outputs, function results, etc. See [examples/agent/fermi_problem.py] for an example.

Finally, run the agent!

```python
result = runner.run(
    session=Session(
        metadata={"content": markdown},
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

## Subroutines

Subroutines allow you to execute templates in an isolated session context with flexible strategies for session initialization and message handling. This is particularly useful when you want to:

- Break down complex tasks into smaller, manageable pieces
- Isolate certain parts of the conversation
- Reuse template logic across different agents
- Control message inheritance and propagation

### Session Initialization Strategies

When executing a subroutine, you can control how the session is initialized using various strategies:

```python
from prompttrail.agent.session_init_strategy import (
    CleanSessionStrategy,
    InheritSystemStrategy,
    LastNMessagesStrategy,
    FilteredInheritStrategy,
)

# Start with a clean session
clean_strategy = CleanSessionStrategy()

# Inherit only system messages
system_strategy = InheritSystemStrategy()

# Keep last N messages from parent
last_n_strategy = LastNMessagesStrategy(n=3)

# Custom filtering of messages
def is_important(msg):
    return msg.metadata.get("importance", 0) > 0.5
filtered_strategy = FilteredInheritStrategy(filter_fn=is_important)
```

### Message Squashing Strategies

After subroutine execution, you can control which messages are propagated back to the parent session:

```python
from prompttrail.agent.squash_strategy import (
    LastMessageStrategy,
    FilterByRoleStrategy,
)

# Keep only the last message
last_message = LastMessageStrategy()

# Keep messages with specific roles
assistant_only = FilterByRoleStrategy(roles=["assistant"])
```

### Using SubroutineTemplate

Here's an example of using subroutines to solve math problems:

```python
from prompttrail.agent.subroutine import SubroutineTemplate
from prompttrail.agent.templates import Template, LinearTemplate

class CalculationTemplate(Template):
    """Template for performing calculations"""
    def _render(self, session):
        yield Message(role="assistant", content="Let me solve this step by step:")
        yield Message(role="assistant", content="1. First let's identify the key numbers...")
        yield Message(role="assistant", content="The result is 42")
        return session

class MathProblemTemplate(Template):
    """Template for solving math problems using subroutines"""
    def __init__(self):
        super().__init__()
        # Create calculation subroutine
        calculation = CalculationTemplate()
        self.calculation_subroutine = SubroutineTemplate(
            template=calculation,
            session_init_strategy=InheritSystemStrategy(),
            squash_strategy=FilterByRoleStrategy(roles=["assistant"]),
        )

    def _render(self, session):
        # Main problem-solving flow
        yield Message(role="assistant", content="I'll help solve this math problem")
        
        # Execute calculation subroutine
        for message in self.calculation_subroutine.render(session):
            yield message
            
        yield Message(role="assistant", content="Problem solved!")
        return session

# Use in a linear template
template = LinearTemplate([
    SystemTemplate(content="You are a math teacher."),
    UserTemplate(content="What is 6 x 7?"),
    MathProblemTemplate(),
])
```

This example demonstrates:
1. Isolated execution of the calculation logic
2. Inheritance of system context
3. Filtering of messages to keep only assistant responses
4. Clean separation of concerns between problem setup and calculation

See [examples/agent/subroutine_example.py](examples/agent/subroutine_example.py) for a complete working example.

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

For `InputTemplate`, it asks for user input using `user_interface` and returns the result as a message.

You can also add your own template. See [template.py] for more details.

## Session

`Session` is a state that is passed to the templates and holds the state of the conversation.

If you're going to build an application with `prompttrail.agent`, you just need the following:

- `Session.metadata`
  - This is a `Metadata` instance that holds the session-level metadata.
  - You can use this to pass data to the templates and store session state.
  - Each message also has its own metadata that can be accessed via `message.metadata`.
  - If a key is not found in the metadata, an error will be raised unless you specify a `default` in the template or hooks.

- `Session.messages`
  - This is a list of messages in the conversation.
  - Each message has `content`, `role`, and `metadata`.

```python
class Session(BaseModel):
    """A session represents a conversation between a user and a model, or API etc..."""

    messages: List[Message] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)
    runner: Optional["Runner"] = Field(default=None, exclude=True)
    debug_mode: bool = Field(default=False)
    stack: List["Stack"] = Field(default_factory=list)
    jump_to_id: Optional[str] = Field(default=None)
```

Other attributes can also be accessed:

- `runner`: You can access the runner itself. If you want to search templates passed to the runner, you can use `session.runner.search_template`.

- `stack`: You can access the template stack. This is used for template control flow.

- `jump_to_id`: You can set this to jump to another template. This is used by control flow templates like `JumpTemplate`.

## Control Flow

PromptTrail provides several templates for controlling the flow of conversation:

- `LoopTemplate`: Repeats a sequence of templates until an exit condition is met or maximum iterations reached
  ```python
  LoopTemplate(
      templates=[...],
      exit_condition=lambda session: session.get_last().content == "END",
      exit_loop_count=10  # Optional: maximum number of iterations
  )
  ```

- `IfTemplate`: Conditionally executes templates based on a condition
  ```python
  IfTemplate(
      condition=lambda session: "answer" in session.metadata,
      true_template=AssistantTemplate(...),
      false_template=BreakTemplate()
  )
  ```

- `JumpTemplate`: Jumps to another template when a condition is met
  ```python
  JumpTemplate(
      jump_to="template_id",
      condition=lambda session: session.metadata["should_jump"]
  )
  ```

The conditions are defined as lambda functions that take a Session object and return a boolean value.

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

Then, you can use this tool with OpenAI's function calling feature through the `ToolingTemplate`:

```python
from prompttrail.agent.templates import (
    LinearTemplate,
    UserTemplate,
    ToolingTemplate,
)

template = LinearTemplate(
    [
        MessageTemplate(
            content="You are a helpful weather assistant that provides weather forecasts.",
            role="system"
        ),
        UserTemplate(
            role="user",
            content="What's the weather in Tokyo?",
        ),
        # This template handles the function calling flow:
        # 1. Sends the function definition to OpenAI
        # 2. Gets back which function to call with what arguments
        # 3. Executes the function with those arguments
        # 4. Sends the result back to OpenAI for final response
        ToolingTemplate(
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

### Subroutine Tool

In addition to regular tools, PromptTrail provides `SubroutineTool` that allows you to execute templates as tools. This is particularly useful when you want to:

- Execute complex conversation flows as tools
- Reuse existing templates in function calling context
- Maintain isolation between main conversation and tool execution

Here's an example of using SubroutineTool:

```python
from prompttrail.agent.tools import SubroutineTool
from prompttrail.agent.templates import Template

class WeatherTemplate(Template):
    """Template for weather forecasting"""
    def _render(self, session):
        yield Message(
            role="assistant",
            content="Let me check the weather forecast..."
        )
        yield Message(
            role="assistant",
            content="Based on the data, it will be sunny with 25°C."
        )
        return session

# Create weather tool from template
weather_tool = SubroutineTool(
    name="get_weather",
    description="Get weather forecast for a location",
    template=WeatherTemplate(),
)

# Use like any other tool in ToolingTemplate
template = LinearTemplate([
    SystemTemplate(content="You are a weather assistant."),
    UserTemplate(content="What's the weather?"),
    ToolingTemplate(tools=[weather_tool])
])
```

The SubroutineTool provides:
- Isolated execution context for templates
- Automatic message handling and result formatting
- Integration with function calling flow
- Access to all template features (hooks, control flow, etc.)

See [examples/agent/subroutine_tool_example.py](examples/agent/subroutine_tool_example.py) for a complete working example.

This allows you to focus on implementing the actual tool functionality rather than dealing with API integration details.
Isn't it great?
