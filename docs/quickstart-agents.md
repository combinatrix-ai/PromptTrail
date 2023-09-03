(agents)=
# `agents`: Agent as Code

```{Note}
If you just want to call LLM API. See [models](#models) section first!
```

An agent is defined as an executable control flow of a text generation session using LLMs, tools, and other functions.
PromptTrail offer a simple and intuitive DSL to build agent with code.

We call this `Agent as Code`.

```{Note}
You can explore `core` module to how you can mock, cache and debug your agent.
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
from prompttrail.agent.templates import LinearTemplate
from prompttrail.agent.templates import OpenAIGenerateTemplate as GenerateTemplate
from prompttrail.agent.templates import OpenAIMessageTemplate as MessageTemplate

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI proofreader that helps users fix markdown.
You're given markdown content by the user.
You only emit the corrected markdown. No explanation, comments, or anything else is needed.
Do not remove > in the code section, which represents the prompt.
""",
            role="system",
        ),
        MessageTemplate(
            content="{{content}}",
            role="user",
        ),
        GenerateTemplate(role="assistant"),
    ],
)
```

The template above is an example of a very simple agent.

`LinearTemplate` is a template that runs templates in order. So, let's see child templates.

The first `MessageTemplate` is a static template to tell LLM what they are, as you see `role` is set to `system` following OpenAI's convention.

In this agent, markdown is passed to LLM and LLM returns the corrected markdown.

The second `MessageTemplate` is a template that takes the user's input. `{{content}}` is a placeholder that will be replaced by `runner`.

This is where the actual markdown is passed. As some of you may have noticed, this is `Jinja2` template syntax. We use Jinja to dynamically generate templates.

Finally, `GenerateTemplate` is a template that runs LLM actually. So, the result is what we are looking for.

OK. You may grasp what's going on here. Let's run this agent.

### Runner

`agent.runner` is a library to run the conversation defined in `agent.templates`.

We have defined how the conversation should go in `agent.templates`.

Then, we need to define how the conversation is actually carried out. You need to pass the following arguments:

- How the agent interacts with LLM?: Model & Parameter
- How the agent interacts with the user?: UserInteractionProvider

In this example, we don't have any user interaction. If you want to see more about user interaction, see [examples/agent/fermi_problem.py](examples/agent/fermi_problem.py).

Let's run the agent above on CLI. Use OpenAI's GPT-3.5-turbo with 16k context. The user is interacted with CLI.

```python
import os
from prompttrail.agent import State
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.models.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

# Setup LLM model
# Don't forget to set OPENAI_API_KEY environment variable
configuration = OpenAIModelConfiguration(api_key=os.environ.get("OPENAI_API_KEY", ""))
parameter = OpenAIModelParameters(
    model_name="gpt-3.5-turbo-16k", temperature=0.0, max_tokens=8000
)
model = OpenAIChatCompletionModel(configuration=configuration)

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

The point here is `state`. `state` is a state that is passed to the templates. In this example, we pass the markdown to the template.

`state.data` is passed to the Jinja2 processor and impute the template.

You can also update the data itself with LLM outputs, function results, etc. See [examples/agent/fermi_problem.py] for an example.

Finally, run the agent!

```python
result = runner.run(
    state=State(
        data={"content": markdown},
    ),
)
```

You will see the following output on your terminal.

```python
StatefulMessage(
  'content': """\nYou're an AI proofreader that helps users fix markdown.\nYou're given markdown content by the user.\nYou only emit the corrected markdown. No explanation, comments, or anything else is needed.\nDo not remove > in the code section, which represents the prompt.""",
  'sender': 'system',
),
StatefulMessage(
  'content': """\n# PromptTrail\n\nPromptTrail is a library to build a text generation agent with LLMs.""",
  'sender': 'user',
),
StatefulMessage(
  'content': """'# PromptTrail\n\nPromptTrail is a library to build a text generation agent with LLMs.""",
  'sender': 'assistant',
)
```

Pretty simple, right?

What we want is the last message, which is the corrected markdown.

Now we saved the output of the runner in `result`, which is the final `state`.

We can extract the conversation as follows:

```python
result.session_history.messages
```

We just need the last message:

```python
corrected_markdown = result.session_history.messages[-1].content
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
GenerateTemplate(
    role="assistant",
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

`ExtractMarkdownCodeBlockHook` extracts a code block of the language specified by `lang` from the generated text and stores it in `state.data["python_segment"]`.

`EvaluatePythonCodeHook` evaluates the code stored in `state.data["python_segment"]` and stores the result in `state.data["answer"]`.

As a convention, `key` is used to represent the key of `state.data` to store the result of the hook.

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

For `MessageTemplate`, it simply renders the template with `state.data` via Jinja2 and returns the result as a message.

For `GenerateTemplate`, it calls the LLM and returns the result as a message.

For `InputTemplate`, it asks for user input using `user_interaction_provider` and returns the result as a message.

You can also add your own template. See [template.py] for more details.

## State

`State` is a state that is passed to the templates and holds the state of the conversation.

If you're going to build an application with `prompttrail.agent`, you just need the following:

- `State.data`
  - This is a Python dictionary you can use to pass data to the templates.
  - Handling `data` is the responsibility of the templates (and hooks, which we will see later).
  - If the specified key is not found in `data`, an error will be raised unless you specify a `default` in the template or hooks.

- `State.current_template_id`
  - You can know which template is currently running by accessing this.
  - For example, this is used by `IfJumpHook` to jump to another template.

```python
class State(object):
    """State holds all the state of the conversation."""

    def __init__(
        self,
        runner: Optional["Runner"] = None,
        model: Optional[Model] = None,
        parameters: Optional[Parameters] = None,
        data: Dict[str, Any] = {},
        session_history: StatefulSession = StatefulSession(),
        current_template_id: Optional["TemplateId"] = None,
    ):
        self.runner = runner
        self.models = model
        self.parameters = parameters
        self.data = data
        self.session_history = session_history
        self.current_template_id = current_template_id
```

Other attributes can also be accessed:

- `runner`: You can access the runner itself. If you want to search templates passed to the runner, you can use `State.runner.search_template`.

- `model` and `parameters`: You can access the model itself. You can make your own call to the model if you want.

- `session_history`: You can access the session history. You can review the history of the conversation.

## Tool (Function Calling)

`agent.tool` is a set of tools that can be used by LLMs, especially OpenAI's function calling feature.

Using `agent.tool`, functions called by LLMs can be written with a unified interface.

Explanation of types of tool input/output is automatically generated from the type annotation of the function!

Therefore, you don't need to write documentation for LLMs!

Furthermore, `prompttrail` automatically interprets the function calling arguments provided by LLMs!

Let's see an example from [examples/agent/weather_forecast.py]:

```python
from prompttrail.agent.tool import Tool, ToolArgument, ToolResult

# First, we must define the IO of the function.

# The function takes two arguments: place and temperature_unit.
# The function returns the weather and temperature.

# Start with the arguments.
# We define the arguments as a subclass of ToolArgument.
# value is the value of the argument. Define the type of value here.
class Place(ToolArgument):
    description: str = "The location to get the weather forecast"
    value: str

# If you want to use an enum, first define the enum.
class TemperatureUnitEnum(enum.Enum):
    Celsius = "Celsius"
    Fahrenheit = "Fahrenheit"

# And then you can use the class as the type of value.
# Note that if you set the type as Optional, it means that the argument is not required.
class TemperatureUnit(ToolArgument):
    description: str = "The unit of temperature"
    value: Optional[TemperatureUnitEnum]

# We can instantiate the arguments like this:
# place = Place(value="Tokyo")
# temperature_unit = TemperatureUnit(value=TemperatureUnitEnum.Celsius)
# However, this is the job of the function itself, so we don't need to do this here.

# Next, we define the result.
# We define the result as a subclass of ToolResult.
# The result must have a show method that can pass the result to the model.
class WeatherForecastResult(ToolResult):
    temperature: int
    weather: str

    def show(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "weather": self.weather}

# Finally, we define the function itself.
# The function must implement the _call method.
# The _call method takes a list of ToolArgument and returns a ToolResult.
# Passed arguments are compared with argument_types and validated. This is why we have to define the type of arguments.
class WeatherForecastTool(Tool):
    name = "get_weather_forecast"
    description = "Get the current weather in a given location and date"
    argument_types = [Place, TemperatureUnit]
    result_type = WeatherForecastResult

    def _call(self, args: Sequence[ToolArgument]) -> ToolResult:
        return WeatherForecastResult(temperature=0, weather="sunny")
```

This tool definition is converted to the following function call:

```json
{
   "name":"get_weather_forecast",
   "description":"Get the current weather in a given location and date",
   "parameters":{
      "type":"object",
      "properties":{
         "place":{
            "type":"string",
            "description":"The location to get the weather forecast"
         },
         "temperatureunit":{
            "type":"string",
            "description":"The unit of temperature",
            "enum":[
               "Celsius",
               "Fahrenheit"
            ]
         }
      },
      "required":[
         "place",
         "temperatureunit"
      ]
   }
}
```

Then, you can let LLM use this function by using `OpenAIGenerateWithFunctionCallingTemplate`:

```python
template = LinearTemplate(
    templates=[
        MessageTemplate(
            role="system",
            content="You're an AI weather forecast assistant that helps your users find the weather forecast.",
        ),
        MessageTemplate(
            role="user",
            content="What's the weather in Tokyo tomorrow?",
        ),
        # In this template, two API calls are made.
        # First, the API is called with the description of the function, which is generated automatically according to the type definition we made.
        # The API returns how they want to call the function.
        # Then, according to the response, the runner calls the function with the arguments provided by the API.
        # Second, the API is called with the result of the function.
        # Finally, the API returns the response.
        # Therefore, this template yields three messages. (sender: assistant, function, assistant)
        OpenAIGenerateWithFunctionCallingTemplate(
            role="assistant",
            functions=[WeatherForecastTool()],
        ),
    ]
)
```

So, you don't have to handle the complex function calling by yourself!

You can save your time of:

- Writing the documentation solely for LLM separated from the function definition
- Formatting the input to OpenAI's function calling API
- Writing the two-stage call of the function calling API
- Writing the interpretation of the function calling API response to feed to the function
- Executing the function

Isn't it great?

