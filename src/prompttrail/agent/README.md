# prompttrail.agent

Agent is a library to build a LLM-based agent with simple DSL.
In this library, agent is defined as an executable control flow of text generation session using LLMs, Tools, and other functions.
You can use agent via CLI, API etc. Therefore, you can of course build chatbot on Agent, but also you can build any application that require multiple-step text generation. If you're just building applications with only single-turn text generation, you just need to use prompttrail.providers, which allows you to use LLMs with simple API.

## Introduction

### `agent.template`

`agent.template` is a library to build a template of text generation session.
You can write how you would like to interact with user, LLM, and functions in it.

Example: A simple proofreading agent

This is actually used in this repository to housekeep README.md etc.
See [examples/dogfooding/fix_markdown.py](../../examples/dogfooding/fix_markdown.py).

```python
from prompttrail.agent.template import LinearTemplate
from prompttrail.agent.template import OpenAIGenerateTemplate as GenerateTemplate
from prompttrail.agent.template import OpenAIMessageTemplate as MessageTemplate

templates = LinearTemplate(
    templates=[
        MessageTemplate(
            content="""
You're an AI proofreader that help user to fix markdown.
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
First `MessageTemplate` is a static template to tell LLM what who they are, as you see `role` is set to `system` following OpenAI's convention.
In this agent, markdown is passed to LLM and LLM returns the cirrected markdown.
Second `MessageTemplate` is a template that takes user's input. `{{content}}` is a placeholder that will be replaced by `runner`.
This is where actual markdown is passed. As some of you may have noticed, this is `Jinja2` template syntax. We use Jinja to dynamically generate templates.
Finally, `GenerateTemplate` is a template that runs LLM actually. So, the result is what we are looking for.

OK. You may grasp what's going on here. Let's run this agent.

### `agent.runner`

`agent.runner` is a library to run the conversation defined in `agent.template`.
We have defined how the conversation should go in `agent.template`.

Then, we need to define how the conversation actually carried out. You need to pass the following arguments:

- How the agent interact with LLM?: Model & Parameter
- How the agent interact with the user?: UserInteractionProvider

In this example, we don't have any user interaction. If you want to see more about user interaction, see [examples/agent/fermi_problem.py](examples/agent/fermi_problem.py).

Let's run the agent above on CLI. Use OpenAI's GPT-3.5-turbo with 16k context. User is interacted with CLI.

```python
import os
from prompttrail.agent import State
from prompttrail.agent.runner import CommandLineRunner
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.provider.openai import (
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

We need to prepare the markdown file to be proofreaded.

```python
markdown = """
# PromptTrail

PrompTrail is library to build text generation agent with LLMs.
"""
```

Then, we need to pass the markdown to the agent.
The point here is `state`. `state` is a state that is passed to the templates. In this example, we pass the markdown to the template.
`state.data` is passed to jinja2 processor and impute the template.
You can also update the data itself with LLM outputs, function results etc. See [examples/agent/fermi_problem.py] for example.

Finally, run the agent!

```python
result = runner.run(
    state=State(
        data = {"content": markdown},
    ),
)
```

You will see the following output on your terminal.

```
StatefulMessage(
{'content': '\n'
            "You're an AI proofreader that help user to fix markdown.\n"
            "You're given markdown content by the user.\n"
            'You only emit the corrected markdown. No explanation, comments, '
            'or anything else is needed.\n'
            'Do not remove > in the code section, which represents the prompt.',
 'sender': 'system',
)
StatefulMessage(
{'content': '\n'
            '# PromptTrail\n'
            '\n'
            'PrompTrail is library to build text generation agent with LLMs.\n',
 'sender': 'user',
)
StatefulMessage(
{'content': '# PromptTrail\n'
            '\n'
            'PromptTrail is a library to build a text generation agent with '
            'LLMs.',
 'sender': 'assistant',
)
```

Pretty simple, right?
What we want is the last message, which is the corrected markdown.
Now we saved the output of runner in `result`, which is the final `state`.
We can extract conversation as follows:

```python
result.session_history.messages
```

We just need last message:

```python
corrected_markdown = result.session_history.messages[-1].content
print(corrected_markdown)
```

The result is (may vary depending on the LLM):

```
# PromptTrail

PromptTrail is a library to build a text generation agent with LLMs.

```

Great! We have built our first agent!
Here we have reviewed the core concepts of `prompttrail.agent`.
You may start using `prompttrail.agent` to build your own agent now!

## Going deeper!

### mocking agent

Let's go deeper. We have created an agent. Yay! But, how do we test it?
A way is calling LLM API, of course. But, it may be costly, slow, and even non-deterministic (eg. [GPT-3.5 and 4 is non-deterministic even if `temperature` is set to `0`](https://152334h.github.io/blog/non-determinism-in-gpt-4/)).

Let's remember what we have done. We have passed `model` and `user_interaction_provider`. We have mocked verion of them:

```python
from prompttrail.agent.user_interaction import (
    OneTurnConversationUserInteractionTextMockProvider,
)
from prompttrail.core import Message
from prompttrail.mock import StaticMockProvider


runner = CommandLineRunner(
    model=OpenAIChatCompletionModelMock(
        configuration=OpenAIModelConfiguration(
            # Of course, you can pass any configuration here.
            api_key="sk-XXX",
        ),
        mock_provider=StaticMockProvider(
            message = Message(
                content="""# PromptTrail
PromptTrail is a library to build a text generation agent with LLMs.
""",
                sender="assistant",
            ),
        ),
    user_interaction_provider=OneTurnConversationUserInteractionTextMockProvider(
        conversation_table={
            "Hello": "Hi"
        }
    ),
    parameters=OpenAIModelParameters(model_name="gpt-3.5-turbo"),
    templates=[agent_template],
)
```

`OpenAIChatCompletionModelMock` is a mock version of `OpenAIChatCompletionModel`. It returns message based on `mock_provider`.
We use `StaticMockProvider` for `mock_provider`. It returns static message regardless of the input.
As our agent is calling LLM only once, we are fine with this.
We don't actually use user_interaction_provider in this example, but for the sake of completeness, we have defined it.
`OneTurnConversationUserInteractionTextMockProvider` returns static message based on the last message.
If last message before asking user input is `Hello`, it returns `Hi` as user input, based on `conversation_table`.
We have other mocking methods, see [prompttrail.mock] for more details.

You can see a more complicated mocking example in [examples/agent/fermi_problem.py].

## `agent.hook`

Hooks are used to enhance the template.
Let's see excerpt from [examples/agent/fermi_problem.py]:

```python
GenerateTemplate(
    role="assistant",
    after_transform=[
        ExtractMarkdownCodeBlockHook(
            key="python_segment", lang="python"
        ),
        EvaluatePythonCodeHook(key="answer", code="python_segment"),
    ],
    after_control=[
        IfJumpHook(
            condition=lambda state: "answer" in state.data,
            true_template="gather_feedback",
            false_template=first.template_id,
        )
    ],
),
```

This template order LLM to generate text, extract python code block from the generated text, and evaluate the code.
`after_transform` is called after LLM generates text. We passed `ExtractMarkdownCodeBlockHook` and `EvaluatePythonCodeHook`.
Let's see what they do.
`ExtractMarkdownCodeBlockHook` extracts code block of the language specified by `lang` from the generated text and store it to `state.data["python_segment"]`.
`EvaluatePythonCodeHook` evaluates the code stored in `state.data["python_segment"]` and store the result to `state.data["answer"]`.
As convention, `key` is used to represent the key of `state.data` to store the result of hook.

After that, `after_control` is called. `IfJumpHook` jumps to `gather_feedback` template if `answer` is in `state.data` or `first.template_id` otherwise. We will explain State later. Now you should remember that hook is dealing with State.
`gather_feedback` and `first.template_id` are template ids, the unique identifier of the template passed to the runner. `template_id` can be set at instantiation like:

```python
GenerateTemplate(
    role="assistant",
    template_id="gather_feedback",
    ...
)
```

However, you can omit it. In that case, `template_id` is automatically generated based. You can get it by `template.template_id`.
Anyway, we omitted the templates in this example, so we will not explain it here. See [examples/agent/fermi_problem.py] for more details.

As there are `after_transform` and `after_control`, there are `before_transform` and `before_control`. They are called before the `rendering` of the template.

The order of hooks are:
- `before_transform`
- `before_control`
- (rendering)
- `after_transform`
- `after_control`

## rendering

`rendering` is a process to create a message from a template.
Every template has a `render` method.
For `MessageTemplate`, it is just rendering the template with `state.data` via jinja2 and return the result as a message.
For `GenerateTemplate`, it is calling LLM and return the result as a message.
For `InputTemplate`, it is asking user input using user_interaction_provider and return the result as a message.

You can of course add your own template. See [template.py] for more details.

### `State`

`State` is a state that is passed to the templates and holds the state of the conversation.
If you're going to build application with `prompttrail.agent`, you just need to the following:

- `State.data`
  - This is a python dictionary you can use to pass data to the templates.
  - Handling `data` is the responsibility of the temlates (and hooks, which we will see later).
  - If specified key is not found in `data`, error will be raised unless you specify `default` in the template or hooks.
- `State.current_template_id`
  - You can know which template is currently running by accessing this.
  - ex. This is used by `IfJumpHook` to jump to another template.

```python
class State(object):
    """State hold the all state of the conversation."""

    def __init__(
        self,
        runner: Optional["Runner"] = None,
        model: Optional[Model] = None,
        parameters: Optional[Parameters] = None,
        data: Dict[str, Any] = {},
        session_history: StatefulSession = StatefulSession(),
        current_template_id: Optional["TemplateId"] = None,
        jump_to_id: Optional["TemplateId"] = None,
    ):
        self.runner = runner
        self.model = model
        self.parameters = parameters
        self.data = data
        self.session_history = session_history
        self.current_template_id = current_template_id
        self.jump_to_id = jump_to_id
```

- Other attributes can be also accessed:
  - `runner`: You can access the runner itself. If you want to search templates passed to runner, you can use `State.runner.search_template`.
  - `model` and `parameter`: You can access the model itself. You can make your own call to the model if you want.
  - `session_history`: You can access the session history. You can review the history of the conversation.
  - `jump_to_id`: This is where you order the runner to jump to another template. Usually, this is manipulated via hooks.


## `agent.tool` (Function Calling)

`agent.tool` is a set of tools that can be used by LLMs. Especially, OpenAI's function calling feature.
Using `agent.tool`, functions called by LLM can be written with an unified interface.
Explanation of types of tool input/output is automatically generated from the type annotation of the function!
Therefore, you don't need to write documentation for LLMs!

Furthermore, `prompttrail` automatically interpret the function calling arguments provided by LLMs!

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


# If you want to use enum, first define the enum.
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
# Howwever, this is the job of the function itself, so we don't need to do this here.


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

This tool definition is converted to following function call:

```
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

```
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
        # In this template, two API calls are made.
        # First, the API is called with the description of the function, which is generated automatically according to the type definition we made.
        # The API return how they want to call the function.
        # Then, according to the response, runner call the function with the arguments provided by the API.
        # Second, the API is called with the result of the function.
        # Finally, the API return the response.
        # Therefore, this template yields three messages. (sender: assistant, function, assistant)
        OpenAIGenerateWithFunctionCallingTemplate(
            role="assistant",
            functions=[WeatherForecastTool()],
        ),
    ]
)
```

So, you don't have to handle the complex function calling by yourself!
You can save your time of ...
- writing the documentation solely for LLM separeted from the function definition
- formatting the input to OpneAI's function calling API
- writing the two-stage call of the function calling API
- writing the interpretation of the function calling API response to feed to the function
- executing the function

Isn't it great?

