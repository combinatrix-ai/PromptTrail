"""
This example demonstrates how to use tools in PromptTrail.

Tools are a way to extend the capabilities of language models by allowing them to call external functions.
In this example, we create a simple weather forecast tool that returns mock weather data.

The example shows:
1. How to define a tool result class that specifies the structure of the tool's output
2. How to define a tool class with arguments and execution logic
3. How to create a template that uses the tool
4. How to run the template with a command line runner
"""

import os
from typing import Any, Dict, Literal

from typing_extensions import TypedDict

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import LinearTemplate, MessageTemplate
from prompttrail.agent.templates.openai import OpenAIToolingTemplate
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.core import Session
from prompttrail.models.openai import OpenAIConfiguration, OpenAIModel, OpenAIParam


class WeatherData(TypedDict):
    """Weather data structure"""

    temperature: float
    weather: Literal["sunny", "rainy", "cloudy", "snowy"]


class WeatherForecastResult(ToolResult):
    """Weather forecast result

    This class defines the structure of the data returned by the weather forecast tool.
    The content field is a dictionary that contains:
    - temperature: The temperature in the specified unit (float)
    - weather: The weather condition (one of: sunny, rainy, cloudy, snowy)
    """

    content: WeatherData


class WeatherForecastTool(Tool):
    """Weather forecast tool

    This tool simulates getting weather forecast data for a location.
    It demonstrates how to:
    - Define tool arguments (location and temperature unit)
    - Process those arguments in the execute method
    - Return structured data using a ToolResult class
    """

    name: str = "get_weather_forecast"
    description: str = "Get the current weather in a given location and date"
    arguments: Dict[str, ToolArgument[Any]] = {
        "location": ToolArgument(
            name="location",
            description="The location to get the weather forecast",
            value_type=str,
            required=True,
        ),
        "unit": ToolArgument(
            name="unit",
            description="The unit of temperature (Celsius or Fahrenheit)",
            value_type=str,
            required=False,
        ),
    }

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute weather forecast tool

        This is a mock implementation that always returns the same data.
        In a real application, this would call a weather API.
        """
        return WeatherForecastResult(content={"temperature": 0.0, "weather": "sunny"})


# Create OpenAI model with configuration
api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")
config = OpenAIConfiguration(api_key=api_key)
model = OpenAIModel(configuration=config)

# Create templates for the conversation
system = MessageTemplate(
    content="You're an AI weather forecast assistant that help your users to find the weather forecast.",
    role="system",
)
function_calling = OpenAIToolingTemplate(tools=[WeatherForecastTool()])

# Create linear template that defines the conversation flow
template = LinearTemplate(
    templates=[
        system,
        MessageTemplate(content="What's the weather in Tokyo tomorrow?", role="user"),
        function_calling,
    ]
)

# Create runner instance with model, parameters, and template
parameters = OpenAIParam(model_name="gpt-4o-mini", max_tokens=100, temperature=0)
runner = CommandLineRunner(
    model=model,
    parameters=parameters,
    template=template,
    user_interaction_provider=EchoUserInteractionTextMockProvider(),
)


if __name__ == "__main__":
    session = Session()
    runner.run(session=session, max_messages=10)
