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
from prompttrail.agent.templates import (
    LinearTemplate,
    OpenAIToolingTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interface import EchoMockInterface
from prompttrail.core import Session
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


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
weather_tool = WeatherForecastTool()
config = OpenAIConfig(
    api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"),
    model_name="gpt-4o-mini",
    max_tokens=100,
    temperature=0,
    tools=[weather_tool],
)
model = OpenAIModel(configuration=config)

# Create templates for the conversation
system = SystemTemplate(
    content="You're an AI weather forecast assistant that help your users to find the weather forecast.",
)
function_calling = OpenAIToolingTemplate(tools=[weather_tool])

# Create linear template that defines the conversation flow
template = LinearTemplate(
    [
        system,
        UserTemplate(content="What's the weather in Tokyo tomorrow?"),
        function_calling,
    ]
)

# Create runner instance with model and template
runner = CommandLineRunner(
    model=model,
    template=template,
    user_interface=EchoMockInterface(),
)


if __name__ == "__main__":
    session = Session()
    runner.run(session=session, max_messages=10)
