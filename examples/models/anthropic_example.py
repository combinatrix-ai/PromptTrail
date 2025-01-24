import logging
import os
from typing import Any, Dict, Literal

from typing_extensions import TypedDict

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import SystemTemplate, ToolingTemplate
from prompttrail.agent.templates._control import LinearTemplate
from prompttrail.agent.templates._core import UserTemplate
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interaction import DefaultEchoMockProvider
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel, AnthropicParam


class WeatherData(TypedDict):
    """Weather data structure"""

    temperature: float
    weather: Literal["sunny", "rainy", "cloudy", "snowy"]
    unit: Literal["Celsius", "Fahrenheit"]


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

    name: str = "get-weather-forecast"
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

    def _execute(self, args: Dict[str, Any]) -> WeatherForecastResult:
        """Execute weather forecast tool

        This is a mock implementation that always returns the same data.
        In a real application, this would call a weather API.
        """
        return WeatherForecastResult(
            content={"temperature": 0.0, "weather": "sunny", "unit": "Celsius"}
        )


def main():
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)

    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    # Initialize components
    config = AnthropicConfig(api_key=api_key)
    model = AnthropicModel(configuration=config)
    weather_tool = WeatherForecastTool()

    # Create parameters with tools
    parameters = AnthropicParam(
        model_name="claude-3-opus-latest",
        temperature=0,
        tools=[weather_tool],  # Set tools in parameters
    )

    # Create template with tools
    template = LinearTemplate(
        templates=[
            SystemTemplate(
                content="""You are a helpful weather assistant.""",
            ),
            UserTemplate(
                content="What's the weather in Tokyo?",
            ),
            ToolingTemplate(
                tools=[weather_tool], role="assistant"  # Set tools in template
            ),
        ]
    )

    # Create runner
    runner = CommandLineRunner(
        model=model,
        parameters=parameters,
        template=template,
        user_interaction_provider=DefaultEchoMockProvider(),
    )

    # Run the conversation
    runner.run(max_messages=10)


if __name__ == "__main__":
    main()
