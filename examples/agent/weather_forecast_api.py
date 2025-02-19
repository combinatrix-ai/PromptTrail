"""
This example demonstrates how to use the APIRunner to expose a weather forecast agent as a REST API.

The example shows:
1. How to set up an APIRunner with the weather forecast tool
2. How to configure the API server
3. How to interact with the API endpoints

To test this example:
1. Start the server: python weather_forecast_api.py
2. Use curl or any HTTP client to interact with the API endpoints
"""

import os
from typing import Any, Dict, Literal

from typing_extensions import TypedDict

from prompttrail.agent.runners import APIRunner
from prompttrail.agent.templates import (
    LinearTemplate,
    OpenAIToolingTemplate,
    SystemTemplate,
    UserTemplate,
)
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interface import CLIInterface
from prompttrail.core import Session
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


class WeatherData(TypedDict):
    """Weather data structure"""

    temperature: float
    weather: Literal["sunny", "rainy", "cloudy", "snowy"]


class WeatherForecastResult(ToolResult):
    """Weather forecast result"""

    content: WeatherData


class WeatherForecastTool(Tool):
    """Weather forecast tool"""

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

    def _execute(self, session: Session, args: Dict[str, Any]) -> ToolResult:
        """Mock weather forecast implementation"""
        return WeatherForecastResult(content={"temperature": 0.0, "weather": "sunny"})


def create_weather_api():
    # Create tool and model
    weather_tool = WeatherForecastTool()
    config = OpenAIConfig(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy_key"),
        model_name="gpt-4o-mini",
        max_tokens=100,
        temperature=0,
        tools=[weather_tool],
    )
    model = OpenAIModel(configuration=config)

    # Create templates
    system = SystemTemplate(
        content="You're an AI weather forecast assistant that helps users find weather forecasts.",
    )
    function_calling = OpenAIToolingTemplate(tools=[weather_tool])

    # Create template that defines the conversation flow
    template = LinearTemplate(
        [
            system,
            UserTemplate(
                content="{{user_input}}"
            ),  # Will be filled from session metadata
            function_calling,
        ]
    )

    # Create API runner
    runner = APIRunner(
        model=model,
        template=template,
        user_interface=CLIInterface(),
        host="127.0.0.1",
        port=8000,
    )

    return runner


if __name__ == "__main__":
    # Create and start the API server
    runner = create_weather_api()
    print("Starting weather forecast API server...")
    print("Example usage:")
    print(
        """
    # Create a new session
    curl -X POST http://localhost:8000/sessions \\
        -H "Content-Type: application/json" \\
        -d '{"metadata": {"user_input": "What'\''s the weather in Tokyo?"}}'

    # Start the session
    curl -X POST http://localhost:8000/sessions/{session_id}/start

    # Check session status
    curl http://localhost:8000/sessions/{session_id}
    """
    )
    runner.start_server()
