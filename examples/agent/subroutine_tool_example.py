"""Example of using SubroutineTool for executing templates as tools"""

from typing import Generator, Optional

from prompttrail.agent.templates import Stack, Template
from prompttrail.agent.tools import SubroutineTool
from prompttrail.core import Message, Session


class WeatherTemplate(Template):
    """Template for weather forecasting"""

    def __init__(self, template_id: Optional[str] = None):
        super().__init__(template_id=template_id)

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        # Simulate weather forecast process
        yield Message(
            role="assistant",
            content="Let me check the weather forecast for you...",
        )
        yield Message(
            role="assistant",
            content="Based on the latest data, it will be sunny with a high of 25°C.",
        )
        return session

    def create_stack(self, session: Session) -> Stack:
        """Create stack frame for this template."""
        return Stack(template_id=self.template_id)


def main():
    """Example usage of SubroutineTool with a weather template"""
    # Create weather tool from template
    weather_tool = SubroutineTool(
        name="get_weather",
        description="Get weather forecast for a location",
        template=WeatherTemplate(),
    )

    # Example 1: Basic usage
    print("\nExample 1: Basic usage")
    result = weather_tool.execute(
        input="What's the weather like in Tokyo?",
    )
    print(f"Result: {result.content}")
    print("\nAll messages:")
    for msg in result.metadata["messages"]:
        print(f"{msg.role}: {msg.content}")

    # Example 2: With system message
    print("\nExample 2: With system message")
    result = weather_tool.execute(
        input="What's the weather like in Tokyo?",
        system_message="You are a professional meteorologist.",
    )
    print(f"Result: {result.content}")
    print("\nAll messages:")
    for msg in result.metadata["messages"]:
        print(f"{msg.role}: {msg.content}")


if __name__ == "__main__":
    main()
