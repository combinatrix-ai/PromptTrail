import json
import os
import unittest

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates._control import LinearTemplate
from prompttrail.agent.templates._core import MessageTemplate
from prompttrail.agent.templates.anthropic import AnthropicToolingTemplate
from prompttrail.agent.templates.openai import OpenAIToolingTemplate
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel, AnthropicParam
from prompttrail.models.openai import OpenAIConfiguration, OpenAIModel, OpenAIParam


class WeatherTool(Tool):
    def __init__(self):
        super().__init__(
            name="get_weather",
            description="Get weather information for a city. Returns condition and temperature in celcius.",
            arguments={
                "city": ToolArgument(
                    name="city", description="City name", value_type=str, required=True
                )
            },
        )

    def execute(self, **kwargs) -> ToolResult:
        city = kwargs["city"]
        return ToolResult(
            content={"temperature": 20, "condition": "sunny", "city": city}
        )


class TestAnthoropicToolingTemplate(unittest.TestCase):
    def setUp(self):
        self.tool = WeatherTool()
        self.model = AnthropicModel(
            configuration=AnthropicConfig(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        )
        self.param = AnthropicParam(
            # You may need sonnet for tooling
            model_name="claude-3-5-sonnet-latest",
            tools=[self.tool],
        )
        self.template = LinearTemplate(
            [
                MessageTemplate(role="user", content="What's the weather in Tokyo?"),
                tooling_template := AnthropicToolingTemplate(tools=[self.tool]),
            ]
        )
        self.tooling_template = tooling_template
        self.runner = CommandLineRunner(
            model=self.model,
            user_interaction_provider=EchoUserInteractionTextMockProvider(),
            template=self.template,
            parameters=self.param,
        )

    def test_get_tool(self):
        """Test tool retrieval by name"""
        tool = self.tooling_template.get_tool("get_weather")
        self.assertEqual(tool.name, "get_weather")

        with self.assertRaises(ValueError):
            self.tooling_template.get_tool("nonexistent_tool")

    def test_tool_execution(self):
        """Test basic tool execution flow"""

        session = Session(messages=[])
        session.runner = self.runner

        # Run through template
        messages = []
        gen = self.template.render(session)
        try:
            while True:
                messages.append(next(gen))
                print(messages[-1])
        except StopIteration as e:
            e.value

        # Session should look like this:
        # Message(role='user', content='What's the weather in Tokyo?')
        # Message(role='assistant', content='[use_tol]', metadata={'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}})
        # Message(role='tool_result', content='{"temperature": 20, "condition": "sunny", "city": "Tokyo"}')
        # Message(role='assistant', content='Weather in Tokyo is 20 degrees and sunny')

        # Verify tool result was processed
        print(session.messages)
        self.assertEqual(len(session.messages), 4)

        self.assertListEqual(
            [m.role for m in session.messages],
            ["user", "assistant", "tool_result", "assistant"],
        )
        self.assertEqual(session.messages[0].content, "What's the weather in Tokyo?")
        self.assertIn("tool_use", session.messages[1].metadata)
        self.assertEqual(
            session.messages[2].content,
            json.dumps(
                {
                    "temperature": 20,
                    "condition": "sunny",
                    "city": "Tokyo",
                    "type": "tool_result",
                }
            ),
        )
        self.assertIn("20", session.messages[3].content)


class TestOpenAIToolingTemplate(unittest.TestCase):
    def setUp(self):
        self.tool = WeatherTool()
        self.model = OpenAIModel(
            configuration=OpenAIConfiguration(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        )
        self.param = OpenAIParam(
            model_name="gpt-4o-mini",
            tools=[self.tool],
        )
        self.template = LinearTemplate(
            [
                MessageTemplate(role="user", content="What's the weather in Tokyo?"),
                tooling_template := OpenAIToolingTemplate(tools=[self.tool]),
            ]
        )
        self.tooling_template = tooling_template
        self.runner = CommandLineRunner(
            model=self.model,
            user_interaction_provider=EchoUserInteractionTextMockProvider(),
            template=self.template,
            parameters=self.param,
        )

    def test_tool_execution(self):
        """Test basic tool execution flow with OpenAI function calling"""

        session = Session(messages=[])
        session.runner = self.runner

        # Run through template
        messages = []
        gen = self.template.render(session)
        try:
            while True:
                messages.append(next(gen))
                print(messages[-1])
        except StopIteration as e:
            e.value

        # Session should look like this:
        # Message(role='user', content='What's the weather in Tokyo?')
        # Message(role='assistant', content='Let me check the weather for you.', metadata={'function_call': {'name': 'get_weather', 'arguments': '{"city": "Tokyo"}'}})
        # Message(role='tool_result', content='{"temperature": 20, "condition": "sunny", "city": "Tokyo"}')
        # Message(role='assistant', content='The weather in Tokyo is currently sunny with a temperature of 20 degrees Celsius.')

        # Verify tool result was processed
        self.assertEqual(len(session.messages), 4)

        self.assertListEqual(
            [m.role for m in session.messages],
            ["user", "assistant", "tool_result", "assistant"],
        )
        self.assertEqual(session.messages[0].content, "What's the weather in Tokyo?")
        self.assertIn("function_call", session.messages[1].metadata)
        self.assertEqual(
            session.messages[2].content,
            json.dumps({"temperature": 20, "condition": "sunny", "city": "Tokyo"}),
        )
        self.assertIn("20", session.messages[3].content)
        self.assertIn("sunny", session.messages[3].content)


if __name__ == "__main__":
    unittest.main()
