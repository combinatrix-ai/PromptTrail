import json
import os
import unittest
from typing import Any, Dict

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import OpenAIToolingTemplate, ToolingTemplate
from prompttrail.agent.templates._control import LinearTemplate
from prompttrail.agent.templates._core import MessageTemplate
from prompttrail.agent.templates._tool import ExecuteToolTemplate
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interaction import EchoUserInteractionTextMockProvider
from prompttrail.core import Session
from prompttrail.core.errors import ParameterValidationError
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel, AnthropicParam
from prompttrail.models.openai import OpenAIConfig, OpenAIModel, OpenAIParam


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


class MockTool(Tool):
    """Mock tool for testing ExecuteToolTemplate"""

    model_config = {"arbitrary_types_allowed": True}

    name: str = "mock_tool"
    description: str = "Mock tool for testing"
    arguments: "Dict[str, ToolArgument[Any]]" = {
        "arg1": ToolArgument(
            name="arg1",
            description="First argument",
            value_type=str,
            required=True,
        ),
        "arg2": ToolArgument(
            name="arg2",
            description="Second argument",
            value_type=int,
            required=False,
        ),
    }

    def _execute(self, args: "Dict[str, Any]") -> ToolResult:
        return ToolResult(content={"result": f"Executed with args: {args}"})


class TestExecuteToolTemplate(unittest.TestCase):
    """Test cases for ExecuteToolTemplate"""

    def test_execute_tool_template_basic(self):
        """Test basic functionality of ExecuteToolTemplate."""
        # Setup
        tool = MockTool()
        template = ExecuteToolTemplate(tool=tool)
        session = Session(metadata={"arg1": "test"})

        # Execute template
        messages = list(template.render(session))

        # Verify
        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message.role, "tool_result")
        self.assertIsInstance(message.content, str)

        # Verify content
        content = json.loads(message.content)
        self.assertIn("result", content)
        self.assertIn("Executed with args: {'arg1': 'test'}", content["result"])

        # Verify session
        self.assertEqual(len(session.messages), 1)
        self.assertEqual(session.messages[0], message)

    def test_execute_tool_template_with_all_args(self):
        """Test ExecuteToolTemplate with all possible arguments."""
        # Setup
        tool = MockTool()
        template = ExecuteToolTemplate(tool=tool)
        session = Session(metadata={"arg1": "test", "arg2": 42})

        # Execute template
        messages = list(template.render(session))

        # Verify
        self.assertEqual(len(messages), 1)
        message = messages[0]
        content = json.loads(message.content)
        self.assertIn(
            "Executed with args: {'arg1': 'test', 'arg2': 42}", content["result"]
        )

    def test_execute_tool_template_missing_required_arg(self):
        """Test ExecuteToolTemplate with missing required argument."""
        # Setup
        tool = MockTool()
        template = ExecuteToolTemplate(tool=tool)
        session = Session(metadata={})  # Missing required arg1

        # Execute template and expect error
        with self.assertRaisesRegex(
            ParameterValidationError, "Missing required argument: arg1"
        ):
            list(template.render(session))

    def test_execute_tool_template_invalid_arg_type(self):
        """Test ExecuteToolTemplate with invalid argument type."""
        # Setup
        tool = MockTool()
        template = ExecuteToolTemplate(tool=tool)
        session = Session(metadata={"arg1": "test", "arg2": "not_an_int"})

        # Execute template and expect error
        with self.assertRaisesRegex(
            ParameterValidationError, "Invalid type for argument arg2"
        ):
            list(template.render(session))

    def test_execute_tool_template_metadata_preservation(self):
        """Test that ExecuteToolTemplate preserves session metadata."""
        # Setup
        tool = MockTool()
        template = ExecuteToolTemplate(tool=tool)
        original_metadata = {"arg1": "test", "extra": "data"}
        session = Session(metadata=original_metadata.copy())

        # Execute template
        messages = list(template.render(session))

        # Verify metadata is preserved
        self.assertEqual(session.metadata, original_metadata)
        self.assertEqual(messages[0].metadata, original_metadata)


class TestAnthoropicToolingTemplate(unittest.TestCase):
    def setUp(self):
        self.tool = WeatherTool()
        self.model = AnthropicModel(
            configuration=AnthropicConfig(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        )
        self.param = AnthropicParam(
            # You may need sonnet for tooling
            model_name="claude-3-5-sonnet-latest",
            tools=[self.tool],
        )
        self.template = LinearTemplate(
            [
                MessageTemplate(role="user", content="What's the weather in Tokyo?"),
                tooling_template := ToolingTemplate(tools=[self.tool]),
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
        # Message(role='assistant', content='[use_tol]', tool_use={'tool_calls': [{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}})
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
        self.assertIsNotNone(session.messages[1].tool_use)
        self.assertEqual(
            session.messages[2].content,
            json.dumps({"temperature": 20, "condition": "sunny", "city": "Tokyo"}),
        )
        self.assertIn("20", session.messages[3].content)


class TestOpenAIToolingTemplate(unittest.TestCase):
    def setUp(self):
        self.tool = WeatherTool()
        self.model = OpenAIModel(
            configuration=OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY"))
        )
        self.param = OpenAIParam(model_name="gpt-4o-mini", tools=[self.tool])
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
