import json
import unittest
from typing import Any, Dict

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import OpenAIToolingTemplate, ToolingTemplate
from prompttrail.agent.templates._control import LinearTemplate
from prompttrail.agent.templates._core import GenerateTemplate, MessageTemplate
from prompttrail.agent.templates._tool import ExecuteToolTemplate
from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.user_interface import EchoMockInterface
from prompttrail.core import Message, Session
from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.mocks import EchoMockProvider, FunctionalMockProvider
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


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

    def _execute(self, session: Session, args: Dict[str, Any]) -> ToolResult:
        city = args["city"]
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

    def _execute(self, session: Session, args: Dict[str, Any]) -> ToolResult:
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
        config = AnthropicConfig(
            api_key="dummy",
            model_name="claude-3-5-sonnet-latest",
            tools=[self.tool],
            mock_provider=EchoMockProvider(role="assistant"),
        )
        self.model = AnthropicModel(config)
        self.template = LinearTemplate(
            [
                MessageTemplate(role="user", content="What's the weather in Tokyo?"),
                tooling_template := ToolingTemplate(tools=[self.tool]),
            ]
        )
        self.tooling_template = tooling_template
        self.runner = CommandLineRunner(
            model=self.model,
            user_interface=EchoMockInterface(),
            template=self.template,
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
        self.assertEqual(len(session.messages), 2)

        self.assertListEqual(
            [m.role for m in session.messages],
            ["user", "assistant"],
        )
        self.assertEqual(session.messages[0].content, "What's the weather in Tokyo?")
        self.assertEqual(session.messages[1].content, "What's the weather in Tokyo?")


class TestOpenAIToolingTemplate(unittest.TestCase):
    def setUp(self):
        self.tool = WeatherTool()
        config = OpenAIConfig(
            api_key="dummy",
            model_name="gpt-4o-mini",
            tools=[self.tool],
            mock_provider=EchoMockProvider(role="assistant"),
        )
        self.model = OpenAIModel(config)
        self.template = LinearTemplate(
            [
                MessageTemplate(role="user", content="What's the weather in Tokyo?"),
                tooling_template := OpenAIToolingTemplate(tools=[self.tool]),
            ]
        )
        self.tooling_template = tooling_template
        self.runner = CommandLineRunner(
            model=self.model,
            user_interface=EchoMockInterface(),
            template=self.template,
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

        # Session should look like this with mock provider:
        # Message(role='user', content='What's the weather in Tokyo?')
        # Message(role='assistant', content='What's the weather in Tokyo?')

        # Verify tool result was processed
        self.assertEqual(len(session.messages), 2)
        self.assertListEqual([m.role for m in session.messages], ["user", "assistant"])
        self.assertEqual(session.messages[0].content, "What's the weather in Tokyo?")
        self.assertEqual(session.messages[1].content, "What's the weather in Tokyo?")


class TestModelOverride(unittest.TestCase):
    """Test cases for model override functionality"""

    def setUp(self):
        self.tool = WeatherTool()
        # Setup OpenAI model
        # Setup mock providers that return fixed responses
        openai_mock = FunctionalMockProvider(
            lambda session: Message(content="OpenAI response", role="assistant")
        )
        anthropic_mock = FunctionalMockProvider(
            lambda session: Message(content="Anthropic response", role="assistant")
        )

        openai_config = OpenAIConfig(
            api_key="dummy",
            model_name="gpt-4o-mini",
            tools=[self.tool],
            mock_provider=openai_mock,
        )
        self.openai_model = OpenAIModel(openai_config)

        # Setup Anthropic model
        anthropic_config = AnthropicConfig(
            api_key="dummy",
            model_name="claude-3-5-sonnet-latest",
            tools=[self.tool],
            mock_provider=anthropic_mock,
        )
        self.anthropic_model = AnthropicModel(anthropic_config)

    def test_generate_template_model_override(self):
        """Test that GenerateTemplate can override runner's model"""
        # Create a session with an initial message
        session = Session()
        session.append(Message(role="user", content="Test message"))

        # Create template with model override
        template = GenerateTemplate(
            role="assistant",
            model=self.anthropic_model,  # Override with Anthropic model
        )

        # Setup runner with OpenAI model
        runner = CommandLineRunner(
            model=self.openai_model,
            user_interface=EchoMockInterface(),
            template=template,
        )
        session.runner = runner

        # Run template
        messages = list(template.render(session))

        # Verify Anthropic model was used
        self.assertEqual(messages[-1].content, "Anthropic response")

    def test_tooling_template_model_distinction(self):
        """Test that ToolingTemplate correctly distinguishes between models"""
        # Create a session with an initial message
        session = Session()
        session.append(Message(role="user", content="Test message"))

        # Test with OpenAI model
        openai_template = ToolingTemplate(tools=[self.tool])
        runner_openai = CommandLineRunner(
            model=self.openai_model,
            user_interface=EchoMockInterface(),
            template=openai_template,
        )
        session_openai = session.model_copy()
        session_openai.runner = runner_openai
        messages_openai = list(openai_template.render(session_openai))
        self.assertEqual(messages_openai[-1].content, "OpenAI response")

        # Test with Anthropic model
        anthropic_template = ToolingTemplate(tools=[self.tool])
        runner_anthropic = CommandLineRunner(
            model=self.anthropic_model,
            user_interface=EchoMockInterface(),
            template=anthropic_template,
        )
        session_anthropic = session.model_copy()
        session_anthropic.runner = runner_anthropic
        messages_anthropic = list(anthropic_template.render(session_anthropic))
        self.assertEqual(messages_anthropic[-1].content, "Anthropic response")

    def test_tooling_template_model_override(self):
        """Test that ToolingTemplate can override runner's model"""
        # Create a session with an initial message
        session = Session()
        session.append(Message(role="user", content="Test message"))

        # Create template with model override
        template = ToolingTemplate(
            tools=[self.tool],
            model=self.anthropic_model,  # Override with Anthropic model
        )

        # Setup runner with OpenAI model
        runner = CommandLineRunner(
            model=self.openai_model,
            user_interface=EchoMockInterface(),
            template=template,
        )
        session.runner = runner

        # Run template
        messages = list(template.render(session))

        # Verify Anthropic model was used
        self.assertEqual(messages[-1].content, "Anthropic response")


if __name__ == "__main__":
    unittest.main()
