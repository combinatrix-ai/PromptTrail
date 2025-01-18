import enum
from typing import Any, Dict

import pytest
from typing_extensions import TypedDict

from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.core.errors import ParameterValidationError


class TestResultData(TypedDict):
    """Test result data type"""

    value: int


class TestResult(ToolResult):
    """Test result class"""

    content: TestResultData


class TestEnumType(enum.Enum):
    """Test enum type"""

    A = "a"
    B = "b"


def test_tool_result():
    """Test tool result creation and validation"""
    # Valid result
    result = TestResult(content={"value": 42})
    assert result.content["value"] == 42

    # Invalid result (wrong type)
    with pytest.raises(ValueError):
        TestResult(content={"value": "not an int"})


def test_tool_argument():
    """Test tool argument creation and validation"""
    # Test with required argument
    arg = ToolArgument[int](
        name="test", description="test argument", value_type=int, required=True
    )
    assert arg.name == "test"
    assert arg.description == "test argument"
    assert arg.value_type == int
    assert arg.required is True

    # Test with optional argument
    arg = ToolArgument[str](
        name="test", description="test argument", value_type=str, required=False
    )
    assert arg.required is False

    # Test with enum type
    arg = ToolArgument[TestEnumType](
        name="test", description="test argument", value_type=TestEnumType, required=True
    )
    assert arg.value_type == TestEnumType


def test_tool():
    """Test tool creation and execution"""

    class TestTool(Tool):
        """Test tool implementation"""

        name: str = "test"
        description: str = "test tool"
        arguments: Dict[str, ToolArgument[Any]] = {
            "arg1": ToolArgument[int](
                name="arg1",
                description="first argument",
                value_type=int,
                required=True,
            ),
            "arg2": ToolArgument[str](
                name="arg2",
                description="second argument",
                value_type=str,
                required=False,
            ),
        }

        def _execute(self, args: Dict[str, Any]) -> ToolResult:
            return TestResult(content={"value": args["arg1"]})

    # Create and execute tool
    tool = TestTool()
    result = tool.execute(arg1=42, arg2="test")
    # Optional argument is not provided
    result = tool.execute(arg1=42)
    # Order of arguments is not important
    result = tool.execute(arg2="test", arg1=42)
    assert isinstance(result, TestResult)
    assert result.content["value"] == 42


def test_tool_validation():
    """Test tool validation"""

    class TestTool(Tool):
        """Test tool implementation"""

        name: str = "test"
        description: str = "test tool"
        arguments: Dict[str, ToolArgument[Any]] = {
            "arg1": ToolArgument[int](
                name="arg1",
                description="first argument",
                value_type=int,
                required=True,
            )
        }

        def _execute(self, args: Dict[str, Any]) -> ToolResult:
            return TestResult(content={"value": args["arg1"]})

    # Test with invalid argument
    tool = TestTool()
    with pytest.raises(ParameterValidationError):
        tool.validate_arguments({"unknown_arg": 42})

    # Test with missing required argument
    with pytest.raises(ParameterValidationError):
        tool.validate_arguments({})

    # Test with wrong type
    with pytest.raises(ParameterValidationError):
        tool.validate_arguments({"arg1": "not an int"})

    # Test with valid arguments
    tool.validate_arguments({"arg1": 42})
