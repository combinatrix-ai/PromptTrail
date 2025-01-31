import enum
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, TypedDict

import pytest

from prompttrail.agent.tools import Tool, ToolArgument, ToolResult
from prompttrail.agent.tools.builtin import EditFile
from prompttrail.core.errors import ParameterValidationError


class SampleResultData(TypedDict):
    result: str


class SampleResult(ToolResult):
    def __init__(self, result: str):
        super().__init__(content=json.dumps({"result": result}))


class SampleEnumType(enum.Enum):
    A = "a"
    B = "b"


class SampleTool(Tool):
    name: str = "test_tool"
    description: str = "test tool"
    arguments: dict[str, ToolArgument[Any]] = {
        "arg1": ToolArgument(
            name="arg1",
            description="arg1",
            value_type=str,
            required=True,
        ),
        "arg2": ToolArgument(
            name="arg2",
            description="arg2",
            value_type=int,
            required=False,
        ),
        "arg3": ToolArgument(
            name="arg3",
            description="arg3",
            value_type=SampleEnumType,
            required=False,
        ),
    }

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        return SampleResult(result="test")


@pytest.fixture
def test_file() -> Generator[Path, None, None]:
    """Create a temporary test file"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "test.txt"
        content = "Hello World!\nThis is a test file."
        file_path.write_text(content)
        yield file_path


def test_tool_validation():
    """Test tool validation"""
    tool = SampleTool()

    # Test required argument
    with pytest.raises(
        ParameterValidationError, match="Missing required argument: arg1"
    ):
        tool.execute()

    # Test invalid argument type
    with pytest.raises(
        ParameterValidationError, match="Invalid type for argument arg2"
    ):
        tool.execute(arg1="test", arg2="invalid")

    # Test invalid enum value
    with pytest.raises(
        ParameterValidationError, match="Invalid type for argument arg3"
    ):
        tool.execute(arg1="test", arg3="invalid")

    # Test valid arguments
    result = tool.execute(arg1="test", arg2=1, arg3=SampleEnumType.A)
    assert isinstance(result, ToolResult)
    assert isinstance(result.content, str)

    result_dict = json.loads(result.content)
    assert isinstance(result_dict, dict)
    assert "result" in result_dict
    assert result_dict["result"] == "test"


def test_edit_file_with_diff(test_file: Path):
    """Test EditFile tool with diff format"""
    tool = EditFile()

    # Test successful diff application
    result = tool.execute(
        path=str(test_file),
        diff="""<<<<<<< SEARCH
Hello World!
=======
Hi Universe!
>>>>>>> REPLACE""",
    )
    result_dict = eval(result.content)
    assert result_dict["status"] == "success"
    assert "Hi Universe!" in test_file.read_text()

    # Test invalid diff format
    result = tool.execute(path=str(test_file), diff="Invalid diff format")
    result_dict = eval(result.content)
    assert result_dict["status"] == "error"
    assert "Invalid diff format" in result_dict["reason"]

    # Test non-matching search block
    result = tool.execute(
        path=str(test_file),
        diff="""<<<<<<< SEARCH
This text does not exist
=======
Something else
>>>>>>> REPLACE""",
    )
    result_dict = eval(result.content)
    assert result_dict["status"] == "error"
    assert "does not match anything in the file" in result_dict["reason"]
