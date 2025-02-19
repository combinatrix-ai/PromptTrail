import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    EndTemplate,
    LinearTemplate,
    SystemTemplate,
    ToolingTemplate,
    UserTemplate,
)
from prompttrail.agent.tools.builtin import (
    CreateOrOverwriteFile,
    EditFile,
    EndConversationTool,
    ReadFile,
    TreeDirectory,
)
from prompttrail.agent.user_interface import EchoMockInterface
from prompttrail.core import Session
from prompttrail.core.const import ReachedEndTemplateException
from prompttrail.core.mocks import OneTurnConversationMockProvider
from prompttrail.models.openai import OpenAIConfig, OpenAIModel


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)  # Set temporary directory as working directory
        yield Path(tmp_dir)
        os.chdir(original_cwd)  # Return to original working directory after test


@pytest.fixture
def test_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary test file"""
    file_path = temp_dir / "test.txt"
    content = "Hello, World!\nThis is a test file."
    file_path.write_text(content)
    yield file_path


def test_read_file(test_file: Path):
    """Test ReadFile tool"""
    tool = ReadFile()

    # Test successful read
    result = tool.execute(Session(), path=str(test_file))
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert "Hello, World!" in result_dict["content"]

    # Test non-existent file
    result = tool.execute(Session(), path=str(test_file.parent / "nonexistent.txt"))
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "No such file or directory" in result_dict["reason"]


@pytest.fixture
def gitignore(temp_dir: Path) -> Path:
    """Create a .gitignore file for testing"""
    gitignore_path = temp_dir / ".gitignore"
    content = """
ignored-e.py
ignored-f
*.ignore_ext
"""
    gitignore_path.write_text(content)
    return gitignore_path


def test_tree_directory_with_system_tree(temp_dir: Path, gitignore: Path):
    """Test TreeDirectory tool using system tree command"""

    # Create test directory structure
    (temp_dir / "included-a").mkdir()
    (temp_dir / "included-a" / "included-b.py").touch()
    (temp_dir / "included-a" / "ignored-c.ignore_ext").touch()
    (temp_dir / "included-a" / "included-d").mkdir()
    (temp_dir / "included-a" / "included-d" / "ignored-e.py").touch()
    (temp_dir / "ignored-f").mkdir()
    (temp_dir / "ignored-f" / "ignored-g.py").touch()

    tool = TreeDirectory()

    # Test complete directory structure
    result = tool.execute(Session(), root_dir=str(temp_dir))
    result_dict = json.loads(result.content)
    print(result_dict)
    assert result_dict["status"] == "success"
    tree = result_dict["tree"]

    print("\nActual tree output:")
    print(tree)

    # Verify expected output without max_depth
    assert "included-a" in tree
    assert "included-b.py" in tree
    assert "ignored-c.ignore_ext" not in tree
    assert "included-d" in tree
    assert "ignored-e.py" not in tree
    assert "ignored-f" not in tree
    assert "ignored-g.py" not in tree

    # Test depth limit
    result = tool.execute(Session(), root_dir=str(temp_dir), max_depth=1)
    result_dict = json.loads(result.content)
    print(result_dict)
    assert result_dict["status"] == "success"
    tree = result_dict["tree"]

    print("\nActual tree output:")
    print(tree)

    # Verify only top level entries
    assert "included-a" in tree
    assert "ignored-f" not in tree

    # エラーケースのテスト
    result = tool.execute(Session(), root_dir="/nonexistent/path")
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "Error opening directory." in result_dict["reason"]


def test_create_or_overwrite_file(temp_dir: Path):
    """Test CreateOrOverwriteFile tool"""
    tool = CreateOrOverwriteFile()
    test_path = str(temp_dir / "new_file.txt")

    # Test file creation
    content = "Test content"
    result = tool.execute(Session(), path=test_path, content=content)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert Path(test_path).read_text() == content

    # Test file overwrite
    new_content = "New content"
    result = tool.execute(Session(), path=test_path, content=new_content)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert Path(test_path).read_text() == new_content


def test_construct_new_file_content():
    """Test construct_new_file_content method of EditFile"""
    tool = EditFile()

    # Test exact match
    original = "def hello():\n    print('Hello')\n    return True"
    diff = "<<<<<<< SEARCH\ndef hello():\n    print('Hello')\n    return True\n=======\ndef hello():\n    print('Hello, World!')\n    return False\n>>>>>>> REPLACE"
    expected = "def hello():\n    print('Hello, World!')\n    return False\n"
    result = tool.construct_new_file_content(diff, original)
    assert result == expected

    # Test line trim match
    original = "def hello():  \n    print('Hello')  \n    return True  "
    result = tool.construct_new_file_content(diff, original)
    assert result == expected

    # Test block anchor match
    original = "def hello():\n    # コメント\n    print('Hello')\n    return True"
    result = tool.construct_new_file_content(diff, original)
    assert result == expected

    # Test new file creation
    original = ""
    diff = "<<<<<<< SEARCH\n=======\ndef hello():\n    print('Hello, World!')\n    return False\n>>>>>>> REPLACE"
    result = tool.construct_new_file_content(diff, original)
    assert result == "def hello():\n    print('Hello, World!')\n    return False\n"

    # Test full file replacement
    original = "def old():\n    print('Old')\n    return None"
    result = tool.construct_new_file_content(diff, original)
    assert result == "def hello():\n    print('Hello, World!')\n    return False\n"

    # Test no match error
    original = "def hello():\n    print('Hello')\n    return True"
    diff = "<<<<<<< SEARCH\ndef goodbye():\n    print('Goodbye')\n    return False\n=======\ndef hello():\n    print('Hello, World!')\n    return False\n>>>>>>> REPLACE"
    with pytest.raises(ValueError, match="does not match anything in the file"):
        tool.construct_new_file_content(diff, original)


def test_edit_file_tool(temp_dir: Path):
    """Test EditFile tool with new diff format"""
    # Create test file
    file_path = temp_dir / "test.py"
    original_content = "def hello():\n    print('Hello')\n    return True"
    file_path.write_text(original_content)

    tool = EditFile()

    # Test successful edit
    diff = "<<<<<<< SEARCH\ndef hello():\n    print('Hello')\n    return True\n=======\ndef hello():\n    print('Hello, World!')\n    return False\n>>>>>>> REPLACE"

    result = tool.execute(Session(), path=str(file_path), diff=diff)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert (
        file_path.read_text()
        == "def hello():\n    print('Hello, World!')\n    return False\n"
    )

    # Test non-matching search block
    diff = "<<<<<<< SEARCH\ndef nonexistent():\n    pass\n=======\ndef hello():\n    print('Hi!')\n    return None\n>>>>>>> REPLACE"

    result = tool.execute(Session(), path=str(file_path), diff=diff)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "does not match anything in the file" in result_dict["reason"]

    # Test invalid diff format
    result = tool.execute(Session(), path=str(file_path), diff="Invalid diff content")
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"


def test_end_conversation_tool():
    """Test EndConversationTool"""
    tool = EndConversationTool()

    # Test default message
    with pytest.raises(ReachedEndTemplateException) as exc_info:
        _ = tool.execute(Session(), **{})
    assert exc_info.value.farewell_message is None

    # Test custom message
    with pytest.raises(ReachedEndTemplateException) as exc_info:
        _ = tool.execute(Session(), **{"message": "Goodbye!"})
    assert exc_info.value.farewell_message == "Goodbye!"


def test_end_template_farewell():
    """Test EndTemplate with farewell message"""
    # Test without farewell message
    session = Session()
    template = EndTemplate()
    with pytest.raises(ReachedEndTemplateException) as exc_info:
        next(template._render(session))
    assert exc_info.value.farewell_message is None
    assert len(session.messages) == 0

    # Test with farewell message
    template = EndTemplate(farewell_message="See you later!")
    session = Session()
    gen = template._render(session)
    message = next(gen)
    assert message.content == "See you later!"
    assert message.role == "assistant"
    with pytest.raises(ReachedEndTemplateException) as exc_info:
        next(gen)
    # farewell_message is consumed. Reset it to None.
    assert exc_info.value.farewell_message is None


def test_end_conversation_with_llm():
    """Test EndConversationTool with LLM decision making"""
    # Configure mock model to simulate LLM using the tool
    from prompttrail.core import Message

    config = OpenAIConfig(
        api_key="dummy",
        model_name="gpt-4o-mini",
        mock_provider=OneTurnConversationMockProvider(
            {
                "Let's end this chat.": Message(
                    role="assistant",
                    content="Let me end this conversation.",
                    metadata={
                        "function_call": {
                            "name": "end_conversation",
                            "arguments": json.dumps(
                                {"message": "Thanks for chatting!"}
                            ),
                        }
                    },
                )
            }
        ),
    )
    mock_model = OpenAIModel(configuration=config)

    # Create template with system instruction and tool
    template = LinearTemplate(
        [
            SystemTemplate(
                content="You can end the conversation using the end_conversation tool."
            ),
            UserTemplate(content="Let's end this chat."),
            ToolingTemplate(tools=[EndConversationTool()]),
        ]
    )

    # Run the conversation
    runner = CommandLineRunner(
        model=mock_model, template=template, user_interface=EchoMockInterface()
    )

    session = runner.run()

    # Verify the conversation flow
    assert len(session.messages) == 3  # system + user + assistant
    assert session.messages[0].role == "system"
    assert session.messages[1].role == "user"
    assert session.messages[2].role == "assistant"
    assert session.messages[2].content == "Let me end this conversation."
    assert "function_call" in session.messages[2].metadata
    assert session.messages[2].metadata["function_call"]["name"] == "end_conversation"
    assert json.loads(session.messages[2].metadata["function_call"]["arguments"]) == {
        "message": "Thanks for chatting!"
    }


if __name__ == "__main__":
    pytest.main(["-v", __file__])
