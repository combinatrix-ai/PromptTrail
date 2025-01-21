import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from examples.dogfooding.dogfooding_tools import (
    CreateOrOverwriteFile,
    EditFile,
    ReadFile,
    TreeDirectory,
    construct_new_file_content,
)


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
    result = tool.execute(path=str(test_file))
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert "Hello, World!" in result_dict["content"]

    # Test non-existent file
    result = tool.execute(path=str(test_file.parent / "nonexistent.txt"))
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "No such file or directory" in result_dict["reason"]


@pytest.fixture
def gitignore(temp_dir: Path) -> Path:
    """Create a .gitignore file for testing"""
    gitignore_path = temp_dir / ".gitignore"
    content = """
.venv/
*.pyc
dist/
.coverage
docs/build/
tmp/
test_ignore/
"""
    gitignore_path.write_text(content)
    return gitignore_path


def test_tree_directory_with_system_tree(temp_dir: Path, gitignore: Path):
    """Test TreeDirectory tool using system tree command"""
    import shutil

    import pytest

    # Skip if tree command is not available
    if shutil.which("tree") is None:
        pytest.skip("tree command not available")

    # Create test directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "main.py").touch()
    (temp_dir / "src" / "test.pyc").touch()  # Should be ignored
    (temp_dir / "docs").mkdir()
    (temp_dir / "docs" / "index.md").touch()
    (temp_dir / "docs" / "build").mkdir()  # Should be ignored
    (temp_dir / ".venv").mkdir()  # Should be ignored
    (temp_dir / ".venv" / "bin").mkdir()
    (temp_dir / ".venv" / "lib").mkdir()
    (temp_dir / "tmp").mkdir()  # Should be ignored
    (temp_dir / "dist").mkdir()  # Should be ignored

    # Write .gitignore content
    gitignore.write_text(
        """
.venv/
*.pyc
dist/
docs/build/
tmp/
"""
    )

    tool = TreeDirectory()

    # Test complete directory structure
    result = tool.execute(path=str(temp_dir))
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    tree = result_dict["tree"]

    print("\nActual tree output:")
    print(tree)

    # Verify expected output
    assert "src" in tree
    assert "main.py" in tree
    assert "docs" in tree
    assert "index.md" in tree

    # Verify files/directories that should be ignored
    assert ".venv" not in tree
    assert "test.pyc" not in tree
    assert "build" not in tree
    assert "tmp" not in tree
    assert "dist" not in tree

    # Test depth limit
    result = tool.execute(path=str(temp_dir), max_depth=1)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    tree = result_dict["tree"]

    # Verify only top level entries
    assert "src" in tree
    assert "docs" in tree
    assert "main.py" not in tree
    assert "index.md" not in tree

    # エラーケースのテスト
    result = tool.execute(path="/nonexistent/path")
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "Error" in result_dict["tree"]


def test_tree_directory_no_tree_command(temp_dir: Path, monkeypatch):
    """Test TreeDirectory tool when tree command is not available"""
    import shutil

    # Simulate tree command not being available
    monkeypatch.setattr(
        shutil, "which", lambda x: None if x == "tree" else "/usr/bin/" + x
    )

    tool = TreeDirectory()
    result = tool.execute(path=str(temp_dir))
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "tree' command not found" in result_dict["tree"]


def test_create_or_overwrite_file(temp_dir: Path):
    """Test CreateOrOverwriteFile tool"""
    tool = CreateOrOverwriteFile()
    test_path = str(temp_dir / "new_file.txt")

    # Test file creation
    content = "Test content"
    result = tool.execute(path=test_path, content=content)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert Path(test_path).read_text() == content

    # Test file overwrite
    new_content = "New content"
    result = tool.execute(path=test_path, content=new_content)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert Path(test_path).read_text() == new_content


def test_construct_new_file_content():
    """Test construct_new_file_content function"""
    # Test exact match
    original = "def hello():\n    print('Hello')\n    return True"
    diff = (
        "<<<<<<< SEARCH\n"
        "def hello():\n"
        "    print('Hello')\n"
        "    return True\n"
        "=======\n"
        "def hello():\n"
        "    print('Hello, World!')\n"
        "    return False\n"
        ">>>>>>> REPLACE"
    )
    expected = "def hello():\n    print('Hello, World!')\n    return False\n"
    result = construct_new_file_content(diff, original, True)
    assert result == expected

    # Test line trim match
    original = "def hello():  \n    print('Hello')  \n    return True  "
    result = construct_new_file_content(diff, original, True)
    assert result == expected

    # Test block anchor match
    original = "def hello():\n    # コメント\n    print('Hello')\n    return True"
    result = construct_new_file_content(diff, original, True)
    assert result == expected

    # Test new file creation
    original = ""
    diff = (
        "<<<<<<< SEARCH\n"
        "=======\n"
        "def hello():\n"
        "    print('Hello, World!')\n"
        "    return False\n"
        ">>>>>>> REPLACE"
    )
    result = construct_new_file_content(diff, original, True)
    assert result == "def hello():\n    print('Hello, World!')\n    return False\n"

    # Test full file replacement
    original = "def old():\n    print('Old')\n    return None"
    result = construct_new_file_content(diff, original, True)
    assert result == "def hello():\n    print('Hello, World!')\n    return False\n"

    # Test no match error
    original = "def hello():\n    print('Hello')\n    return True"
    diff = (
        "<<<<<<< SEARCH\n"
        "def goodbye():\n"
        "    print('Goodbye')\n"
        "    return False\n"
        "=======\n"
        "def hello():\n"
        "    print('Hello, World!')\n"
        "    return False\n"
        ">>>>>>> REPLACE"
    )
    with pytest.raises(ValueError, match="does not match anything in the file"):
        construct_new_file_content(diff, original, True)


def test_edit_file_tool(temp_dir: Path):
    """Test EditFile tool with new diff format"""
    # Create test file
    file_path = temp_dir / "test.py"
    original_content = "def hello():\n    print('Hello')\n    return True"
    file_path.write_text(original_content)

    tool = EditFile()

    # Test successful edit
    diff = (
        "<<<<<<< SEARCH\n"
        "def hello():\n"
        "    print('Hello')\n"
        "    return True\n"
        "=======\n"
        "def hello():\n"
        "    print('Hello, World!')\n"
        "    return False\n"
        ">>>>>>> REPLACE"
    )

    result = tool.execute(path=str(file_path), diff=diff)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "success"
    assert (
        file_path.read_text()
        == "def hello():\n    print('Hello, World!')\n    return False\n"
    )

    # Test non-matching search block
    diff = (
        "<<<<<<< SEARCH\n"
        "def nonexistent():\n"
        "    pass\n"
        "=======\n"
        "def hello():\n"
        "    print('Hi!')\n"
        "    return None\n"
        ">>>>>>> REPLACE"
    )

    result = tool.execute(path=str(file_path), diff=diff)
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"
    assert "does not match anything in the file" in result_dict["reason"]

    # Test invalid diff format
    result = tool.execute(path=str(file_path), diff="Invalid diff content")
    result_dict = json.loads(result.content)
    assert result_dict["status"] == "error"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
