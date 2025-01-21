import glob
import json
import logging
import os
import tempfile
from typing import Any, Dict, Tuple, Union

from tqdm import tqdm

from prompttrail.agent.tools import Tool, ToolArgument, ToolResult


def print_debug(*args, **kwargs):
    """Helper function for debug output"""
    print("[DEBUG]", *args, **kwargs)


def disable_noisy_loggers():
    """Disable the specified loggers."""
    # httpx
    logging.getLogger("httpx").setLevel(logging.ERROR)
    # httpcore
    logging.getLogger("httpcore").setLevel(logging.ERROR)
    # openai
    logging.getLogger("openai").setLevel(logging.ERROR)
    # anthropic
    logging.getLogger("anthropic").setLevel(logging.ERROR)


def load_all_important_files():
    text = ""
    for file in tqdm(list(glob.glob("examples/**/*.py", recursive=True))):
        text += f"Example filename: {file}\n"
        text += f"```python\n{open(file, 'r').read()}\n```\n"

    for file in tqdm(list(glob.glob("tests/**/*.py", recursive=True))):
        text += f"Test filename: {file}\n"
        text += f"```python\n{open(file, 'r').read()}\n```\n"

    # add README.md content
    text += f"```README\n{open('README.md', 'r').read()}\n```\n"

    # add docs *.md content
    for file in tqdm(list(glob.glob("docs/*.md", recursive=False))):
        text += f"Docs filename: {file}\n"
        text += f"```markdown\n{open(file, 'r').read()}\n```\n"

    return text


class ReadImportantFiles(Tool):
    name: str = "read_all_important_files"
    description: str = (
        "Read all files in src/ tests/ examples/ and docs/ and return the content"
    )
    arguments: Dict[str, ToolArgument[Any]] = {
        "output": ToolArgument(
            name="output",
            description="Output text.",
            value_type=str,
            required=False,
        )
    }

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(content={"result": load_all_important_files()})


class ExecuteCommand(Tool):
    name: str = "execute_command"
    description: str = "You can run any linux command with arguments like: ls, git status, git commit --amend --no-edit. Dangerous operation is reviewed by user."
    arguments: Dict[str, ToolArgument[Any]] = {
        "command": ToolArgument(
            name="command",
            description="Command to run.",
            value_type=str,
            required=False,
        )
    }

    def execute(self, **kwargs) -> ToolResult:
        if "command" not in kwargs:
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "No command was passed."}
                )
            )
        # execute the command and get stdout and stderr
        import subprocess

        command = kwargs["command"]
        stdout_text = ""
        stderr_text = ""
        is_error = False
        try:
            stdout_text = subprocess.check_output(command, shell=True, text=True)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr
            is_error = True
        finally:
            if is_error:
                return ToolResult(
                    content=json.dumps(
                        {
                            "status": "error",
                            "reason": "command exited with non-zero status",
                            "stderr": stderr_text,
                            "stdout": stdout_text,
                        }
                    )
                )
            else:
                return ToolResult(
                    content=json.dumps({"status": "success", "stdout": stdout_text})
                )


class ReadFile(Tool):
    name: str = "read_file"
    description: str = "Read content of a specific file"
    arguments: Dict[str, ToolArgument[Any]] = {
        "path": ToolArgument(
            name="path",
            description="Path to the file to read",
            value_type=str,
            required=True,
        )
    }

    def execute(self, **kwargs) -> ToolResult:
        if "path" not in kwargs:
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "No path was provided."}
                )
            )

        path = kwargs["path"]
        try:
            with open(path, "r") as f:
                content = f.read()
            return ToolResult(
                content=json.dumps({"status": "success", "content": content})
            )
        except Exception as e:
            return ToolResult(content=json.dumps({"status": "error", "reason": str(e)}))


class TreeDirectory(Tool):
    name: str = "tree_directory"
    description: str = "Display directory structure in a tree-like format"
    arguments: Dict[str, ToolArgument[Any]] = {
        "path": ToolArgument(
            name="path",
            description="Path to the directory",
            value_type=str,
            required=True,
        ),
        "max_depth": ToolArgument(
            name="max_depth",
            description="Maximum depth of directory traversal",
            value_type=int,
            required=False,
        ),
    }

    def _get_gitignore_patterns(self) -> str:
        """Get ignore patterns from .gitignore file."""
        import subprocess

        try:
            # First, try .gitignore in current directory
            gitignore_content = subprocess.check_output(
                "cat .gitignore 2> /dev/null || "
                "cat $(git rev-parse --show-toplevel 2> /dev/null)/.gitignore 2> /dev/null || "
                "echo 'node_modules'",
                shell=True,
                text=True,
            )

            # Process lines
            patterns = []
            for line in gitignore_content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # Normalize patterns
                    if line.endswith("/"):
                        # Directory pattern
                        base = line.rstrip("/")
                        patterns.append(base)
                        patterns.append(f"{base}/*")  # All files in directory
                    elif "*" in line:
                        # Wildcard pattern
                        if "/" in line:
                            # Wildcard with path
                            patterns.append(line)
                        else:
                            # Global wildcard
                            patterns.append(f"*/{line}")  # Match in any directory
                            patterns.append(line)  # Match at top level
                    else:
                        # Normal pattern
                        patterns.append(line)
                        if "/" in line:
                            # For path patterns, ignore everything under that path
                            patterns.append(f"{line}/*")

            # Combine patterns with | and escape single quotes
            combined_patterns = "|".join(p.replace("'", "\\'") for p in patterns)
            return combined_patterns

        except subprocess.CalledProcessError:
            return "node_modules"

    def _generate_tree(
        self,
        path: str,
        prefix: str = "",
        max_depth: int | None = None,
        current_depth: int = 0,
    ) -> Dict[str, str]:
        """Generate tree structure using the tree command."""
        import shutil
        import subprocess

        # Check if tree command exists
        if shutil.which("tree") is None:
            return {
                "status": "error",
                "tree": "Error: 'tree' command not found. Please install it first.",
            }

        try:
            # Temporarily change current directory for execution
            current_dir = os.getcwd()
            os.chdir(path)

            try:
                # Prepare max_depth option
                depth_opt = f"-L {max_depth + 1}" if max_depth is not None else ""

                # Get and format ignore patterns
                ignore_patterns = self._get_gitignore_patterns()
                ignore_patterns = ignore_patterns.replace(
                    "'", "\\'"
                )  # Escape single quotes

                # Execute tree command (using relative path)
                # --noreport: Don't show summary line
                # -a: Show hidden files
                # --prune: Don't show empty directories
                # --matchdirs: Apply pattern matching to directories
                # --charset=ascii: Use ASCII characters only
                # -d: Show only directories (when max_depth=1)
                if max_depth == 1:
                    cmd = (
                        f"tree --noreport -a --prune -d "
                        f"--matchdirs --charset=ascii -I '{ignore_patterns}' ."
                    )
                else:
                    cmd = (
                        f"tree --noreport {depth_opt} -a --prune "
                        f"--matchdirs --charset=ascii -I '{ignore_patterns}' ."
                    )
                output = subprocess.check_output(cmd, shell=True, text=True)

                # Remove first line (current directory) and empty lines
                tree_lines = [
                    line
                    for line in output.split("\n")[1:]
                    if line.strip() and not line.endswith("/")
                ]

                # Remove color codes
                import re

                tree_lines = [
                    re.sub(r"\x1b\[[0-9;]*[mK]", "", line) for line in tree_lines
                ]

                # Exclude .gitignore (when max_depth=1)
                if max_depth == 1:
                    tree_lines = [
                        line for line in tree_lines if ".gitignore" not in line
                    ]

                return {"status": "success", "tree": "\n".join(tree_lines) + "\n"}

            finally:
                # Always restore current directory
                os.chdir(current_dir)

        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "tree": f"Error executing tree command: {str(e)}",
            }

    def execute(self, **kwargs) -> ToolResult:
        if "path" not in kwargs:
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "No path was provided."}
                )
            )

        path = kwargs["path"]
        # Convert max_depth to integer
        try:
            max_depth = int(kwargs["max_depth"]) if "max_depth" in kwargs else None
        except ValueError:
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "max_depth must be a valid integer"}
                )
            )

        try:
            result = self._generate_tree(path, max_depth=max_depth)
            return ToolResult(content=json.dumps(result))
        except FileNotFoundError as e:
            return ToolResult(
                content=json.dumps({"status": "error", "tree": f"Error: {str(e)}"})
            )
        except Exception as e:
            return ToolResult(
                content=json.dumps({"status": "error", "tree": f"Error: {str(e)}"})
            )


class CreateOrOverwriteFile(Tool):
    name: str = "create_or_overwrite_file"
    description: str = "Create a new file or overwrite an existing one"
    arguments: Dict[str, ToolArgument[Any]] = {
        "path": ToolArgument(
            name="path",
            description="Path where to create/overwrite the file",
            value_type=str,
            required=True,
        ),
        "content": ToolArgument(
            name="content",
            description="Content to write to the file",
            value_type=str,
            required=True,
        ),
    }

    def execute(self, **kwargs) -> ToolResult:
        print_debug("=== CreateOrOverwriteFile.execute ===")
        print_debug("kwargs:", kwargs)

        if "path" not in kwargs or "content" not in kwargs:
            print_debug("Missing required arguments")
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "Both path and content are required."}
                )
            )

        # Validate that path is under current working directory or temp directory
        path = kwargs["path"]
        content = kwargs["content"]
        cwd = os.getcwd()
        abs_path = os.path.abspath(path)
        temp_dir = tempfile.gettempdir()

        print_debug("Current working directory:", cwd)
        print_debug("Temporary directory:", temp_dir)
        print_debug("Absolute path:", abs_path)
        print_debug("Path starts with cwd?", abs_path.startswith(cwd))
        print_debug("Path starts with temp?", abs_path.startswith(temp_dir))

        if not (abs_path.startswith(cwd) or abs_path.startswith(temp_dir)):
            print_debug("Path is not under current working directory or temp directory")
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "reason": "Path must be under current working directory or temp directory.",
                    }
                )
            )

        try:
            dirname = os.path.dirname(path)
            print_debug("Creating directory:", dirname)
            os.makedirs(dirname, exist_ok=True)

            print_debug("Writing file:", path)
            with open(path, "w") as f:
                f.write(content)

            print_debug("File written successfully")
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": f"File written successfully: {path}",
                    }
                )
            )
        except Exception as e:
            print_debug("Error writing file:", str(e))
            return ToolResult(content=json.dumps({"status": "error", "reason": str(e)}))


def line_trimmed_fallback_match(
    original_content: str, search_content: str, start_index: int
) -> Union[Tuple[int, int], bool]:
    """
    Try line-by-line trimmed fallback matching.
    Compare lines ignoring whitespace at the beginning and end.
    """
    print_debug("=== line_trimmed_fallback_match ===")
    print_debug("Original content:", repr(original_content))
    print_debug("Search content:", repr(search_content))
    print_debug("Start index:", start_index)

    # Split both contents into lines
    original_lines = original_content.split("\n")
    search_lines = search_content.split("\n")

    print_debug("Original lines:", len(original_lines))
    print_debug("Search lines:", len(search_lines))

    # Remove trailing empty lines
    if search_lines and search_lines[-1] == "":
        search_lines.pop()
        print_debug("Removed empty last line from search content")

    # Find the line number that contains start_index
    start_line_num = 0
    current_index = 0
    while current_index < start_index and start_line_num < len(original_lines):
        current_index += len(original_lines[start_line_num]) + 1
        start_line_num += 1

    print_debug("Starting search from line:", start_line_num)

    # Try matching at each starting position in original content
    for i in range(start_line_num, len(original_lines) - len(search_lines) + 1):
        matches = True
        print_debug(f"Trying match at line {i}")

        # Check if all search lines match from this position
        for j in range(len(search_lines)):
            original_trimmed = original_lines[i + j].strip()
            search_trimmed = search_lines[j].strip()

            print_debug(f"  Comparing line {j}:")
            print_debug(f"    Original: {repr(original_trimmed)}")
            print_debug(f"    Search:   {repr(search_trimmed)}")

            if original_trimmed != search_trimmed:
                matches = False
                print_debug("    No match")
                break
            print_debug("    Match found")

        # If match found, calculate exact character positions
        if matches:
            print_debug("Full match found at line", i)
            # Calculate start character position
            match_start_index = 0
            for k in range(i):
                match_start_index += len(original_lines[k]) + 1

            # Calculate end character position
            match_end_index = match_start_index
            for k in range(len(search_lines)):
                match_end_index += len(original_lines[i + k]) + 1

            print_debug("Match positions:", match_start_index, match_end_index)
            return (match_start_index, match_end_index)

    print_debug("No match found")
    return False


def block_anchor_fallback_match(
    original_content: str, search_content: str, start_index: int
) -> Union[Tuple[int, int], bool]:
    """
    Try fallback matching using first and last lines of block as anchors.
    Only applies to blocks with 3 or more lines.
    """
    print_debug("=== block_anchor_fallback_match ===")
    print_debug("Original content:", repr(original_content))
    print_debug("Search content:", repr(search_content))
    print_debug("Start index:", start_index)

    original_lines = original_content.split("\n")
    search_lines = search_content.split("\n")

    # Don't use for blocks with less than 3 lines
    if len(search_lines) < 3:
        print_debug("Search block too short (<3 lines)")
        return False

    # Remove trailing empty lines
    if search_lines and search_lines[-1] == "":
        search_lines.pop()
        print_debug("Removed empty last line from search content")

    first_line_search = search_lines[0].strip()
    last_line_search = search_lines[-1].strip()
    search_block_size = len(search_lines)

    print_debug("First line:", repr(first_line_search))
    print_debug("Last line:", repr(last_line_search))
    print_debug("Block size:", search_block_size)

    # Find the line number that contains start_index
    start_line_num = 0
    current_index = 0
    while current_index < start_index and start_line_num < len(original_lines):
        current_index += len(original_lines[start_line_num]) + 1
        start_line_num += 1

    print_debug("Starting search from line:", start_line_num)

    # Search for matching start and end lines
    for i in range(start_line_num, len(original_lines) - search_block_size + 1):
        print_debug(f"Trying match at line {i}")
        # Check if first line matches
        if original_lines[i].strip() != first_line_search:
            print_debug("First line doesn't match")
            continue

        # Check if last line matches at expected position
        end_line_index = i + search_block_size - 1
        while end_line_index < len(original_lines):
            if original_lines[end_line_index].strip() == last_line_search:
                print_debug("Found matching block")

                # Calculate exact character positions
                match_start_index = 0
                for k in range(i):
                    match_start_index += len(original_lines[k]) + 1

                match_end_index = match_start_index
                for k in range(end_line_index - i + 1):
                    match_end_index += len(original_lines[i + k]) + 1

                print_debug("Match positions:", match_start_index, match_end_index)
                return (match_start_index, match_end_index)
            end_line_index += 1

        print_debug("Last line doesn't match")

    print_debug("No match found")
    return False


def construct_new_file_content(
    diff_content: str, original_content: str, is_final: bool
) -> str:
    """
    Apply diff content to original file content to construct new file content.
    """
    print_debug("=== construct_new_file_content ===")
    print_debug("Original content:", repr(original_content))
    print_debug("Diff content:", repr(diff_content))
    print_debug("Is final:", is_final)

    result = ""
    last_processed_index = 0

    current_search_content = ""
    current_replace_content = ""
    in_search = False
    in_replace = False

    search_match_index = -1
    search_end_index = -1

    lines = diff_content.split("\n")
    print_debug("Number of lines in diff:", len(lines))

    # Remove last line if it's an incomplete marker
    if (
        lines
        and any(lines[-1].startswith(p) for p in ["<", "=", ">"])
        and lines[-1] not in ["<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"]
    ):
        lines.pop()
        print_debug("Removed incomplete marker line")

    for line in lines:
        print_debug("Processing line:", repr(line))
        if line == "<<<<<<< SEARCH":
            print_debug("Found SEARCH marker")
            in_search = True
            current_search_content = ""
            current_replace_content = ""
            continue

        if line == "=======":
            print_debug("Found SEPARATOR marker")
            in_search = False
            in_replace = True

            print_debug("Current search content:", repr(current_search_content))
            if not current_search_content:
                print_debug("Empty search block")
                # Empty search block
                if len(original_content) == 0:
                    # New file scenario
                    search_match_index = 0
                    search_end_index = 0
                    print_debug("New file scenario")
                else:
                    # Full file replacement scenario
                    search_match_index = 0
                    search_end_index = len(original_content)
                    print_debug("Full file replacement scenario")
            else:
                # Exact match search
                exact_index = original_content.find(
                    current_search_content, last_processed_index
                )
                print_debug("Exact match result:", exact_index)
                if exact_index != -1:
                    search_match_index = exact_index
                    search_end_index = exact_index + len(current_search_content)
                else:
                    # Try line trim match
                    line_match = line_trimmed_fallback_match(
                        original_content, current_search_content, last_processed_index
                    )
                    print_debug("Line trim match result:", line_match)
                    if isinstance(line_match, tuple):
                        search_match_index, search_end_index = line_match
                    else:
                        # Try block anchor match
                        block_match = block_anchor_fallback_match(
                            original_content,
                            current_search_content,
                            last_processed_index,
                        )
                        print_debug("Block anchor match result:", block_match)
                        if isinstance(block_match, tuple):
                            search_match_index, search_end_index = block_match
                        else:
                            raise ValueError(
                                f"The SEARCH block:\n{current_search_content.rstrip()}\n...does not match anything in the file."
                            )

            # Output content up to match position
            result += original_content[last_processed_index:search_match_index]
            print_debug("Added content up to match")
            continue

        if line == ">>>>>>> REPLACE":
            print_debug("Found REPLACE marker")
            # End of replacement block
            last_processed_index = search_end_index

            # Reset
            in_search = False
            in_replace = False
            current_search_content = ""
            current_replace_content = ""
            search_match_index = -1
            search_end_index = -1
            continue

        # Accumulate search or replace content
        if in_search:
            current_search_content += line + "\n"
            print_debug("Added to search content:", repr(line))
        elif in_replace:
            current_replace_content += line + "\n"
            print_debug("Added to replace content:", repr(line))
            # If match position is known, output replacement line immediately
            if search_match_index != -1:
                result += line + "\n"
                print_debug("Added replacement line")

    # For the last chunk, add remaining original content
    if is_final and last_processed_index < len(original_content):
        result += original_content[last_processed_index:]
        print_debug("Added remaining content")

    final_result = result.rstrip() + "\n"
    print_debug("Final result:", repr(final_result))
    return final_result


class EditFile(Tool):
    name: str = "edit_file"
    description: str = "Edit a file by applying a diff in SEARCH/REPLACE block format"
    arguments: Dict[str, ToolArgument[Any]] = {
        "path": ToolArgument(
            name="path",
            description="Path to the file to edit",
            value_type=str,
            required=True,
        ),
        "diff": ToolArgument(
            name="diff",
            description="Diff content in format: SEARCH/REPLACE blocks",
            value_type=str,
            required=True,
        ),
    }

    def execute(self, **kwargs) -> ToolResult:
        if "path" not in kwargs or "diff" not in kwargs:
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "Both path and diff are required."}
                )
            )

        path = kwargs["path"]
        diff = kwargs["diff"]

        try:
            # Read file content
            with open(path, "r") as f:
                content = f.read()

            # Validate diff format
            if not (
                "<<<<<<< SEARCH" in diff
                and "=======" in diff
                and ">>>>>>> REPLACE" in diff
            ):
                return ToolResult(
                    content=json.dumps(
                        {"status": "error", "reason": "Invalid diff format"}
                    )
                )

            # Apply diff using construct_new_file_content
            new_content = construct_new_file_content(diff, content, True)

            # Write updated content
            with open(path, "w") as f:
                f.write(new_content)

            return ToolResult(
                content=json.dumps(
                    {"status": "success", "message": "File edited successfully"}
                )
            )
        except ValueError as e:
            return ToolResult(content=json.dumps({"status": "error", "reason": str(e)}))
        except Exception as e:
            return ToolResult(content=json.dumps({"status": "error", "reason": str(e)}))
