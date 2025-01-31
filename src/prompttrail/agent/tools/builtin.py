import json
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

from prompttrail.agent.tools import Tool, ToolArgument, ToolResult


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

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        path = args["path"]
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
    description: str = "Display directory structure from cwd in a tree-like format"
    arguments: Dict[str, ToolArgument[Any]] = {
        "max_depth": ToolArgument(
            name="max_depth",
            description="Maximum depth of directory traversal",
            value_type=int,
            required=False,
        ),
        "root_dir": ToolArgument(
            name="root_dir",
            description="Root directory to display. Deafult is current working directory. Respect .gitignore if it exists in the root directory.",
            value_type=str,
            required=False,
        ),
    }

    def run_command(self, max_depth: Optional[int], root_dir: Optional[str]) -> str:
        """Get ignore patterns from .gitignore file."""

        depth_part = f"-L {max_depth}" if max_depth else ""

        # https://gist.github.com/jpwilliams/dabff82b0ceb95dd57a7552ea7f2d675

        # TODO: check if .gitignore exists in the root directory etc...

        return subprocess.check_output(
            " ".join(
                [
                    f"tree {root_dir}",
                    depth_part,
                    "-C -I \"$( (cat .gitignore 2>/dev/null || cat $(git rev-parse --show-toplevel 2>/dev/null)/.gitignore 2>/dev/null) | grep -Ev '^#.*$|^[[:space:]]*$' | tr '\\n' '|' | rev | cut -c 2- | rev )\"",
                ]
            ),
            shell=True,
            text=True,
        ).strip()

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        # Convert max_depth to integer
        try:
            max_depth = int(args["max_depth"]) if "max_depth" in args else None
            root_dir = args["root_dir"] if "root_dir" in args else "."
            # Check if root_dir exists
            if not os.path.exists(root_dir):
                return ToolResult(
                    content=json.dumps(
                        {
                            "status": "error",
                            "reason": f"Error opening directory. Please check if the directory exists: {root_dir}",
                        }
                    )
                )
        except ValueError:
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "reason": "max_depth must be a valid integer or root_dir must be a valid string.",
                    }
                )
            )

        try:
            # check if tree command is installed
            subprocess.check_output("tree --version", shell=True)
        except subprocess.CalledProcessError:
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "reason": "tree command is not installed. Please install tree command to use this tool.",
                    }
                )
            )
        try:
            result = self.run_command(max_depth, root_dir)
            if "error opening dir" in result:
                return ToolResult(
                    content=json.dumps(
                        {
                            "status": "error",
                            "reason": "Error opening directory. Please check if the directory exists.\n"
                            + result,
                        }
                    )
                )
            return ToolResult(content=json.dumps({"status": "success", "tree": result}))
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

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        path = args["path"]
        content = args["content"]
        cwd = os.getcwd()
        abs_path = os.path.abspath(path)
        temp_dir = tempfile.gettempdir()

        self.debug("Current working directory: %s", cwd)
        self.debug("Temporary directory: %s", temp_dir)
        self.debug("Absolute path: %s", abs_path)
        self.debug("Path starts with cwd? %s", abs_path.startswith(cwd))
        self.debug("Path starts with temp? %s", abs_path.startswith(temp_dir))

        if not (abs_path.startswith(cwd) or abs_path.startswith(temp_dir)):
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
            self.debug("Creating directory: %s", dirname)
            os.makedirs(dirname, exist_ok=True)

            self.debug("Writing file: %s", path)
            with open(path, "w") as f:
                f.write(content)

            self.debug("File written successfully")
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "success",
                        "message": f"File written successfully: {path}",
                    }
                )
            )
        except Exception as e:
            self.debug("Error writing file: %s", str(e))
            return ToolResult(content=json.dumps({"status": "error", "reason": str(e)}))


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

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        path = args["path"]
        diff = args["diff"]

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
                        {
                            "status": "error",
                            "reason": "Invalid diff format. Missing any of the markers: <<<<<<< SEARCH, =======, >>>>>>> REPLACE. You can rewrite content using CreateOrOverwriteFile tool.",
                        }
                    )
                )

            # Apply diff using construct_new_file_content
            new_content = self.construct_new_file_content(diff, content)

            # Write updated content
            with open(path, "w") as f:
                f.write(new_content)

            return ToolResult(
                content=json.dumps(
                    {"status": "success", "message": "File edited successfully"}
                )
            )
        except ValueError as e:
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "reason": str(e)
                        + " If this error persists you can rewrite content using CreateOrOverwriteFile tool.",
                    }
                )
            )
        except Exception as e:
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "reason": str(e)
                        + " If this error persists you can rewrite content using CreateOrOverwriteFile tool.",
                    }
                )
            )

    def construct_new_file_content(
        self, diff_content: str, original_content: str, is_final: bool = True
    ) -> str:
        """
        Apply diff content to original file content to construct new file content.
        """
        self.debug("=== construct_new_file_content ===")
        self.debug("Original content: %s", repr(original_content))
        self.debug("Diff content: %s", repr(diff_content))
        self.debug("Is final: %s", is_final)

        result = ""
        last_processed_index = 0

        current_search_content = ""
        current_replace_content = ""
        in_search = False
        in_replace = False

        search_match_index = -1
        search_end_index = -1

        lines = diff_content.split("\n")
        self.debug("Number of lines in diff: %s", len(lines))

        # Remove last line if it's an incomplete marker
        if (
            lines
            and any(lines[-1].startswith(p) for p in ["<", "=", ">"])
            and lines[-1] not in ["<<<<<<< SEARCH", "=======", ">>>>>>> REPLACE"]
        ):
            lines.pop()
            self.debug("Removed incomplete marker line")

        for line in lines:
            self.debug("Processing line: %s", repr(line))
            if line == "<<<<<<< SEARCH":
                self.debug("Found SEARCH marker")
                in_search = True
                current_search_content = ""
                current_replace_content = ""
                continue

            if line == "=======":
                self.debug("Found SEPARATOR marker")
                in_search = False
                in_replace = True

                self.debug("Current search content: %s", repr(current_search_content))
                if not current_search_content:
                    self.debug("Empty search block")
                    # Empty search block
                    if len(original_content) == 0:
                        # New file scenario
                        search_match_index = 0
                        search_end_index = 0
                        self.debug("New file scenario")
                    else:
                        # Full file replacement scenario
                        search_match_index = 0
                        search_end_index = len(original_content)
                        self.debug("Full file replacement scenario")
                else:
                    # Exact match search
                    exact_index = original_content.find(
                        current_search_content, last_processed_index
                    )
                    self.debug("Exact match result: %s", exact_index)
                    if exact_index != -1:
                        search_match_index = exact_index
                        search_end_index = exact_index + len(current_search_content)
                    else:
                        # Try line trim match
                        line_match = self.line_trimmed_fallback_match(
                            original_content,
                            current_search_content,
                            last_processed_index,
                        )
                        self.debug("Line trim match result: %s", line_match)
                        if isinstance(line_match, tuple):
                            search_match_index, search_end_index = line_match
                        else:
                            # Try block anchor match
                            block_match = self.block_anchor_fallback_match(
                                original_content,
                                current_search_content,
                                last_processed_index,
                            )
                            self.debug("Block anchor match result: %s", block_match)
                            if isinstance(block_match, tuple):
                                search_match_index, search_end_index = block_match
                            else:
                                raise ValueError(
                                    f"The SEARCH block:\n{current_search_content.rstrip()}\n...does not match anything in the file."
                                )

                # Output content up to match position
                result += original_content[last_processed_index:search_match_index]
                self.debug("Added content up to match")
                continue

            if line == ">>>>>>> REPLACE":
                self.debug("Found REPLACE marker")
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
                self.debug("Added to search content: %s", repr(line))
            elif in_replace:
                current_replace_content += line + "\n"
                self.debug("Added to replace content: %s", repr(line))
                # If match position is known, output replacement line immediately
                if search_match_index != -1:
                    result += line + "\n"
                    self.debug("Added replacement line")

        # For the last chunk, add remaining original content
        if is_final and last_processed_index < len(original_content):
            result += original_content[last_processed_index:]
            self.debug("Added remaining content")

        final_result = result.rstrip() + "\n"
        self.debug("Final result: %s", repr(final_result))
        return final_result

    def line_trimmed_fallback_match(
        self, original_content: str, search_content: str, start_index: int
    ) -> Union[Tuple[int, int], bool]:
        """
        Try line-by-line trimmed fallback matching.
        Compare lines ignoring whitespace at the beginning and end.
        """
        self.debug("=== line_trimmed_fallback_match ===")
        self.debug("Original content: %s", repr(original_content))
        self.debug("Search content: %s", repr(search_content))
        self.debug("Start index: %s", start_index)

        # Split both contents into lines
        original_lines = original_content.split("\n")
        search_lines = search_content.split("\n")

        self.debug("Original lines: %s", len(original_lines))
        self.debug("Search lines: %s", len(search_lines))

        # Remove trailing empty lines
        if search_lines and search_lines[-1] == "":
            search_lines.pop()
            self.debug("Removed empty last line from search content")

        # Find the line number that contains start_index
        start_line_num = 0
        current_index = 0
        while current_index < start_index and start_line_num < len(original_lines):
            current_index += len(original_lines[start_line_num]) + 1
            start_line_num += 1

        self.debug("Starting search from line: %s", start_line_num)

        # Try matching at each starting position in original content
        for i in range(start_line_num, len(original_lines) - len(search_lines) + 1):
            matches = True
            self.debug("Trying match at line %s", i)

            # Check if all search lines match from this position
            for j in range(len(search_lines)):
                original_trimmed = original_lines[i + j].strip()
                search_trimmed = search_lines[j].strip()

                self.debug("  Comparing line %s:", j)
                self.debug("    Original: %s", repr(original_trimmed))
                self.debug("    Search:   %s", repr(search_trimmed))

                if original_trimmed != search_trimmed:
                    matches = False
                    self.debug("    No match")
                    break
                self.debug("    Match found")

            # If match found, calculate exact character positions
            if matches:
                self.debug("Full match found at line %s", i)
                # Calculate start character position
                match_start_index = 0
                for k in range(i):
                    match_start_index += len(original_lines[k]) + 1

                # Calculate end character position
                match_end_index = match_start_index
                for k in range(len(search_lines)):
                    match_end_index += len(original_lines[i + k]) + 1

                self.debug(
                    "Match positions: %s, %s", match_start_index, match_end_index
                )
                return (match_start_index, match_end_index)

        self.debug("No match found")
        return False

    def block_anchor_fallback_match(
        self, original_content: str, search_content: str, start_index: int
    ) -> Union[Tuple[int, int], bool]:
        """
        Try fallback matching using first and last lines of block as anchors.
        Only applies to blocks with 3 or more lines.
        """
        self.debug("=== block_anchor_fallback_match ===")
        self.debug("Original content: %s", repr(original_content))
        self.debug("Search content: %s", repr(search_content))
        self.debug("Start index: %s", start_index)

        original_lines = original_content.split("\n")
        search_lines = search_content.split("\n")

        # Don't use for blocks with less than 3 lines
        if len(search_lines) < 3:
            self.debug("Search block too short (<3 lines)")
            return False

        # Remove trailing empty lines
        if search_lines and search_lines[-1] == "":
            search_lines.pop()
            self.debug("Removed empty last line from search content")

        first_line_search = search_lines[0].strip()
        last_line_search = search_lines[-1].strip()
        search_block_size = len(search_lines)

        self.debug("First line: %s", repr(first_line_search))
        self.debug("Last line: %s", repr(last_line_search))
        self.debug("Block size: %s", search_block_size)

        # Find the line number that contains start_index
        start_line_num = 0
        current_index = 0
        while current_index < start_index and start_line_num < len(original_lines):
            current_index += len(original_lines[start_line_num]) + 1
            start_line_num += 1

        self.debug("Starting search from line: %s", start_line_num)

        # Search for matching start and end lines
        for i in range(start_line_num, len(original_lines) - search_block_size + 1):
            self.debug("Trying match at line %s", i)
            # Check if first line matches
            if original_lines[i].strip() != first_line_search:
                self.debug("First line doesn't match")
                continue

            # Check if last line matches at expected position
            end_line_index = i + search_block_size - 1
            while end_line_index < len(original_lines):
                if original_lines[end_line_index].strip() == last_line_search:
                    self.debug("Found matching block")

                    # Calculate exact character positions
                    match_start_index = 0
                    for k in range(i):
                        match_start_index += len(original_lines[k]) + 1

                    match_end_index = match_start_index
                    for k in range(end_line_index - i + 1):
                        match_end_index += len(original_lines[i + k]) + 1

                    self.debug(
                        "Match positions: %s, %s", match_start_index, match_end_index
                    )
                    return (match_start_index, match_end_index)
                end_line_index += 1

            self.debug("Last line doesn't match")

        self.debug("No match found")
        return False


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

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        if "command" not in args:
            return ToolResult(
                content=json.dumps(
                    {"status": "error", "reason": "No command was passed."}
                )
            )

        # execute the command and get stdout and stderr
        command = args["command"]
        stdout_text = ""
        stderr_text = ""
        is_error = False
        try:
            stdout_text = subprocess.check_output(command, shell=True, text=True)
        except subprocess.CalledProcessError as e:
            stderr_text = e.stderr
            is_error = True
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
