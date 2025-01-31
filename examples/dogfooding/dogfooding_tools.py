import glob
import json
import logging
import re
import subprocess
from typing import Any, Dict

from tqdm import tqdm

from prompttrail.agent.subroutine.session_init_strategy import CleanSessionStrategy
from prompttrail.agent.subroutine.squash_strategy import LastMessageStrategy
from prompttrail.agent.templates import (
    ExecuteToolTemplate,
    LinearTemplate,
    SystemTemplate,
)
from prompttrail.agent.templates._core import AssistantTemplate
from prompttrail.agent.tools import SubroutineTool, Tool, ToolArgument, ToolResult
from prompttrail.agent.tools.builtin import ExecuteCommand


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

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        return ToolResult(content={"result": load_all_important_files()})


class RunTest(Tool):
    name: str = "run_test"
    description: str = "Run standard pytest tests in the project using rye run test"
    arguments: Dict[str, ToolArgument[Any]] = {
        "output": ToolArgument(
            name="output",
            description="Output text.",
            value_type=str,
            required=False,
        )
    }

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        return ExecuteCommand().execute(command="rye run test")


class RunAllTests(Tool):
    name: str = "run_all_tests"
    description: str = (
        "Run all tests and checks (rye run all) and extract failed test results"
    )
    arguments: Dict[str, ToolArgument[Any]] = {
        "output": ToolArgument(
            name="output",
            description="Output text.",
            value_type=str,
            required=False,
        )
    }

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        try:
            # Run rye run all
            result = subprocess.run(
                "rye run all",
                shell=True,
                text=True,
                capture_output=True,
            )

            # Extract failed test information
            failed_tests = []
            if result.returncode != 0:
                # Parse pytest output
                pytest_output = result.stdout + result.stderr
                test_failures = re.findall(
                    r"(FAILED|ERROR) (.+?)::(.+?) - (.+?)(?=\n|$)",
                    pytest_output,
                    re.MULTILINE,
                )
                for failure_type, module, test, error in test_failures:
                    failed_tests.append(
                        {
                            "type": failure_type,
                            "module": module,
                            "test": test,
                            "error": error.strip(),
                        }
                    )

                # Parse mypy errors
                mypy_errors = re.findall(
                    r"(.+?):(\d+): error: (.+?)(?=\n|$)",
                    pytest_output,
                    re.MULTILINE,
                )
                for file, line, error in mypy_errors:
                    failed_tests.append(
                        {
                            "type": "MYPY",
                            "module": file,
                            "line": line,
                            "error": error.strip(),
                        }
                    )

                # Parse flake8 errors
                flake8_errors = re.findall(
                    r"(.+?):(\d+):(\d+): (\w+) (.+?)(?=\n|$)",
                    pytest_output,
                    re.MULTILINE,
                )
                for file, line, col, code, error in flake8_errors:
                    failed_tests.append(
                        {
                            "type": "FLAKE8",
                            "module": file,
                            "line": line,
                            "column": col,
                            "code": code,
                            "error": error.strip(),
                        }
                    )

            return ToolResult(
                content=json.dumps(
                    {
                        "status": "success" if result.returncode == 0 else "error",
                        "failed_tests": failed_tests,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
            )

        except Exception as e:
            return ToolResult(
                content=json.dumps(
                    {
                        "status": "error",
                        "reason": str(e),
                    }
                )
            )


RunAllTestsWithSummaryTemplate = LinearTemplate(
    [
        SystemTemplate(
            """
You're given the result of the tests of project. You're given mypy, flake8, and pytest results.
The log is too long. So you should omit the unnecessary lines and show only the failed tests.
If everything is okay, you can just say "All tests passed successfully."
"""
        ),
        ExecuteToolTemplate(
            tool=RunAllTests(),
        ),
        AssistantTemplate(),
    ]
)

RunAllTestsWithSummary = SubroutineTool(
    name="run_all_tests_with_summary",
    description="Run all tests and checks and extract failed test results",
    template=RunAllTestsWithSummaryTemplate,
    session_init_strategy=CleanSessionStrategy(),
    squash_strategy=LastMessageStrategy(),
)
