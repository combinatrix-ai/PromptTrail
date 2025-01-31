import json
from unittest.mock import Mock, patch

from examples.dogfooding.dogfooding_tools import RunAllTests


def test_run_all_tests_success():
    # Mock successful test run
    mock_process = Mock()
    mock_process.returncode = 0
    mock_process.stdout = "All tests passed!"
    mock_process.stderr = ""

    with patch("subprocess.run", return_value=mock_process):
        tool = RunAllTests()
        result = tool.execute()
        result_data = json.loads(result.content)

        assert result_data["status"] == "success"
        assert len(result_data["failed_tests"]) == 0
        assert result_data["stdout"] == "All tests passed!"
        assert result_data["stderr"] == ""


def test_run_all_tests_pytest_failure():
    # Mock failed pytest run
    mock_process = Mock()
    mock_process.returncode = 1
    mock_process.stdout = """
============================= test session starts ==============================
FAILED tests/test_example.py::test_function - AssertionError: expected 1 but got 2
FAILED tests/test_other.py::test_other - ValueError: invalid value
=========================== 2 failed, 3 passed ================================
"""
    mock_process.stderr = ""

    with patch("subprocess.run", return_value=mock_process):
        tool = RunAllTests()
        result = tool.execute()
        result_data = json.loads(result.content)

        assert result_data["status"] == "error"
        assert len(result_data["failed_tests"]) == 2

        # Check first failure
        assert result_data["failed_tests"][0]["type"] == "FAILED"
        assert result_data["failed_tests"][0]["module"] == "tests/test_example.py"
        assert result_data["failed_tests"][0]["test"] == "test_function"
        assert (
            result_data["failed_tests"][0]["error"]
            == "AssertionError: expected 1 but got 2"
        )

        # Check second failure
        assert result_data["failed_tests"][1]["type"] == "FAILED"
        assert result_data["failed_tests"][1]["module"] == "tests/test_other.py"
        assert result_data["failed_tests"][1]["test"] == "test_other"
        assert result_data["failed_tests"][1]["error"] == "ValueError: invalid value"


def test_run_all_tests_mypy_failure():
    # Mock failed mypy run
    mock_process = Mock()
    mock_process.returncode = 1
    mock_process.stdout = """
src/example.py:10: error: Incompatible types in assignment
src/other.py:20: error: Name 'undefined_var' is not defined
"""
    mock_process.stderr = ""

    with patch("subprocess.run", return_value=mock_process):
        tool = RunAllTests()
        result = tool.execute()
        result_data = json.loads(result.content)

        assert result_data["status"] == "error"
        assert len(result_data["failed_tests"]) == 2

        # Check first failure
        assert result_data["failed_tests"][0]["type"] == "MYPY"
        assert result_data["failed_tests"][0]["module"] == "src/example.py"
        assert result_data["failed_tests"][0]["line"] == "10"
        assert (
            result_data["failed_tests"][0]["error"]
            == "Incompatible types in assignment"
        )

        # Check second failure
        assert result_data["failed_tests"][1]["type"] == "MYPY"
        assert result_data["failed_tests"][1]["module"] == "src/other.py"
        assert result_data["failed_tests"][1]["line"] == "20"
        assert (
            result_data["failed_tests"][1]["error"]
            == "Name 'undefined_var' is not defined"
        )


def test_run_all_tests_flake8_failure():
    # Mock failed flake8 run
    mock_process = Mock()
    mock_process.returncode = 1
    mock_process.stdout = """
src/example.py:15:1: E302 expected 2 blank lines, found 1
src/other.py:25:80: E501 line too long (100 > 79 characters)
"""
    mock_process.stderr = ""

    with patch("subprocess.run", return_value=mock_process):
        tool = RunAllTests()
        result = tool.execute()
        result_data = json.loads(result.content)

        assert result_data["status"] == "error"
        assert len(result_data["failed_tests"]) == 2

        # Check first failure
        assert result_data["failed_tests"][0]["type"] == "FLAKE8"
        assert result_data["failed_tests"][0]["module"] == "src/example.py"
        assert result_data["failed_tests"][0]["line"] == "15"
        assert result_data["failed_tests"][0]["column"] == "1"
        assert result_data["failed_tests"][0]["code"] == "E302"
        assert (
            result_data["failed_tests"][0]["error"] == "expected 2 blank lines, found 1"
        )

        # Check second failure
        assert result_data["failed_tests"][1]["type"] == "FLAKE8"
        assert result_data["failed_tests"][1]["module"] == "src/other.py"
        assert result_data["failed_tests"][1]["line"] == "25"
        assert result_data["failed_tests"][1]["column"] == "80"
        assert result_data["failed_tests"][1]["code"] == "E501"
        assert (
            result_data["failed_tests"][1]["error"]
            == "line too long (100 > 79 characters)"
        )


def test_run_all_tests_execution_error():
    # Mock execution error
    with patch("subprocess.run", side_effect=Exception("Command execution failed")):
        tool = RunAllTests()
        result = tool.execute()
        result_data = json.loads(result.content)

        assert result_data["status"] == "error"
        assert result_data["reason"] == "Command execution failed"
