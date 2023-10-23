import logging
import os
import sys
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from prompttrail.agent import State
    from prompttrail.agent.hooks import Hook


def logger_multiline(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    """
    Log a multiline message with the specified logger.

    Args:
        logger (logging.Logger): The logger to use.
        message (str): The message to log.
        level (int, optional): The log level. Defaults to logging.DEBUG.
    """
    for line in message.splitlines():
        logger.log(level, line)


def hook_logger(
    hook: "Hook",
    state: "State",
    message: str,
    level: int = logging.DEBUG,
):
    """
    Log a message with the specified hook and state.

    Args:
        hook (Hook): The hook to use.
        state (State): The state to use.
        message (str): The message to log.
        level (int, optional): The log level. Defaults to logging.DEBUG.
    """
    template_id = state.get_current_template_id()
    logger = logging.getLogger(hook.__class__.__name__ + "@" + str(template_id))
    logger_multiline(logger, message, level)


def is_in_test_env() -> bool:
    """
    Check if the code is running in a test environment.

    Returns:
        bool: True if running in a test environment, False otherwise.
    """
    # We use pytest as the test runner.
    # https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    if "pytest" in sys.modules:
        return True
    if os.environ.get("CI"):
        return True
    if os.environ.get("DEBUG"):
        return True
    return False


def count_tokens(text: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    encoded = encoding.encode(text)
    return len(encoded)
