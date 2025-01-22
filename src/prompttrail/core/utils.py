import inspect
import logging
import os
import sys
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from prompttrail.agent.hooks import Hook
    from prompttrail.core import Session


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
    session: "Session",
    message: str,
    level: int = logging.DEBUG,
):
    """
    Log a message with the specified hook and session.

    Args:
        hook (Hook): The hook to use.
        session (Session): The session to use.
        message (str): The message to log.
        level (int, optional): The log level. Defaults to logging.DEBUG.
    """
    template_id = session.get_current_template_id()
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


logging.basicConfig(
    level=logging.DEBUG,
    # Remove %(name)s and add %(filename)s:%(lineno)d in the desired position:
    format="%(levelname)-8s %(filename)s:%(lineno)d %(message)s",
)


class Loggable:
    def __init__(self):
        # Use __name__ so the logger name matches your module name (e.g. "utils")
        self.logger = logging.getLogger(__name__)
        self.enable_logging = True

    def setup_logger_for_pydantic(self):
        """Setup logger for the class. This should be called after pydantic initialization. You should't remove this method unless you have a good reason."""
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(__name__)
        if not hasattr(self, "enable_logging"):
            self.enable_logging = True

    def disable_log(self):
        if hasattr(self, "logger"):
            self.logger.disabled = True

    def log(self, level: int, msg: str, *args, **kwargs):
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back  # in debug/info/etc.
        if frame is not None:
            frame = frame.f_back  # in real caller
        if frame is None:
            func_name = "unknown"
        else:
            func_name = frame.f_code.co_name

        # Insert [ClassName.functionName] into the final message
        self.logger.log(
            level, "[%s.%s] " + msg, self.__class__.__name__, func_name, *args, **kwargs
        )

    def debug(self, msg: str, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)
