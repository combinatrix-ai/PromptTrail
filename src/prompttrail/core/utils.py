import inspect
import logging
import os
import sys
from typing import Optional

import tiktoken


def is_in_test_env() -> bool:
    """
    Check if the code is running in a test environment.

    Returns:
        bool: True if running in a test environment, False otherwise.
    """
    if "pytest" in sys.modules:
        return True
    if os.environ.get("CI"):
        return True
    if os.environ.get("DEBUG"):
        return True
    return False


def count_tokens(text: str, encoding_name: str) -> int:
    """
    Count the number of tokens in text using specified encoding.

    Args:
        text: Text to count tokens for
        encoding_name: Name of the tiktoken encoding to use

    Returns:
        Number of tokens in text
    """
    encoding = tiktoken.get_encoding(encoding_name)
    encoded = encoding.encode(text)
    return len(encoded)


class Debuggable:
    """Base class adding logging capabilities to child classes."""

    def __init__(self) -> None:
        """Initialize logging for the class."""
        self.logger: Optional[logging.Logger] = logging.getLogger(__name__)
        self.debug_mode = True

    def setup_logger_for_pydantic(self):
        """
        Setup logger after pydantic initialization.
        Should be called after pydantic initialization.
        """
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(__name__)
        if not hasattr(self, "enable_logging"):
            self.debug_mode = True

    def disable_log(self):
        """Disable logging for this instance."""
        self.debug_mode = False

    def enable_log(self):
        """Enable logging for this instance."""
        self.debug_mode = True

    def log(self, level: int, msg: str, *args, **kwargs):
        """
        Log a message at specified level with class and function name context.

        Args:
            level: Logging level
            msg: Message format string
            args: Format string arguments
            kwargs: Additional logging arguments
        """
        frame = inspect.currentframe()
        if frame is not None:
            frame = frame.f_back  # in debug/info/etc.
        if frame is not None:
            frame = frame.f_back  # in real caller
        if frame is None:
            func_name = "unknown"
        else:
            func_name = frame.f_code.co_name

        if "logger" not in dir(self) or self.logger is None:
            raise RuntimeError(
                "Incorrect initialization of the class inherits Debuggable. You may need to call super().__init__() in the __init__ method of the class."
            )

        self.logger.log(
            level,
            "[%s.%s] " + msg,
            self.__class__.__name__,
            func_name,
            *args,
            **kwargs,
        )

    def debug(self, msg: str, *args, **kwargs):
        """Log a debug level message."""
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log an info level message."""
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log a warning level message."""
        self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log an error level message."""
        self.log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log a critical level message."""
        self.log(logging.CRITICAL, msg, *args, **kwargs)
