import inspect
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional

import tiktoken

if TYPE_CHECKING:
    pass


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
    # Remove %(name)s and add %(filename)s:%(lineno)d in the desired position:
    format="%(levelname)-8s %(filename)s:%(lineno)d %(message)s",
)


class Debuggable:
    def __init__(self) -> None:
        # Use __name__ so the logger name matches your module name (e.g. "utils")
        self.logger: Optional[logging.Logger] = logging.getLogger(__name__)
        self.debug_mode = True

    def setup_logger_for_pydantic(self):
        """Setup logger for the class. This should be called after pydantic initialization. You should't remove this method unless you have a good reason."""
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = logging.getLogger(__name__)
        if not hasattr(self, "enable_logging"):
            self.debug_mode = True

    def disable_log(self):
        self.debug_mode = False

    def enable_log(self):
        self.debug_mode = True

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
        if self.logger is None:
            # raise error for incorect initizalization
            raise RuntimeError(
                "Incorrect initialization of the class inherits Debuggable"
            )
        else:
            self.logger.log(
                level,
                "[%s.%s] " + msg,
                self.__class__.__name__,
                func_name,
                *args,
                **kwargs,
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
