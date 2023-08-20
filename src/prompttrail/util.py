import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompttrail.agent.core import State
    from prompttrail.agent.hook.core import Hook


def logger_multiline(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    for line in message.splitlines():
        logger.log(level, line)


def hook_logger(
    hook: "Hook",
    state: "State",
    message: str,
    level: int = logging.DEBUG,
):
    template_id = state.get_current_template_id()
    logger = logging.getLogger(hook.__class__.__name__ + "@" + str(template_id))
    logger_multiline(logger, message, level)


def is_in_test_env() -> bool:
    # We use pytest as the test runner.
    # https://stackoverflow.com/questions/25188119/test-if-code-is-executed-from-within-a-py-test-session
    if "pytest" in sys.modules:
        return True
    if os.environ.get("CI"):
        return True
    if os.environ.get("DEBUG"):
        return True
    return False
