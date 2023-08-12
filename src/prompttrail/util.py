import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompttrail.agent.core import FlowState
    from prompttrail.agent.hook.core import Hook

MAX_TEMPLATE_LOOP = int(os.environ.get("MAX_TEMPLATE_LOOP", 10))
END_TEMPLATE_ID = "END"


def logger_multiline(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    for line in message.splitlines():
        logger.log(level, line)


def hook_logger(
    hook: "Hook",
    flow_state: "FlowState",
    message: str,
    level: int = logging.DEBUG,
):
    if flow_state.current_template is not None:
        # To avoid circular import
        from prompttrail.agent.template import Template

        if isinstance(flow_state.current_template, Template):
            template_id = flow_state.current_template.template_id
        else:
            template_id = flow_state.current_template

        logger = logging.getLogger(hook.__class__.__name__ + "@" + str(template_id))
    else:
        logger = logging.getLogger(hook.__class__.__name__)
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
