import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.prompttrail.flow.core import FlowState
    from src.prompttrail.flow.hooks.core import FeatherChainHook

MAX_TEMPLATE_LOOP = int(os.environ.get("MAX_TEMPLATE_LOOP", 10))


def logger_multiline(logger: logging.Logger, message: str, level: int = logging.DEBUG):
    for line in message.splitlines():
        logger.log(level, line)


def hook_logger(
    hook: "FeatherChainHook",
    flow_state: "FlowState",
    message: str,
    level: int = logging.DEBUG,
):
    if flow_state.current_template is not None:
        logger = logging.getLogger(
            hook.__class__.__name__ + "@" + str(flow_state.current_template.template_id)
        )
    else:
        logger = logging.getLogger(hook.__class__.__name__)
    logger_multiline(logger, message, level)
