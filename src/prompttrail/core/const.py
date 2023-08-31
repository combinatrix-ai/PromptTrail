import os
from typing import Optional

END_TEMPLATE_ID = "END"
RESERVED_TEMPLATE_IDS = [END_TEMPLATE_ID]
MAX_TEMPLATE_LOOP = int(os.environ.get("MAX_TEMPLATE_LOOP", 10))
CONTROL_TEMPLATE_ROLE = "control"
OPENAI_SYSTEM_ROLE = "system"


class ReachedEndTemplateException(Exception):
    """Exception raised when EndTemplate is rendered."""


class JumpException(Exception):
    """Exception raised when JumpTemplate is rendered."""

    def __init__(self, jump_to: str, message: Optional[str] = None):
        super().__init__(message)
        self.jump_to = jump_to


class BreakException(Exception):
    """Exception raised when BreakTemplate is rendered."""
