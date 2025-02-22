import os

END_TEMPLATE_ID = "END"
RESERVED_TEMPLATE_IDS = [END_TEMPLATE_ID]
MAX_TEMPLATE_LOOP = int(os.environ.get("MAX_TEMPLATE_LOOP", 10))
CONTROL_TEMPLATE_ROLE = "control"


class ReachedEndTemplateException(Exception):
    """Exception raised when EndTemplate is rendered."""

    def __init__(self, farewell_message: str | None = None) -> None:
        super().__init__()
        self.farewell_message = farewell_message


class BreakException(Exception):
    """Exception raised when BreakTemplate is rendered."""
