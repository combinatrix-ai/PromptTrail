import os

END_TEMPLATE_ID = "END"
RESERVED_TEMPLATE_IDS = [END_TEMPLATE_ID]
MAX_TEMPLATE_LOOP = int(os.environ.get("MAX_TEMPLATE_LOOP", 10))
CONTROL_TEMPLATE_ROLE = "control"
OPENAI_SYSTEM_ROLE = "system"

ReachedEndTemplateException = type("ReachedEndTemplateException", (Exception,), {})
