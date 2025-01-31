import os

from prompttrail.core import Message, Session
from prompttrail.models.openai import OpenAIConfig, OpenAIModel

api_key = os.environ.get("OPENAI_API_KEY", "")

config = OpenAIConfig(
    api_key=api_key, model_name="gpt-4o-mini", max_tokens=1000, temperature=0
)

model = OpenAIModel(configuration=config)

session = Session(
    messages=[
        Message(content="Hey", role="user"),
    ]
)

message = model.send(session=session)

print(message)
