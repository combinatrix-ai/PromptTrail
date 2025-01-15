import os

from prompttrail.core import Message, Session
from prompttrail.models.openai import OpenAIConfiguration, OpenAIModel, OpenAIParam

api_key = os.environ.get("OPENAI_API_KEY", "")

config = OpenAIConfiguration(api_key=api_key)
parameters = OpenAIParam(model_name="gpt-3.5-turbo", max_tokens=1000, temperature=0)

model = OpenAIModel(configuration=config)

session = Session(
    messages=[
        Message(content="Hey", role="user"),
    ]
)

message = model.send(parameters=parameters, session=session)

print(message)
