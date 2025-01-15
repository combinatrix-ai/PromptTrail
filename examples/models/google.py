import os

from prompttrail.core import Message, Session
from prompttrail.models.google_cloud import (
    GoogleCloudChatModel,
    GoogleCloudChatModelConfiguration,
    GoogleCloudChatModelParameters,
)

api_key = os.environ.get("GOOGLE_CLOUD_API_KEY", "")

session = Session(
    messages=[
        Message(content="What is 17 times 31? Just provide the number.", role="user"),
    ]
)


config = GoogleCloudChatModelConfiguration(api_key=api_key)
parameters = GoogleCloudChatModelParameters(model_name="models/gemini-1.5-flash")
model = GoogleCloudChatModel(configuration=config)
message = model.send(parameters=parameters, session=session)

print(message)
