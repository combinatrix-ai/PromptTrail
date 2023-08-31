import os

from prompttrail.core import Message, Session
from prompttrail.models.google_cloud import (
    GoogleCloudChatModel,
    GoogleCloudChatParameters,
    GoogleCloudConfiguration,
)

api_key = os.environ.get("GOOGLE_CLOUD_API_KEY", "")

session = Session(
    messages=[
        Message(content="Hey how are you?", sender="user"),
    ]
)


config = GoogleCloudConfiguration(api_key=api_key)
parameters = GoogleCloudChatParameters(model_name="models/chat-bison-001")
model = GoogleCloudChatModel(configuration=config)
message = model.send(parameters=parameters, session=session)

print(message)
