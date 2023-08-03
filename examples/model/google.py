import os

from prompttrail.core import (
    Session,
    Message,
)

from prompttrail.provider.google_cloud import (
    GoogleCloudChatModel,
    GoogleCloudConfiguration,
    GoogleCloudChatParameters,
)

api_key = os.environ.get("GOOGLE_API_KEY", "")

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
