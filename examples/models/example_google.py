import os

from src.prompttrail.core import (
    Session,
    Message,
)

from src.prompttrail.providers.google_cloud import (
    # GoogleCloudChatExample
    GoogleCloudChatModel,
    GoogleCloudConfiguration,
    GoogleCloudChatParameters,
)

# show log level
import logging

logging.basicConfig(level=logging.DEBUG)

api_key = os.environ.get("GOOGLE_API_KEY", "")

session = Session(
    messages=[
        Message(content="Hey how are you?", sender="user"),
    ]
)


config = GoogleCloudConfiguration(api_key=api_key)
parameters = GoogleCloudChatParameters(model_name="models/chat-bison-001")
model = GoogleCloudChatModel(configuration=config)
print(model.list_models())
message = model.send(parameters=parameters, session=session)

from IPython import embed  # type: ignore

embed()
