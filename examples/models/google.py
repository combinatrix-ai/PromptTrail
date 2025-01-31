import os

from prompttrail.core import Message, Session
from prompttrail.models.google import GoogleConfig, GoogleModel

api_key = os.environ.get("GOOGLE_CLOUD_API_KEY", "")

session = Session(
    messages=[
        Message(content="What is 17 times 31? Just provide the number.", role="user"),
    ]
)

config = GoogleConfig(api_key=api_key, model_name="models/gemini-1.5-flash")
model = GoogleModel(configuration=config)
message = model.send(session=session)

print(message)
