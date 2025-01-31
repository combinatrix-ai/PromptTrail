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
        Message(content="Hey how are you?", role="user"),
    ]
)
print("Calling GPT-3.5 with this conversation history:")
print(session)
print("Response from OpenAI API:")
message_generator = model.send_async(session=session, yield_type="all")
for message in message_generator:
    print(message)

session = Session(
    messages=[
        Message(content="Tell me about yourself.", role="user"),
    ]
)
print("\nOf course, you can show the results incrementally!")
print("Calling GPT-3.5 with this conversation history:")
print(session)
print("Response from OpenAI API:")
message_generator = model.send_async(session=session, yield_type="new")
for message in message_generator:
    print(message.content, end="", flush=True)
