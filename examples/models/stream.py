import os

from prompttrail.core import Message, Session
from prompttrail.models.openai import OpenAIConfiguration, OpenAIModel, OpenAIParam

api_key = os.environ.get("OPENAI_API_KEY", "")

config = OpenAIConfiguration(api_key=api_key)
parameters = OpenAIParam(model_name="gpt-3.5-turbo", max_tokens=1000, temperature=0)

model = OpenAIModel(configuration=config)

session = Session(
    messages=[
        Message(content="Hey how are you?", role="user"),
    ]
)
print("Calling GPT-3.5 with this conversation history:")
print(session)
print("Response from OpenAI API:")
message_generator = model.send_async(
    parameters=parameters, session=session, yield_type="all"
)
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
message_generator = model.send_async(
    parameters=parameters, session=session, yield_type="new"
)
for message in message_generator:
    print(message.content, end="", flush=True)
