import os

from prompttrail.core import Message, Session
from prompttrail.core.mocks import OneTurnConversationMockProvider
from prompttrail.models.openai import OpenAIConfig, OpenAIModel

api_key = os.environ.get("OPENAI_API_KEY", "")

config = OpenAIConfig(
    api_key=api_key,
    model_name="gpt-4o-mini",
    max_tokens=1000,
    temperature=0,
    mock_provider=OneTurnConversationMockProvider(
        conversation_table={
            "1+1": Message(content="1215973652716", role="assistant"),
        },
    ),
)

model = OpenAIModel(configuration=config)

session = Session(
    messages=[
        Message(content="1+1", role="user"),
    ]
)

message = model.send(session=session)

print("message should be: 1215973652716, as defined in the mock!")
print(message)

session = Session(
    messages=[
        Message(content="1+2", role="user"),
    ]
)

try:
    # This will raise an error because the mock provider is not defined for the message "1+2"
    message = model.send(session=session)
except ValueError as e:
    assert str(e) == "Unexpected message is passed to mock provider: 1+2"
    print("Error:", e)
    print("1+2 is not defined in the mock provider, so this is expected!")
else:
    raise ValueError("This should not happen")
