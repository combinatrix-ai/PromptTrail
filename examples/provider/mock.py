import os

from prompttrail.core import Message, Session
from prompttrail.mock import OneTurnConversationMockProvider
from prompttrail.provider.openai import (
    OpenAIChatCompletionModelMock,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

api_key = os.environ.get("OPENAI_API_KEY", "")

config = OpenAIModelConfiguration(api_key=api_key)
parameters = OpenAIModelParameters(
    model_name="gpt-3.5-turbo", max_tokens=1000, temperature=0
)

model = OpenAIChatCompletionModelMock(configuration=config)

model.setup(
    mock_provider=OneTurnConversationMockProvider(
        conversation_table={
            "1+1": Message(content="1215973652716", sender="assistant"),
        },
        sender="assistant",
    )
)

session = Session(
    messages=[
        Message(content="1+1", sender="user"),
    ]
)

message = model.send(parameters=parameters, session=session)

print("message should be: 1215973652716, as defined in the mock!")
print(message)

session = Session(
    messages=[
        Message(content="1+2", sender="user"),
    ]
)

try:
    # This will raise an error because the mock provider is not defined for the message "1+2"
    message = model.send(parameters=parameters, session=session)
except ValueError as e:
    assert str(e) == "Unexpected message is passed to mock provider: 1+2"
    print("Error:", e)
    print("1+2 is not defined in the mock provider, so this is expected!")
else:
    raise ValueError("This should not happen")
