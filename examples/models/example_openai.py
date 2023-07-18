import os

from src.prompttrail.core import (
    Session,
    Message,
)

from src.prompttrail.providers.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

api_key = os.environ.get("OPENAI_API_KEY", "")

config = OpenAIModelConfiguration(api_key=api_key)
parameters = OpenAIModelParameters(
    model_name="gpt-3.5-turbo", max_tokens=1000, temperature=0
)

model = OpenAIChatCompletionModel(configuration=config)

session = Session(
    messages=[
        Message(content="Hey", sender="user"),
    ]
)

message = model.send(parameters=parameters, session=session)

from IPython import embed  # type: ignore

embed()
