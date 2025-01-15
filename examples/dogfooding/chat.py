import os

from examples.dogfooding.utils.load_all_important_files import load_all_important_files
from prompttrail.agent.runners import CommandLineRunner
from prompttrail.agent.templates import (
    AssistantTemplate,
    LinearTemplate,
    LoopTemplate,
    UserTemplate,
)
from prompttrail.agent.user_interaction import UserInteractionTextCLIProvider
from prompttrail.core import Session
from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel, AnthropicParam

templates = LinearTemplate(
    templates=[
        UserTemplate(
            content="""
You're given source code and test scripts and documents for a library, PromptTrail as below:
{{code}}
Discuss the question with user. User is the author of this library, who want to improve the design, implementation, and documentation of the library.
""",
        ),
        LoopTemplate(
            [
                UserTemplate(
                    description="Input:",
                ),
                AssistantTemplate(),
            ]
        ),
    ],
)

configuration = AnthropicConfig(api_key=os.environ["ANTHROPIC_API_KEY"])
parameter = AnthropicParam(
    model_name="claude-3-sonnet-20240229",
    temperature=1,
    max_tokens=4096,
)
model = AnthropicModel(configuration=configuration)

content = load_all_important_files()

runner = CommandLineRunner(
    model=model,
    parameters=parameter,
    template=templates,
    user_interaction_provider=UserInteractionTextCLIProvider(),
)

initial_session = Session(initial_metadata={"code": content})
runner.run(session=initial_session)
