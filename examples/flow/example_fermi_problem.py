import logging
import os

from src.prompttrail.providers.openai import (
    OpenAIChatCompletionModel,
    OpenAIModelConfiguration,
    OpenAIModelParameters,
)

from src.prompttrail.flow.templates import (
    LinearTemplate,
    MessageTemplate,
    LoopTemplate,
)
from src.prompttrail.flow.hooks import (
    AskUserHook,
    GenerateChatHook,
    EvaluatePythonCodeHook,
    IfJumpHook,
    BooleanHook,
    ExtractMarkdownCodeBlockHook,
)
from prompttrail.flow.runner.core import CommanLineRunner

logging.basicConfig(level=logging.DEBUG)

flow_template = LinearTemplate(
    [
        MessageTemplate(
            role="system",
            content="""
            You're a helpful assistant to solve Fermi Problem.
            Answer the equation to estimate the answer to the user's query.

            For example, if user ask you `How many cats in Japan?`, your answer should look like below.
            Note that you must return python expression to for user to calculate it using Python.
            
            Thoughts:
            - I don't have no knowledge about the number of cats in Japan. However, I can estimate it using available knowledge.
            - As of 2019, about 67% of households in the United States owned either dogs or cats, with about 49% of households owning cats (source: American Pet Products Association).
            - Assuming the trend in Japan follows a similar pattern to the U.S., we can look at the total number of households in Japan, and estimate that about 49% of them own a cat.
            - The total number of households in Japan is about 53 million (source: Japan's Ministry of Internal Affairs and Communications, 2020).
            - We also need to consider the average number of cats per household.
            - In the U.S, the average number of cats per household is 2.1 (source: American Veterinary Medical Association).
            - If we apply this to Japan, we can estimate the total number of cats in Japan.
            - Now I can estimate the answer.

            Equation to be calculated:
            - Total Number of Cats in Japan = Total Number of Households in Japan * Rate of Households owning cats  * Average Number of Cats Per Household
            
            Calculation:
            ```python
            5300000 * 0.49 * 2.1
            ```
            """,
        ),
        LoopTemplate(
            [
                # Note: you can name the template using walrus operator (not recommended)
                first := MessageTemplate(
                    role="user",
                    before_transform=[
                        # You can modify the content of the message using TransformHook
                        # In this case, this hook ask user to input new question
                        # This input is passed to key: "prompt", which can be accessed in the template using {{ prompt }}
                        # Also, you can access "prompt" in Hook using flow_state.data["prompt"]
                        # TODO: Of course, you dont have to use hook, we will provide ExternalInputHook to do this
                        AskUserHook(
                            key="prompt",
                            description="Input:",
                            default="How many elephants in Japan?",
                        )
                    ],
                    content="""
                    {{ prompt }}
                    """,
                ),
                MessageTemplate(
                    role="assistant",
                    # GenerateChatHook is a hook that generate text using model
                    # We will provide TextGenerationHook to do this in the future
                    before_transform=[GenerateChatHook(key="generated_text")],
                    content="""
                    {{ generated_text }}
                    """,
                    after_transform=[
                        # This is where things get interesting
                        # You can extract code block from markdown using ExtractMarkdownCodeBlockHook
                        ExtractMarkdownCodeBlockHook(
                            key="python_segment", lang="python"
                        ),
                        # Then, you can evaluate the code block using EvaluatePythonCodeHook
                        # Note that the data is passed with key: "python_segment"
                        EvaluatePythonCodeHook(key="answer", code="python_segment"),
                    ],
                    after_control=[
                        # Maybe you want to jump to another template based on the answer
                        # You can do this using JumpHook
                        # In this case, if no python code block is found, jump to first template and retry with another question given by user
                        IfJumpHook(
                            condition=lambda flow_state: flow_state.data["answer"],
                            true_template="gather_feedback",
                            false_template=first.id,
                        )
                    ],
                ),
                MessageTemplate(
                    # You can also give assistant message without using model
                    template_id="gather_feedback",
                    role="assistant",
                    content="""
                    The answer is {{ answer }}. Satisfied?
                    """,
                ),
                MessageTemplate(
                    role="user",
                    before_transform=[
                        AskUserHook(
                            key="feedback", description="Input:", default="Let's retry."
                        )
                    ],
                    content="""
                    {{ feedback }}
                    """,
                ),
                MessageTemplate(
                    # You can also give assistant message without using model
                    role="assistant",
                    content="""
                    The user has stated their feedback. If you think the user is satisified, you must answer `END`. Otherwise, you must answer `RETRY`.
                    """,
                ),
                check_end := MessageTemplate(
                    role="assistant",
                    # Let API to decide the flow, see exit condition below
                    before_transform=[GenerateChatHook(key="generated_text")],
                    content="""
                    {{ generated_text }}
                    """,
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda flow_state: (
                    # Exit condition: if the last message given by API is END, then exit, else continue (in this case, go to top of loop)
                    flow_state.get_current_template().id == check_end.id
                    and "END" in flow_state.get_last_message().content
                )
            ),
        ),
    ],
)

# This runner runs model. We will provide other runner, which will enable you to interact user with API, etc.
runner = CommanLineRunner(
    model=OpenAIChatCompletionModel(
        configuration=OpenAIModelConfiguration(api_key=os.environ["OPENAI_API_KEY"])
    ),
    parameters=OpenAIModelParameters(model_name="gpt-4"),
    templates=[flow_template],
)

runner.run()

# You can also turn based on the model
# while message := runner.turn():
#     print(message.content)
