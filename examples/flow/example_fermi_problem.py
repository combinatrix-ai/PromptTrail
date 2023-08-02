# In this example, we create a bot that can solve Fermi Problem using PromptTrail.
# Fermi Problem is a kind of problem that requires you to estimate the quantitative answer to a question using common sense and logic.
# For example, "How many cats in Japan?" is a Fermi Problem, as shown below. (This example is actually from OpenAI API)
# We use OpenAI API to generate text, and use Python to calculate the answer.
# Let's see how it works!

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
    is_same_template,
)
from src.prompttrail.flow.hooks import (
    AskUserHook,
    GenerateChatHook,
    EvaluatePythonCodeHook,
    IfJumpHook,
    BooleanHook,
    ExtractMarkdownCodeBlockHook,
)
from prompttrail.flow.runner import CommandLineRunner

logging.basicConfig(level=logging.DEBUG)

flow_template = LinearTemplate(
    [
        MessageTemplate(
            # First, let's give an instruction to the API
            # In OpenAI API, system is a special role that gives instruction to the API
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
                # Then, we can start the conversation with user
                # First, we ask user for their question
                first := MessageTemplate(
                    # Note: we can name the template using walrus operator (though not recommended), this is used later.
                    role="user",
                    before_transform=[
                        # You can modify the content of the message using TransformHook
                        # In this case, this hook ask user to input new question
                        # This input is passed to key: "prompt", which can be accessed in the template using {{ prompt }}
                        # Also, you can access "prompt" in Hook using flow_state.data["prompt"]
                        # TODO: Of course, you dont have to use hook in future, we will provide AskUserTemplate to do this
                        AskUserHook(
                            key="prompt",
                            description="Input:",
                            default="How many elephants in Japan?",
                        )
                    ],
                    content="{{ prompt }}",
                ),
                MessageTemplate(
                    role="assistant",
                    # Now we have the user's question, we can ask the API to generate the answer
                    # GenerateChatHook is a hook that generate text using model, with passing the previous messages to the model
                    # TODO: We will provide TextGenerationTemplate to do this in the future
                    before_transform=[GenerateChatHook(key="generated_text")],
                    content="{{ generated_text }}",
                    after_transform=[
                        # This is where things get interesting!
                        # You can extract code block from markdown using ExtractMarkdownCodeBlockHook
                        # As we give an example, the API may include something like this in their response, which is stored in this message
                        #  ```python
                        #     5300000 * 0.49 * 2.1
                        #  ```
                        # This hook will extract a Python code block and store it in key: "python_segment"
                        ExtractMarkdownCodeBlockHook(
                            key="python_segment", lang="python"
                        ),
                        # Then, you can evaluate the code block using EvaluatePythonCodeHook
                        # Yeah, this is a bit dangerous, we will provide safer way to do this in the future.
                        # Note that the key "python_segment" from ExtractMarkdownCodeBlockHook is used here
                        # The result of the evaluation is stored in key: "answer"
                        EvaluatePythonCodeHook(key="answer", code="python_segment"),
                    ],
                    after_control=[
                        # Maybe you want to jump to another template based on the answer
                        # You can do this using IfJumpHook
                        # In this case, if no python code block is found, jump to first template and retry with another question given by user
                        IfJumpHook(
                            condition=lambda flow_state: "answer" in flow_state.data,
                            true_template="gather_feedback",
                            false_template=first.template_id,
                        )
                    ],
                ),
                MessageTemplate(
                    # You can also give assistant message without using model, as if the assistant said it
                    # In this case, we want to ask user if the answer is satisfied or not
                    # Analysing the user response is always hard, so we let the API to decide
                    # First, we must ask user for their feedback
                    # Let's ask user for question!
                    template_id="gather_feedback",
                    role="assistant",
                    content="The answer is {{ answer }}. Satisfied?",
                ),
                MessageTemplate(
                    # Here is where we ask user for their feedback
                    # You can call AskUserHook in previous template, but we may want to keep the message separated for clarity
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
                    # Based on the feedback, we can decide to retry or end the conversation
                    # Ask the API to analyze the user's sentiment
                    role="assistant",
                    content="The user has stated their feedback. If you think the user is satisified, you must answer `END`. Otherwise, you must answer `RETRY`.",
                ),
                check_end := MessageTemplate(
                    role="assistant",
                    # API will return END or RETRY (mostly!)
                    # Then, we can decide to end the conversation or retry, see exit_condition below
                    before_transform=[GenerateChatHook(key="generated_text")],
                    content="{{ generated_text }}",
                ),
            ],
            exit_condition=BooleanHook(
                condition=lambda flow_state:
                # Exit condition: if the last message given by API is END, then exit, else continue (in this case, go to top of loop)
                is_same_template(
                    flow_state.get_current_template(), check_end.template_id
                )
                and "END" in flow_state.get_last_message().content
            ),
        ),
    ],
)

# This runner runs model in cli.
# We will provide other runner, which will enable you to input/output via HTTP, etc...
runner = CommandLineRunner(
    model=OpenAIChatCompletionModel(
        configuration=OpenAIModelConfiguration(api_key=os.environ["OPENAI_API_KEY"])
    ),
    parameters=OpenAIModelParameters(model_name="gpt-4"),
    templates=[flow_template],
)

runner.run()
