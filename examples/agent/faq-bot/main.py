# from prompttrail.agent.templates.core import LoopTemplate, MessageTemplate


# main_flow = LinearTemplate(
#     templates = [
#         start_template := MessageTemplate(
#             content="""
# You're an AI assistant that help your users to find the answer to their questions.
# You're given a question from the user, and some information about the question.
# You can safely ignore the information if you don't need it.
#             """,
#             role="system"
#         ),
#         MessageTemplate(
#             content="What is your question?",
#             role="assistant"
#         ),
#         LoopTemplate(
#             templates=[
#                 UserInputTemplate(key="question", after_transform = )),
#                 MessageTemplate(
#                     before_transform = VectorSearchHook(lambda state: state.data.get("question")),
#                     content="""
# Additional Information:

# """,
#                     role="assistant"
#                 ),
#                 GenarateTemplate(),
#         )
#                 MessageTemplate(
#     ]

# coroutine_flow = LinearTemplate(
#     templates = [
#         SessionDumpTemplate(),


# )
