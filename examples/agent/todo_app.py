# import os
# from typing import Any, Dict

# from prompttrail.agent.runners import CommandLineRunner
# from prompttrail.agent.templates import *
# from prompttrail.agent.tools._base import Tool, ToolArgument, ToolResult
# from prompttrail.agent.tools.builtin import *
# from prompttrail.agent.user_interface import CLIInterface
# from prompttrail.core import Session
# from prompttrail.models.anthropic import AnthropicConfig, AnthropicModel


# # Custom tool for todo operations
# class AddTodo(Tool):
#     name: str = "add_todo"
#     description: str = "Add todo items to a todo list"
#     arguments: list[ToolArgument[Any]] = [
#         ToolArgument(
#             name="done",
#             description="If passed, the task is created as a done task",
#             value_type=bool,
#             required=False,
#         ),
#         ToolArgument(
#             name="content",
#             description="Task content to add",
#             value_type=str,
#             required=True,
#         ),
#         ToolArgument(
#             name="tags",
#             description="A string of tags separated by spaces like #work #urgent",
#             value_type=str,
#             required=False,
#         ),
#         ToolArgument(
#             name="due",
#             description="Due date for the task, in the format YYYY-MM-DD",
#             value_type=str,
#             required=False,
#         ),
#     ]

#     def __init__(self, todo_file: str = "todo.txt"):
#         self.todo_file = todo_file

#     def _execute(self, args: Dict[str, Any]) -> ToolResult:


# def main():
#     # Initialize model
#     model = AnthropicModel(
#         configuration=AnthropicConfig(
#             api_key=os.environ["ANTHROPIC_API_KEY"],
#             model_name="claude-3-5-sonnet-latest",
#             max_tokens=4096,
#             tools=[TodoTool()],
#         )
#     )

#     # Create templates
#     templates = LinearTemplate(
#         [
#             SystemTemplate(
#                 content="""You are a todo list manager that helps users manage their tasks using a todo.txt format.

# Available commands:
# - add <task>: Add a new todo item
# - search <keyword>: Search todos containing keyword
# - edit <line_number> <new_task>: Edit a specific todo item
# - list: Show all todos
# - exit: Exit the application

# For each user input:
# 1. If the input is "exit", respond with "Goodbye!" and stop.
# 2. Otherwise:
#    - Parse the input to determine the appropriate todo command
#    - Use the todo tool to execute the command
#    - After using the tool, always respond with a brief message confirming what was done

# Example interactions:
# User: add buy milk
# Assistant: Using todo tool to add task...
# [Uses todo tool with command="add", task="buy milk"]
# Task added successfully.

# User: list
# Assistant: Here are your current tasks:
# [Uses todo tool with command="list"]
# [Shows task list]

# User: exit
# Assistant: Goodbye!

# Be concise and direct. Do not ask follow-up questions."""
#             ),
#             LoopTemplate(
#                 [
#                     UserTemplate(description="Enter command: "),
#                     ToolingTemplate(tools=[TodoTool()]),
#                 ],
#                 exit_condition=lambda session: session.messages[-1]
#                 .content.strip()
#                 .lower()
#                 == "exit",
#             ),
#         ]
#     )

#     # Run the application
#     runner = CommandLineRunner(
#         model=model, template=templates, user_interface=CLIInterface()
#     )
#     runner.run(Session())


# if __name__ == "__main__":
#     main()
