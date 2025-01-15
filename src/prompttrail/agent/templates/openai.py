import json
import logging
from typing import Any, Dict, Generator, List, Optional, Sequence, cast

from prompttrail.agent.hooks import TransformHook
from prompttrail.agent.templates import GenerateTemplate, MessageTemplate
from prompttrail.agent.tools import Tool
from prompttrail.core import Message, Session
from prompttrail.core.const import OPENAI_SYSTEM_ROLE
from prompttrail.models.openai import OpenAIModel, OpenAIParam, OpenAIrole

logger = logging.getLogger(__name__)


def check_tool_arguments(args_str: str, tool: Tool) -> Dict[str, Any]:
    """Validate and process tool arguments

    Args:
        args_str: JSON string of arguments from the API
        tool: Tool instance to validate against

    Returns:
        Dict[str, Any]: Processed arguments

    Raises:
        ValueError: If required arguments are missing or types don't match
        json.JSONDecodeError: If arguments string is not valid JSON
    """
    try:
        args_dict = json.loads(args_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in arguments: {e}")

    result = {}

    # Check required arguments
    for name, arg in tool.arguments.items():
        if arg.required and name not in args_dict:
            raise ValueError(f"Missing required argument: {name}")

        if name in args_dict:
            value = args_dict[name]
            if not arg.validate_value(value):
                raise ValueError(
                    f"Invalid type for argument {name}: expected {arg.value_type}, got {type(value)}"
                )
            result[name] = value

    # Warn about unexpected arguments
    for name in args_dict:
        if name not in tool.arguments:
            logger.warning(f"Unexpected argument: {name}")

    return result


class OpenAIGenerateTemplate(GenerateTemplate):
    """GenerateTemplate for OpenAI. `role` is narrowed down to OpenAIrole."""

    def __init__(
        self,
        role: OpenAIrole,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            role=role,
            before_transform=before_transform,
            after_transform=after_transform,
        )


class OpenAIGenerateWithFunctionCallingTemplate(GenerateTemplate):
    """Function calling template for OpenAI. This template handle multiple turns of function calling."""

    def __init__(
        self,
        role: OpenAIrole,
        functions: Sequence[Tool],
        template_id: Optional[str] = None,
        before_transform: Optional[List[TransformHook]] = None,
        after_transform: Optional[List[TransformHook]] = None,
    ):
        super().__init__(
            template_id=template_id,
            role=role,
            before_transform=before_transform if before_transform is not None else [],
            after_transform=after_transform if after_transform is not None else [],
        )
        self.functions = {func.name: func for func in functions}

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        # before_transform
        for before_transform_hook in self.before_transform:
            session = before_transform_hook.hook(session)
        # render
        runner = session.runner
        if runner is None:
            raise ValueError(
                "Runner must be given to use GenerateTemplate. Do you use Runner correctly? Runner must be passed via Session."
            )
        if not isinstance(runner.models, OpenAIModel):
            raise ValueError(
                "Function calling can only be used with OpenAIChatCompletionModel."
            )

        temporary_parameters = cast(OpenAIParam, runner.parameters.model_copy())
        temporary_parameters.tools = list(
            self.functions.values()
        )  # Update to use tools parameter

        # 1st message: pass functions and let the model use it
        rendered_message = runner.models.send(temporary_parameters, session)
        message = Message(
            content=rendered_message.content,
            role=self.role,
            metadata={"template_id": self.template_id, **rendered_message.metadata},
        )
        session.append(message)
        yield message

        if rendered_message.metadata.get("function_call"):
            # 2nd message: call function if the model asks for it
            function_call = rendered_message.metadata["function_call"]
            if function_call["name"] not in self.functions:
                raise ValueError(
                    f"Function {function_call['name']} is not defined in the template."
                )

            tool = self.functions[function_call["name"]]
            arguments = check_tool_arguments(function_call["arguments"], tool)
            result = tool.execute(**arguments)

            # Send result
            function_message = Message(
                role="function",
                metadata={"function_call": {"name": tool.name}},
                content=json.dumps(result.content),  # Use result.content directly
            )
            session.append(function_message)
            yield function_message

            second_response = runner.models.send(runner.parameters, session)
            message = Message(
                content=second_response.content,
                role=second_response.role,
                metadata={"template_id": self.template_id},
            )
            session.append(message)
            yield message

        # after_transform
        for after_transform_hook in self.after_transform:
            session = after_transform_hook.hook(session)
        return session


class OpenAISystemTemplate(MessageTemplate):
    """MessageTemplate for OpenAI. `role` is set to `system`."""

    def __init__(
        self,
        content: str,
        template_id: Optional[str] = None,
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=OPENAI_SYSTEM_ROLE,
        )


class OpenAIMessageTemplate(MessageTemplate):
    """MessageTemplate for OpenAI. `role` is narrowed down to OpenAIrole."""

    def __init__(
        self,
        content: str,
        role: OpenAIrole,
        template_id: Optional[str] = None,
        before_transform: List[TransformHook] = [],
        after_transform: List[TransformHook] = [],
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=role,
            before_transform=before_transform,
            after_transform=after_transform,
        )
