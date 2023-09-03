import json
from typing import Generator, List, Optional, Sequence

from prompttrail.agent import State, StatefulMessage
from prompttrail.agent.hooks import TransformHook
from prompttrail.agent.templates import GenerateTemplate, MessageTemplate
from prompttrail.agent.tools import Tool, check_arguments
from prompttrail.core import Message
from prompttrail.core.const import OPENAI_SYSTEM_ROLE
from prompttrail.models.openai import OpenAIChatCompletionModel, OpenAIrole


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

    def _render(self, state: "State") -> Generator[Message, None, State]:
        # before_transform
        for before_transform_hook in self.before_transform:
            state = before_transform_hook.hook(state)
        # render
        runner = state.runner
        if runner is None:
            raise ValueError(
                "Runner must be given to use GenerateTemplate. Do you use Runner correctly? Runner must be passed via State."
            )
        if not isinstance(runner.models, OpenAIChatCompletionModel):
            raise ValueError(
                "Function calling can only be used with OpenAIChatCompletionModel."
            )

        temporary_parameters = runner.parameters.model_copy()
        temporary_parameters.functions = self.functions

        # 1st message: pass functions and let the model use it
        rendered_message = runner.models.send(
            temporary_parameters, state.session_history
        )
        message = StatefulMessage(
            content=rendered_message.content,
            sender=self.role,
            template_id=self.template_id,
            data=rendered_message.data,
        )
        state.session_history.messages.append(message)  # type: ignore
        yield message

        if rendered_message.data.get("function_call"):
            # 2nd message: call function if the model asks for it
            if rendered_message.data["function_call"]["name"] not in self.functions:
                raise ValueError(
                    f"Function {rendered_message.data['function_call']['name']} is not defined in the template."
                )
            function_ = self.functions[rendered_message.data["function_call"]["name"]]
            arguments = check_arguments(
                rendered_message.data["function_call"]["arguments"],
                function_.argument_types,
            )
            result = function_.call(arguments, state)  # type: ignore
            # Send result
            function_message = Message(
                sender="function",
                data={"function_call": {"name": function_.name}},
                content=json.dumps(result.show()),
            )
            state.session_history.messages.append(function_message)  # type: ignore
            yield function_message
            second_response = runner.models.send(
                runner.parameters, state.session_history
            )
            message = StatefulMessage(
                content=second_response.content,
                sender=second_response.sender,
                template_id=self.template_id,
            )
            state.session_history.messages.append(message)  # type: ignore
            yield message

        # after_transform
        for after_transform_hook in self.after_transform:
            state = after_transform_hook.hook(state=state)
        return state


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
