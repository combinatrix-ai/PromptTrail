import json
from typing import Generator, List, Optional, Sequence, cast

from prompttrail.agent.hooks import TransformHook
from prompttrail.agent.templates import GenerateTemplate, MessageTemplate
from prompttrail.agent.tools import Tool, check_arguments
from prompttrail.core import Message, Session
from prompttrail.core.const import OPENAI_SYSTEM_ROLE
from prompttrail.models.openai import OpenAIModel, OpenAIParam, OpenAIrole


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
        temporary_parameters.functions = self.functions  # type: ignore

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
            if rendered_message.metadata["function_call"]["name"] not in self.functions:
                raise ValueError(
                    f"Function {rendered_message.metadata['function_call']['name']} is not defined in the template."
                )
            function_ = self.functions[
                rendered_message.metadata["function_call"]["name"]
            ]
            arguments = check_arguments(
                rendered_message.metadata["function_call"]["arguments"],
                function_.argument_types,
            )
            result = function_.call(arguments, session)
            # Send result
            function_message = Message(
                role="function",
                metadata={"function_call": {"name": function_.name}},
                content=json.dumps(result.show()),
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
