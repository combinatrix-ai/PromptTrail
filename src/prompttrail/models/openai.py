import logging
import typing
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ConfigDict

from prompttrail.agent.tools import Tool, ToolResult
from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = logging.getLogger(__name__)


class OpenAIConfiguration(Configuration):
    api_key: str
    organization_id: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None


class OpenAIParam(Parameters):
    """Parameters for OpenAI models.

    Inherits common parameters from Parameters base class and adds OpenAI-specific parameters.
    """

    model_name: str
    temperature: Optional[float] = 1.0
    max_tokens: int = 1024

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class OpenAIModel(Model):
    configuration: OpenAIConfiguration  # type: ignore

    def _authenticate(self) -> None:
        openai.api_key = self.configuration.api_key  # type: ignore
        openai.organization = self.configuration.organization_id  # type: ignore
        if self.configuration.api_base is not None:
            openai.api_base = self.configuration.api_base  # type: ignore
        if self.configuration.api_version is not None:
            openai.api_version = self.configuration.api_version  # type: ignore

    def format_tool(self, tool: Tool) -> Dict[str, Any]:
        """Convert tool to OpenAI Function Calling format"""
        return tool.to_schema()

    def format_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """Format result for OpenAI API"""
        return {
            "role": "function",
            "name": result.metadata.get("function_name"),
            "content": str(result.content),
        }

    def validate_tools(self, tools: List[Tool]) -> None:
        """Validate tools according to OpenAI API requirements"""
        for tool in tools:
            # Validate tool name
            if not (tool.name.isalnum() or "_" in tool.name):
                raise ParameterValidationError(
                    f"Tool name must be alphanumeric or underscore: {tool.name}"
                )
            # Validate description
            if not tool.description:
                raise ParameterValidationError(
                    f"Tool description is required: {tool.name}"
                )
            # Validate arguments
            if not tool.arguments:
                raise ParameterValidationError(
                    f"Tool must have at least one argument: {tool.name}"
                )

    def before_send(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[Optional[Configuration], Optional[Parameters], Optional[Session]]:
        self._authenticate()
        return (None, None, None)

    def _send(self, parameters: Parameters, session: Session) -> Message:
        if not isinstance(parameters, OpenAIParam):
            raise ParameterValidationError(
                f"{OpenAIParam.__name__} is expected, but {type(parameters).__name__} is given."
            )

        messages = self._session_to_openai_messages(session)

        # Create parameters for OpenAI API
        create_params: Dict[str, Any] = {
            "model": parameters.model_name,
            "temperature": parameters.temperature,
            "max_tokens": parameters.max_tokens,
            "messages": messages,
        }

        if parameters.tools:
            create_params["tools"] = [
                {"type": "function", "function": self.format_tool(tool)}
                for tool in parameters.tools
            ]

        response = openai.chat.completions.create(**create_params)  # type: ignore

        message = response.choices[0].message
        content = message.content
        if content is None:
            content = ""

        result = Message(content=content, role=message.role)  # type: ignore

        # Process tool call results
        if message.tool_calls:
            tool_call = message.tool_calls[0]  # Currently handle only first call
            result.metadata["function_call"] = {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }

        return result

    def _send_async(
        self,
        parameters: Parameters,
        session: Session,
        yiled_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        if not isinstance(parameters, OpenAIParam):
            raise ParameterValidationError(
                f"{OpenAIParam.__name__} is expected, but {type(parameters).__name__} is given."
            )

        messages = self._session_to_openai_messages(session)

        # Create parameters for OpenAI API
        create_params: Dict[str, Any] = {
            "model": parameters.model_name,
            "temperature": parameters.temperature,
            "max_tokens": parameters.max_tokens,
            "messages": messages,
            "stream": True,
        }

        if parameters.tools:
            create_params["tools"] = [
                {"type": "function", "function": self.format_tool(tool)}
                for tool in parameters.tools
            ]

        response: openai.Stream = openai.chat.completions.create(**create_params)  # type: ignore

        all_text: str = ""
        role = None
        for message in response:  # type: ignore
            logger.debug(f"Received message: {message}")
            if role is None:
                role = message.choices[0].delta.role  # type: ignore
            new_text: str = message.choices[0].delta.content or ""  # type: ignore
            if yiled_type == "new":
                yield Message(content=new_text, role=role)  # type: ignore
            elif yiled_type == "all":
                all_text: str = all_text + new_text  # type: ignore
                yield Message(content=all_text, role=role)  # type: ignore
            else:
                raise ParameterValidationError(
                    f"{self.__class__.__name__}: yiled_type should be 'all' or 'new'."
                )

    def validate_session(self, session: Session, is_async: bool) -> None:
        """Validate session for OpenAI models.

        Extends the base validation with OpenAI-specific role validation.
        """
        super().validate_session(session, is_async)

        # OpenAI-specific validation for allowed roles
        allowed_roles = list(typing.get_args(OpenAIrole)) + [CONTROL_TEMPLATE_ROLE]
        if any([message.role not in allowed_roles for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: role should be one of {allowed_roles} in a session."
            )

    @staticmethod
    def _session_to_openai_messages(
        session: Session,
    ) -> List[ChatCompletionMessageParam]:
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        def convert_message(message: Message) -> ChatCompletionMessageParam:
            if "function_call" in message.metadata:
                return {
                    "content": message.content,
                    "role": cast(Literal["function"], message.role),
                    "name": message.metadata["function_call"]["name"],
                }
            elif message.role == "system":
                return {"content": message.content, "role": "system"}
            elif message.role == "user":
                return {"content": message.content, "role": "user"}
            elif message.role == "assistant":
                return {"content": message.content, "role": "assistant"}
            else:
                raise ValueError(f"Unsupported role: {message.role}")

        return [convert_message(message) for message in messages]

    def list_models(self) -> List[str]:
        self._authenticate()
        response = openai.models.list()
        return [model.id for model in response.data]  # type: ignore


OpenAIrole = Literal["system", "assistant", "user", "function"]
