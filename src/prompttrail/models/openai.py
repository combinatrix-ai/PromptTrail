import logging
from typing import Any, Dict, Generator, List, Literal, Optional

import openai
from pydantic import ConfigDict

from prompttrail.agent.tools import Tool
from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = logging.getLogger(__name__)


class OpenAIConfig(Configuration):
    """Configuration for OpenAI Chat API."""

    api_key: str
    """API key for OpenAI API."""
    organization_id: Optional[str] = None
    """Organization ID for OpenAI API."""
    api_base: Optional[str] = None
    """Base URL for OpenAI API."""
    api_version: Optional[str] = None
    """API version for OpenAI API."""


class OpenAIParam(Parameters):
    """Parameters for OpenAI Chat models.

    Inherits common parameters from Parameters base class and adds OpenAI-specific parameters.
    For detailed description of each parameter, see https://platform.openai.com/docs/api-reference/chat
    """

    model_name: str = "gpt-4o-mini"
    """ Name of the model to use. Use OpenAIModel.list_models() to get the list of available models. """
    temperature: float = 1.0
    """ Temperature for sampling. """
    max_tokens: int = 100
    """ Maximum number of tokens to generate. """


class OpenAIModel(Model):
    """Model for OpenAI Chat API."""

    configuration: OpenAIConfig  # type: ignore

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _authenticate(self) -> None:
        openai.api_key = self.configuration.api_key
        if self.configuration.organization_id:
            openai.organization = self.configuration.organization_id
        if self.configuration.api_base:
            openai.base_url = self.configuration.api_base
        if self.configuration.api_version:
            openai.api_version = self.configuration.api_version

    def format_tool(self, tool: Tool) -> Dict[str, Any]:
        """Convert tool to OpenAI format"""
        schema = tool.to_schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        }

    def validate_tools(self, tools: List[Tool]) -> None:
        """Validate tools according to OpenAI API requirements"""
        for tool in tools:
            # Validate tool name
            if not all(c.isalnum() or c in "-_" for c in tool.name):
                raise ParameterValidationError(
                    f"Tool name must be alphanumeric, hyphen, or underscore: {tool.name}"
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

    def _session_to_openai_messages(
        self, session: Session
    ) -> List[Dict[str, Any]]:  # type: ignore
        """Convert session messages to OpenAI format"""
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        result = []
        for i, message in enumerate(messages):
            if message.role == "system":
                result.append({"content": message.content, "role": "system"})
            elif message.role == "user":
                result.append({"content": message.content, "role": "user"})
            elif message.role == "assistant":
                if "function_call" in message.metadata:
                    result.append(
                        {
                            "content": message.content,
                            "role": "assistant",
                            "function_call": message.metadata["function_call"],
                        }
                    )
                else:
                    result.append({"content": message.content, "role": "assistant"})
            elif message.role == "tool_result":
                # Convert tool_result to function for OpenAI API
                # Look for the previous assistant message with function_call
                for j in range(i - 1, -1, -1):
                    prev_message = messages[j]
                    if (
                        prev_message.role == "assistant"
                        and "function_call" in prev_message.metadata
                    ):
                        name = prev_message.metadata["function_call"].get("name")
                        if name:
                            result.append(
                                {
                                    "content": str(message.content),
                                    "role": "function",
                                    "name": name,
                                }
                            )
                            break
                else:
                    # If no matching function_call found, treat as assistant message
                    result.append(
                        {
                            "content": str(message.content),
                            "role": "assistant",
                        }
                    )
            else:
                raise ValueError(f"Unsupported role: {message.role}")
        return result

    def validate_session(self, session: Session, is_async: bool) -> None:
        """Validate session for OpenAI Chat models.

        Extends the base validation with OpenAI-specific validation:
        - At most one system message at the beginning
        - No tool_result messages allowed
        """
        super().validate_session(session, is_async)

        # OpenAI-specific validation for checking at least one non-system message
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        non_system_messages = [
            message for message in messages if message.role != "system"
        ]
        if len(non_system_messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session must contain at least one non-system message."
            )

    def _send(self, parameters: Parameters, session: Session) -> Message:
        self._authenticate()
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
        logger.debug(f"Response: {response}")

        message = response.choices[0].message  # type: ignore
        content = message.content  # type: ignore
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
                    f"Invalid yield_type: {yiled_type}. Must be either 'all' or 'new'."
                )

    def list_models(self) -> List[str]:
        self._authenticate()
        response = openai.models.list()
        return [model.id for model in response.data]  # type: ignore
