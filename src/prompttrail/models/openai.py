import json
import logging
from pprint import pformat
from typing import Any, Dict, Generator, List, Literal, Optional

import openai
from openai import OpenAI
from pydantic import ConfigDict

from prompttrail.agent.tools import Tool
from prompttrail.core import Config, Message, Model, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE

logger = logging.getLogger(__name__)


class OpenAIConfig(Config):
    """Integration configuration class for OpenAI Chat API.

    Manages authentication credentials and model parameters in a centralized way.
    """

    # Authentication
    api_key: str
    """API key for OpenAI API."""
    organization_id: Optional[str] = None
    """Organization ID for OpenAI API."""
    api_base: Optional[str] = None
    """Base URL for OpenAI API."""
    api_version: Optional[str] = None
    """API version for OpenAI API."""

    # Model parameters (inherited from UnifiedModelConfig)
    model_name: str = "gpt-4o-mini"
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 100

    def _validate_model_settings(self) -> None:
        """OpenAI-specific configuration validation"""
        super()._validate_model_settings()
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    def _validate_tools(self) -> None:
        """OpenAI-specific tool validation"""
        super()._validate_tools()
        for tool in self.tools:  # type: ignore
            if not all(c.isalnum() or c in "-_" for c in tool.name):
                raise ValueError(
                    f"Tool name must be alphanumeric, hyphen, or underscore: {tool.name}"
                )


class OpenAIModel(Model):
    """Model class for OpenAI Chat API."""

    configuration: OpenAIConfig
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Optional[OpenAI] = None

    def _authenticate(self) -> None:
        """Configure OpenAI API authentication."""
        self.client = OpenAI(
            api_key=self.configuration.api_key,
            organization=self.configuration.organization_id,
            base_url=self.configuration.api_base,
        )

    def format_tool(self, tool: "Tool") -> Dict[str, Any]:
        """Convert tool to OpenAI format."""
        schema = tool.to_openai_schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema["parameters"],
        }

    def _session_to_openai_messages(self, session: Session) -> List[Dict[str, Any]]:
        """Convert session messages to OpenAI format."""
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
                if message.tool_use:
                    result.append(message.tool_use)
                else:
                    result.append({"content": message.content, "role": "assistant"})
            elif message.role == "tool_result":
                # unpack content
                content = json.loads(message.content)
                result.append(
                    {
                        "content": json.dumps(content["content"]),
                        # TODO: decide how to handle tool call id
                        "tool_call_id": content["tool_call_id"],
                        "role": "tool",
                    }
                )
            else:
                raise ValueError(f"Unsupported role: {message.role}")
        return result

    def _send(self, session: Session) -> Message:
        """Send messages and return the response."""
        self._authenticate()
        messages = self._session_to_openai_messages(session)

        # Create parameters for OpenAI API
        create_params: Dict[str, Any] = {
            "model": self.configuration.model_name,
            "messages": messages,
        }

        if self.configuration.temperature is not None:
            create_params["temperature"] = self.configuration.temperature
        if self.configuration.max_tokens is not None:
            create_params["max_tokens"] = self.configuration.max_tokens

        if session.available_tools or self.configuration.tools:
            if session.available_tools and self.configuration.tools:
                self.info(
                    "Both template tools and model tools are present. Using template tools."
                )
            if session.available_tools:
                create_params["tools"] = [
                    {"type": "function", "function": self.format_tool(tool)}
                    for tool in session.available_tools
                ]
            else:
                if self.configuration.tools:
                    create_params["tools"] = [
                        {"type": "function", "function": self.format_tool(tool)}
                        for tool in self.configuration.tools
                    ]
        response = self.client.chat.completions.create(**create_params)  # type: ignore
        self.debug("Response: %s", pformat(response))

        message = response.choices[0].message  # type: ignore
        content = message.content  # type: ignore
        if content is None:
            content = "Tool Call Request"
            # TODO: handle multiple tool calls
            assert len(message.tool_calls) == 1, "Only one tool call is supported"
            # Save whole message as tool use to send it later
            result = Message(
                content=content, role=message.role, tool_use=message.to_dict()
            )
        else:
            result = Message(content=content, role=message.role)

        return result

    def _send_async(
        self,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """Send messages asynchronously and return the response."""
        self._authenticate()
        messages = self._session_to_openai_messages(session)

        # Create parameters for OpenAI API
        create_params: Dict[str, Any] = {
            "model": self.configuration.model_name,
            "messages": messages,
            "stream": True,
        }

        if self.configuration.temperature is not None:
            create_params["temperature"] = self.configuration.temperature
        if self.configuration.max_tokens is not None:
            create_params["max_tokens"] = self.configuration.max_tokens

        if self.configuration.tools:
            create_params["tools"] = [
                {"type": "function", "function": self.format_tool(tool)}
                for tool in self.configuration.tools
            ]

        response: openai.Stream = self.client.chat.completions.create(**create_params)  # type: ignore
        self.debug("Response: %s", pformat(response))

        all_text: str = ""
        role = None
        for message in response:  # type: ignore
            if role is None:
                role = message.choices[0].delta.role  # type: ignore
            new_text: str = message.choices[0].delta.content or ""  # type: ignore
            if yield_type == "new":
                yield Message(content=new_text, role=role)  # type: ignore
            elif yield_type == "all":
                all_text: str = all_text + new_text  # type: ignore
                yield Message(content=all_text, role=role)  # type: ignore
            else:
                raise ValueError(
                    f"Invalid yield_type: {yield_type}. Must be either 'all' or 'new'."
                )

    def list_models(self) -> List[str]:
        """Return a list of available models."""
        self._authenticate()
        response = self.client.models.list()  # type: ignore
        return [model.id for model in response.data]  # type: ignore
