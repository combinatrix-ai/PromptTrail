from logging import getLogger
from pprint import pformat
from typing import Dict, List, Optional, Tuple

import anthropic
from pydantic import BaseModel, ConfigDict  # type: ignore

from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = getLogger(__name__)


class AnthropicClaudeModelConfiguration(Configuration):
    """Configuration for AnthoropicClaudeModel."""

    api_key: str
    """API key for Anthropic API."""


class AnthropicClaudeModelParameters(Parameters):
    """Parameters for Anthropic Claude models.

    Inherits common parameters from Parameters base class and adds Anthropic-specific parameters.
    For detailed description of each parameter, see https://github.com/anthropics/anthropic-sdk-python/blob/main/api.md
    """

    model_name: str = "claude-3-opus-20240229"
    """ Name of the model to use. use AnthoropicClaudeModel.list_models() to get the list of available models. """
    temperature: Optional[float] = 1.0
    """ Temperature for sampling. """
    max_tokens: int = 1024
    """ Maximum number of tokens to generate. """
    top_p: Optional[float] = None
    """ Top-p value for sampling. """
    top_k: Optional[int] = None
    """ Top-k value for sampling. """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class AnthropicClaudeModel(Model):
    """Model for Anthoropic Claude API."""

    configuration: AnthropicClaudeModelConfiguration  # type: ignore
    client: Optional[anthropic.Anthropic] = None
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def _authenticate(self) -> None:
        if self.client is None:
            self.client = anthropic.Anthropic(api_key=self.configuration.api_key)

    def _send(self, parameters: Parameters, session: Session) -> Message:
        self._authenticate()
        if not isinstance(parameters, AnthropicClaudeModelParameters):
            raise ParameterValidationError(
                f"{AnthropicClaudeModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )

        messages, system_prompt = self._session_to_anthropic_messages(session)

        additional_args = {}
        if parameters.temperature is not None:
            additional_args["temperature"] = parameters.temperature
        if parameters.top_p is not None:
            additional_args["top_p"] = parameters.top_p
        if parameters.top_k is not None:
            additional_args["top_k"] = parameters.top_k
        if system_prompt is not None:
            additional_args["system"] = system_prompt  # type: ignore
            # (TODO: Fix this mypy error: Incompatible types in assignment (expression has type "str", target has type "float")

        response: anthropic.Message = self.client.messages.create(  # type: ignore
            model=parameters.model_name,
            max_tokens=parameters.max_tokens,
            messages=messages,
            **additional_args,
        )
        logger.debug(pformat(object=response))  # type: ignore

        # TODO: should handle non-text response in future
        content = "".join([block.text for block in response.content])
        # TODO: Change to error that can be retriable
        if content == "":
            raise ValueError("Response is empty.")

        return Message(content=content, sender=response.role)

    def validate_session(self, session: Session, is_async: bool) -> None:
        """Validate session for Anthropic Claude models.

        Extends the base validation with Anthropic-specific validations:
        - At most one system message at the beginning
        - Only specific roles allowed
        - No empty messages
        """
        super().validate_session(session, is_async)

        # Anthropic-specific validation for checking at least one non-system message

        # Filter out control template messages
        messages = [
            message
            for message in session.messages
            if message.sender != CONTROL_TEMPLATE_ROLE
        ]

        non_system_messages = [
            message for message in messages if message.sender != "system"
        ]
        if len(non_system_messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session must contain at least one non-system message."
            )

        # Anthropic API allow zero or one system message at the beginning

        # Anthropic-specific validation for system message
        if (
            messages[0].sender == "system"
            and len([message for message in messages if message.sender == "system"]) > 1
        ) or (
            messages[0].sender != "system"
            and len([message for message in messages if message.sender == "system"]) > 0
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should have at most one system message at the beginning. (Anthropic API restriction)"
            )

        # Anthropic-specific validation for allowed roles
        if any(
            [
                message.sender not in ["user", "assistant", "system"]
                for message in messages
            ]
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have sender of 'user', 'assistant', or 'system'. (Anthropic API restriction)"
            )
        if any([not isinstance(message.content, str) for message in messages]):  # type: ignore
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )

        # Anthropic-specific validation for empty messages
        if any([message.content == "" for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should not be empty string. (Anthoropic API restriction)"
            )

    @staticmethod
    def _session_to_anthropic_messages(
        session: Session,
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        # TODO: decide what to do with MetaTemplate (role=prompttrail)
        # TODO: can content be empty?
        messages = [
            message
            for message in session.messages
            if message.sender != CONTROL_TEMPLATE_ROLE
        ]
        # if system message
        if messages[0].sender == "system":
            return (
                [
                    {"role": message.sender, "content": message.content}  # type: ignore
                    for message in messages[1:]
                ],  # type: ignore
                messages[0].content,
            )
        else:
            return (
                [
                    {"role": message.sender, "content": message.content}  # type: ignore
                    for message in messages
                ],
                None,
            )  # type: ignore

    def list_models(self) -> List[str]:
        self._authenticate()
        if self.client is None:
            raise RuntimeError("Failed to initialize Anthropic client")
        models = self.client.models.list()
        return [model.id for model in models.data]
