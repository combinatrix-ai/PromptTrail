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
    """Parameter for AnthoropicClaudeModel.

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
        return Message(content=content, sender=response.role)

    def validate_session(self, session: Session, is_async: bool) -> None:
        if len(session.messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should be a Session object and have at least one message."
            )
        # Anthropic API allow zero or one system message at the beginning
        if (
            session.messages[0].sender == "system"
            and len(
                [message for message in session.messages if message.sender == "system"]
            )
            > 1
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should have at most one system message at the beginning. (Anthropic API restriction)"
            )
        # Anthropic API allow only "user", "assistant", and "system" as sender
        if any(
            [
                message.sender not in ["user", "assistant", "system"]
                for message in session.messages
            ]
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have sender of 'user', 'assistant', or 'system'. (Anthropic API restriction)"
            )
        if any([not isinstance(message.content, str) for message in session.messages]):  # type: ignore
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )
        # TODO: OpenAI allow empty string, but Google Cloud does not. In principle, we should not allow empty string. Should we impose this restriction on OpenAI as well?
        if any([message.content == "" for message in session.messages]):  # type: ignore
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should not be empty string. (Google Cloud API restriction)"
            )
        if any([message.sender is None for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have sender."
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
                [{"role": message.sender, "content": message.content} for message in messages[1:]],  # type: ignore
                messages[0].content,
            )
        else:
            return ([{"role": message.sender, "content": message.content} for message in messages], None)  # type: ignore

    def list_models(self) -> List[str]:
        # see https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message_create_params.py#L115C1-L122C15
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1'",
            "claude-2.0",
            "claude-instant-1.2",
        ]
