from logging import getLogger
from typing import List, Optional

import google.generativeai as genai  # type: ignore
from pydantic import BaseModel, ConfigDict

from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError, ProviderResponseError

logger = getLogger(__name__)


class GoogleConfig(Configuration):
    """Configuration for Google API."""

    api_key: str
    """API key for Google API."""

    model_config = ConfigDict(protected_namespaces=())


class GoogleChatExample(BaseModel):
    """Example for few-shot learning in Google API."""

    prompt: str
    """Example prompt."""
    response: str
    """Example response."""

    model_config = ConfigDict(protected_namespaces=())


class GoogleParam(Parameters):
    """Parameters for Google models.

    For detailed parameter descriptions, see:
    https://cloud.google.com/ai-platform/training/docs/using-gpus#using_tpus
    """

    model_name: str = "models/gemini-1.5-flash"
    """Model name. Use list_models() to see available models."""
    temperature: Optional[float] = 1.0
    """Sampling temperature."""
    max_tokens: Optional[int] = 1024
    """Maximum output tokens."""
    top_p: Optional[float] = None
    """Nucleus sampling threshold."""
    top_k: Optional[int] = None
    """Top-k sampling threshold."""
    candidate_count: Optional[int] = None
    """Number of candidate responses."""
    context: Optional[str] = None
    """Optional context for the model."""
    examples: Optional[List[GoogleChatExample]] = None
    """Optional few-shot learning examples."""

    model_config = ConfigDict(protected_namespaces=())


class GoogleModel(Model):
    """Google API model implementation."""

    configuration: GoogleConfig
    model_config = ConfigDict(protected_namespaces=())

    def _authenticate(self) -> None:
        """Configure API authentication."""
        genai.configure(api_key=self.configuration.api_key)

    def validate_session(self, session: Session, is_async: bool) -> None:
        """Validate session for Google API requirements.

        Args:
            session: Session to validate
            is_async: Whether validation is for async operation

        Raises:
            ParameterValidationError: If session is invalid
        """
        super().validate_session(session, is_async)

        if any(not message.content for message in session.messages):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Empty messages not allowed"
            )

        messages = [
            msg for msg in session.messages if msg.role != CONTROL_TEMPLATE_ROLE
        ]
        if any(msg.role == "tool_result" for msg in messages):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Tool result messages not supported"
            )

    def _send(self, parameters: Parameters, session: Session) -> Message:
        """Send a request to the Google API.

        Args:
            parameters: Generation parameters
            session: Chat session

        Returns:
            Generated message

        Raises:
            ParameterValidationError: If parameters are invalid
            ProviderResponseError: If API request fails
        """
        self._authenticate()

        if not isinstance(parameters, GoogleParam):
            raise ParameterValidationError(
                f"Expected {GoogleParam.__name__}, got {type(parameters).__name__}"
            )

        model = genai.GenerativeModel(parameters.model_name)
        chat = model.start_chat()

        if parameters.context:
            chat.send_message(parameters.context)

        if parameters.examples:
            for example in parameters.examples:
                chat.send_message(example.prompt)
                chat.send_message(example.response)

        for message in session.messages[:-1]:
            chat.send_message(message.content)

        response = chat.send_message(
            session.messages[-1].content,
            generation_config=genai.types.GenerationConfig(
                temperature=parameters.temperature,
                candidate_count=parameters.candidate_count,
                top_p=parameters.top_p,
                top_k=parameters.top_k,
                max_output_tokens=parameters.max_tokens,
            ),
        )

        if response.prompt_feedback.block_reason:
            raise ProviderResponseError(
                f"Blocked: {response.prompt_feedback.block_reason}", response=response
            )

        if not response.text:
            raise ProviderResponseError("Empty response text", response=response)

        return Message(content=response.text, role="assistant")

    def list_models(self) -> List[str]:
        """Get list of available model names.

        Returns:
            List of model names
        """
        self._authenticate()
        return [model.name for model in genai.list_models()]
