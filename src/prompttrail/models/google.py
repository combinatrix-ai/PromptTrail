from logging import getLogger
from typing import List, Optional

import google.generativeai as genai  # type: ignore
from pydantic import BaseModel, ConfigDict

from prompttrail.core import Config, Message, Model, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError, ProviderResponseError

logger = getLogger(__name__)


class GoogleChatExample(BaseModel):
    """Example for few-shot learning in Google API."""

    prompt: str
    """Example prompt."""
    response: str
    """Example response."""

    model_config = ConfigDict(protected_namespaces=())


class GoogleConfig(Config):
    """Integration configuration class for Google API.

    Manages authentication credentials and model parameters in a centralized way.
    """

    # Authentication
    api_key: str
    """API key for Google API."""

    # Model parameters (inherited and overridden)
    model_name: str = "models/gemini-1.5-flash"
    """Model name. Use list_models() to see available models."""
    temperature: Optional[float] = 1.0
    """Sampling temperature."""
    max_tokens: Optional[int] = 1024
    """Maximum output tokens."""

    # Google-specific parameters
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

    def _validate_model_settings(self) -> None:
        """Google-specific configuration validation"""
        super()._validate_model_settings()
        if not self.api_key:
            raise ValueError("Google API key is required")
        if self.temperature is not None and (
            self.temperature < 0 or self.temperature > 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1):
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")


class GoogleModel(Model):
    """Google API model implementation."""

    configuration: GoogleConfig
    model_config = ConfigDict(protected_namespaces=())

    def _authenticate(self) -> None:
        """Configure API authentication."""
        genai.configure(api_key=self.configuration.api_key)

    def validate_session(self, session: Session, is_async: bool = False) -> None:
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

    def _send(self, session: Session) -> Message:
        """Send request to Google API and return the response."""
        self._authenticate()

        model = genai.GenerativeModel(self.configuration.model_name)
        chat = model.start_chat()

        if self.configuration.context:
            chat.send_message(self.configuration.context)

        if self.configuration.examples:
            for example in self.configuration.examples:
                chat.send_message(example.prompt)
                chat.send_message(example.response)

        for message in session.messages[:-1]:
            chat.send_message(message.content)

        response = chat.send_message(
            session.messages[-1].content,
            generation_config=genai.types.GenerationConfig(
                temperature=self.configuration.temperature,
                candidate_count=self.configuration.candidate_count,
                top_p=self.configuration.top_p,
                top_k=self.configuration.top_k,
                max_output_tokens=self.configuration.max_tokens,
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
