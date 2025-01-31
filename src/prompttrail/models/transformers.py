import logging
from typing import Generator, List, Literal, Optional

from pydantic import ConfigDict
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

from prompttrail.core import Config, Message, Model, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = logging.getLogger(__name__)


class TransformersConfig(Config):
    """Integration configuration class for Transformers models.

    Manages authentication credentials and model parameters in a centralized way.
    """

    # Device settings
    device: Optional[str] = None
    """Device to run model on (e.g. 'cpu', 'cuda')."""

    # Model parameters (inherited and overridden)
    model_name: str
    """Name of the model to use."""
    temperature: Optional[float] = 1.0
    """Sampling temperature between 0 and 1."""
    max_tokens: Optional[int] = 1024
    """Maximum number of tokens to generate."""

    # Transformers-specific parameters
    top_p: Optional[float] = 1.0
    """Nucleus sampling probability."""
    top_k: Optional[int] = None
    """Top-k sampling."""
    repetition_penalty: Optional[float] = 1.0
    """Higher values penalize repeated tokens more strongly."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    def _validate_model_settings(self) -> None:
        """Transformers-specific configuration validation"""
        super()._validate_model_settings()
        if self.temperature is not None and (
            self.temperature < 0 or self.temperature > 1
        ):
            raise ValueError("temperature must be between 0 and 1")
        if self.top_p is not None and (self.top_p <= 0 or self.top_p > 1):
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if self.repetition_penalty is not None and self.repetition_penalty < 1:
            raise ValueError("repetition_penalty must be greater than or equal to 1")


class TransformersModel(Model):
    """Model class for running transformer models locally.

    Args:
        configuration: Model configuration.
        model: Pre-trained transformer model.
        tokenizer: Tokenizer for the model.
    """

    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        configuration: TransformersConfig,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
    ):
        super().__init__(configuration=configuration)
        self.model = model
        self.tokenizer = tokenizer

    def before_send(
        self, session: Optional[Session], is_async: bool
    ) -> Optional[Session]:
        """Perform pre-send processing."""
        return None

    def _validate_and_prepare(
        self,
    ) -> tuple["AutoModelForCausalLM", "AutoTokenizer"]:
        """Prepare the model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer must be initialized before sending messages"
            )

        assert self.model is not None  # for type checker
        assert self.tokenizer is not None  # for type checker

        return self.model, self.tokenizer

    def _prepare_inputs(
        self,
        session: Session,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
    ):
        """Prepare input for the model."""
        input_text = self._session_to_text(session)
        return tokenizer(input_text, return_tensors="pt").to(model.device)

    def _create_generate_kwargs(
        self,
        streamer: Optional["TextStreamer"] = None,
    ) -> dict:
        """Create parameters for text generation."""
        kwargs = {
            "max_new_tokens": self.configuration.max_tokens,
            "temperature": self.configuration.temperature,
            "top_p": self.configuration.top_p,
            "top_k": self.configuration.top_k,
            "repetition_penalty": self.configuration.repetition_penalty,
            "do_sample": True,
        }
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    def _send(self, session: Session) -> Message:
        """Generate text."""
        model, tokenizer = self._validate_and_prepare()
        inputs = self._prepare_inputs(session, model, tokenizer)
        generate_kwargs = self._create_generate_kwargs()

        outputs = model.generate(**inputs, **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return Message(content=generated_text, role="assistant")

    def _send_async(
        self,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """Generate text asynchronously."""
        model, tokenizer = self._validate_and_prepare()
        inputs = self._prepare_inputs(session, model, tokenizer)
        streamer = self._create_streamer(yield_type)
        generate_kwargs = self._create_generate_kwargs(streamer)

        model.generate(**inputs, **generate_kwargs)
        yield from self._streamer_messages

    def _create_streamer(self, yield_type: Literal["all", "new"]) -> "TextStreamer":
        """Create a custom streamer for generating text incrementally."""
        from transformers import TextStreamer

        self._streamer_messages: List[Message] = []
        self._all_text = ""

        outer_self = self

        class TransformersStreamer(TextStreamer):
            def __init__(self, tokenizer, *args, **kwargs):
                super().__init__(tokenizer, *args, **kwargs)
                self.yield_type = yield_type

            def on_finalized_text(self, text: str, stream_end: bool = False):
                if self.yield_type == "new":
                    outer_self._streamer_messages.append(
                        Message(content=text, role="assistant")
                    )
                elif self.yield_type == "all":
                    outer_self._all_text += text
                    outer_self._streamer_messages.append(
                        Message(content=outer_self._all_text, role="assistant")
                    )

        return TransformersStreamer(self.tokenizer)

    def validate_session(self, session: Session, is_async: bool = False) -> None:
        """Validate session for transformer models."""
        super().validate_session(session, is_async)

    @staticmethod
    def _session_to_text(session: Session) -> str:
        """Convert session messages to text input for the model."""
        messages = [
            message
            for message in session.messages
            if message.role != CONTROL_TEMPLATE_ROLE
        ]

        for message in messages:
            if message.role == "tool_result":
                raise ParameterValidationError(
                    "TransformersModel: Tool result messages are not supported"
                )

        return "\n".join(f"{message.role}: {message.content}" for message in messages)
