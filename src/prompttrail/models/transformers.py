import logging
from typing import Generator, List, Literal, Optional, Tuple

from pydantic import ConfigDict
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
)

from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = logging.getLogger(__name__)


class TransformersConfig(Configuration):
    """Configuration for TransformersModel.

    Attributes:
        device: Device to run model on (e.g. 'cpu', 'cuda'). Defaults to None.
    """

    device: Optional[str] = None


class TransformersParam(Parameters):
    """Parameters for TransformersModel.

    Parameters for controlling text generation with transformer models.

    Attributes:
        temperature: Sampling temperature between 0 and 1. Higher values mean more random outputs.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling.
        repetition_penalty: Higher values penalize repeated tokens more strongly.
    """

    temperature: Optional[float] = 1.0
    max_tokens: int = 1024
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = 1.0

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


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
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[Optional[Configuration], Optional[Parameters], Optional[Session]]:
        return (None, None, None)

    def _validate_and_prepare(
        self, parameters: Parameters
    ) -> tuple[TransformersParam, "AutoModelForCausalLM", "AutoTokenizer"]:
        """Validate parameters and prepare model for generation."""
        if not isinstance(parameters, TransformersParam):
            raise ParameterValidationError(
                f"{TransformersParam.__name__} is expected, but {type(parameters).__name__} is given."
            )

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer must be initialized before sending messages"
            )

        assert self.model is not None  # for type checker
        assert self.tokenizer is not None  # for type checker

        return parameters, self.model, self.tokenizer

    def _prepare_inputs(
        self,
        session: Session,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer",
    ):
        """Prepare inputs for the model."""
        input_text = self._session_to_text(session)
        return tokenizer(input_text, return_tensors="pt").to(model.device)

    def _create_generate_kwargs(
        self,
        parameters: TransformersParam,
        streamer: Optional["TextStreamer"] = None,
    ) -> dict:
        """Create generation kwargs for the model."""
        kwargs = {
            "max_new_tokens": parameters.max_tokens,
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "top_k": parameters.top_k,
            "repetition_penalty": parameters.repetition_penalty,
            "do_sample": True,
        }
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    def _send(self, parameters: Parameters, session: Session) -> Message:
        """Generate text using the model."""
        params, model, tokenizer = self._validate_and_prepare(parameters)
        inputs = self._prepare_inputs(session, model, tokenizer)
        generate_kwargs = self._create_generate_kwargs(params)

        outputs = model.generate(**inputs, **generate_kwargs)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return Message(content=generated_text, role="assistant")

    def _send_async(
        self,
        parameters: Parameters,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        """Generate text asynchronously with streaming output."""
        params, model, tokenizer = self._validate_and_prepare(parameters)
        inputs = self._prepare_inputs(session, model, tokenizer)
        streamer = self._create_streamer(yield_type)
        generate_kwargs = self._create_generate_kwargs(params, streamer)

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

    def validate_session(self, session: Session, is_async: bool) -> None:
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
