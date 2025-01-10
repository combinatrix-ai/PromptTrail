import logging
from typing import Dict, Generator, List, Literal, Optional, Tuple, TYPE_CHECKING

from pydantic import ConfigDict
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompttrail.agent.tools import Tool
from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.const import CONTROL_TEMPLATE_ROLE
from prompttrail.core.errors import ParameterValidationError

logger = logging.getLogger(__name__)


class TransformersModelConfiguration(Configuration):
    device: Optional[str] = None


class TransformersModelParameters(Parameters):
    temperature: Optional[float] = 1.0
    max_tokens: int = 1024
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = 1.0

    # pydantic
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())


class TransformersModel(Model):
    configuration: Optional[TransformersModelConfiguration] = None  # type: ignore
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None

    # Pydantic config
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        configuration: TransformersModelConfiguration,
        model: "AutoModelForCausalLM",
        tokenizer: "AutoTokenizer"
    ):
        super().__init__()
        self.configuration = configuration
        self.model = model
        self.tokenizer = tokenizer

    def before_send(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[Optional[Configuration], Optional[Parameters], Optional[Session]]:
        return (None, None, None)

    def _send(self, parameters: Parameters, session: Session) -> Message:
        if not isinstance(parameters, TransformersModelParameters):
            raise ParameterValidationError(
                f"{TransformersModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )

        input_text = self._session_to_text(session)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": parameters.max_tokens,
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "top_k": parameters.top_k,
            "repetition_penalty": parameters.repetition_penalty,
            "do_sample": True,
        }

        outputs = self.model.generate(**inputs, **generate_kwargs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return Message(content=generated_text, sender="assistant")

    def _send_async(
        self,
        parameters: Parameters,
        session: Session,
        yield_type: Literal["all", "new"] = "new",
    ) -> Generator[Message, None, None]:
        if not isinstance(parameters, TransformersModelParameters):
            raise ParameterValidationError(
                f"{TransformersModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )

        input_text = self._session_to_text(session)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": parameters.max_tokens,
            "temperature": parameters.temperature,
            "top_p": parameters.top_p,
            "top_k": parameters.top_k,
            "repetition_penalty": parameters.repetition_penalty,
            "do_sample": True,
            "streamer": self._create_streamer(yield_type),
        }

        self.model.generate(**inputs, **generate_kwargs)
        yield from self._streamer_messages

    def _create_streamer(self, yield_type: Literal["all", "new"]) -> "TextStreamer":
        from transformers import TextStreamer

        self._streamer_messages = []
        self._all_text = ""

        class TransformersStreamer(TextStreamer):
            def __init__(self, tokenizer, *args, **kwargs):
                super().__init__(tokenizer, *args, **kwargs)
                self.yield_type = yield_type

            def on_finalized_text(self, text: str, stream_end: bool = False):
                if self.yield_type == "new":
                    self._streamer_messages.append(
                        Message(content=text, sender="assistant")
                    )
                elif self.yield_type == "all":
                    self._all_text += text
                    self._streamer_messages.append(
                        Message(content=self._all_text, sender="assistant")
                    )

        return TransformersStreamer(self.tokenizer)

    def validate_session(self, session: Session, is_async: bool) -> None:
        if len(session.messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should be a Session object and have at least one message."
            )
        if any([not isinstance(message.content, str) for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )
        if any([message.sender is None for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have sender."
            )

    @staticmethod
    def _session_to_text(session: Session) -> str:
        messages = [
            message
            for message in session.messages
            if message.sender != CONTROL_TEMPLATE_ROLE
        ]
        return "\n".join(
            f"{message.sender}: {message.content}" for message in messages
        )