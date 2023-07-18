from logging import getLogger
from pprint import pprint
from typing import List, Optional

import google.generativeai as palm  # type: ignore
from google.generativeai.types.discuss_types import (  # type: ignore
    ChatResponse,
    MessageDict,
)
from pydantic import BaseModel  # type: ignore

logger = getLogger(__name__)

from ..core import Configuration, Message, Model, Parameters, Session, TextMessage
from ..error import ParameterValidationError, ProviderResponseError


class GoogleCloudConfiguration(Configuration):
    api_key: str


class GoogleCloudChatExample(BaseModel):
    prompt: str
    response: str


class GoogleCloudChatParameters(Parameters):
    model_name: str = "models/chat-bison-001"
    temperature: Optional[float] = 0
    max_output_tokens: Optional[int] = 1000
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    context: Optional[str] = None
    examples: Optional[List[GoogleCloudChatExample]] = None


class GoogleCloudChatModel(Model):
    configuration: GoogleCloudConfiguration

    def _send(self, parameters: Parameters, session: Session) -> Message:
        if not isinstance(parameters, GoogleCloudChatParameters):
            raise ParameterValidationError(
                f"{GoogleCloudChatParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )

        response: ChatResponse = palm.chat(  # type: ignore
            model=parameters.model_name,
            context=parameters.context,
            examples=[
                (example.prompt, example.response) for example in parameters.examples
            ]
            if parameters.examples is not None and len(parameters.examples) > 0
            else None,
            messages=[MessageDict(content=message.content, author=message.sender) for message in session.messages],  # type: ignore #TODO: More robust error handling
            temperature=parameters.temperature,
            candidate_count=parameters.candidate_count,
            top_p=parameters.top_p,
            top_k=parameters.top_k,
            prompt=None,  # TODO: Figure out this is the best way to handle this
        )
        logger.debug(pprint(response))  # type: ignore
        if len(response.candidates) == 0:  # type: ignore
            if hasattr(response, "filters") and len(response.filters) > 0:  # type: ignore
                raise ProviderResponseError(f"Blocked: {response.filters}", response=response)  # type: ignore
            raise ProviderResponseError(f"No candidates returned.", response=response)  # type: ignore

        message = response.candidates[0]  # type: ignore #TODO: More robust error handling
        return TextMessage(content=message["content"], sender=message["author"])  # type: ignore #TODO: More robust error handling

    def validate_session(self, session: Session, is_async: bool) -> None:
        if any([not isinstance(message.content, str) for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )
        if any([message.sender is None for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have sender."
            )
        if any(
            [
                message.sender not in ["system", "assistant", "user"]
                for message in session.messages
            ]
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Sender should be one of 'system', 'assistant', 'user' for all message in a session."
            )

    def list_models(self) -> List[str]:
        response: List[palm.Model] = palm.list_models()  # type: ignore
        return [model.name for model in response]  # type: ignore
