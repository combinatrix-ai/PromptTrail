from logging import getLogger
from pprint import pformat
from typing import List, Optional

import google.generativeai as palm  # type: ignore
from google.generativeai.types.discuss_types import (  # type: ignore
    ChatResponse,
    MessageDict,
)
from pydantic import BaseModel, ConfigDict  # type: ignore

from prompttrail.core import Configuration, Message, Model, Parameters, Session
from prompttrail.core.errors import ParameterValidationError, ProviderResponseError

logger = getLogger(__name__)


class GoogleCloudChatModelConfiguration(Configuration):
    """Configuration for GoogleCloudChatModel."""

    api_key: str
    """API key for Google Cloud Chat API."""

    # required for autodoc
    model_config = ConfigDict(protected_namespaces=())


class GoogleCloudChatExample(BaseModel):
    """Example for Google Cloud Chat API."""

    prompt: str
    """ Prompt for the example. """
    response: str
    """ Response for the example. """

    # required for autodoc
    model_config = ConfigDict(protected_namespaces=())


class GoogleCloudChatModelParameters(Parameters):
    """Parameter for GoogleCloudChatModel.

    For detailed description of each parameter, see https://cloud.google.com/ai-platform/training/docs/using-gpus#using_tpus
    """

    model_name: str = "models/chat-bison-001"
    """ Name of the model to use. use GoogleCloudChatModel.list_models() to get the list of available models. """
    temperature: Optional[float] = 1.0
    """ Temperature for sampling. """
    max_tokens: Optional[int] = 1024
    """ Maximum number of tokens to generate. `max_output_tokens` on API. Name is changed to be consistent with other models. """
    top_p: Optional[float] = None
    """ Top-p value for sampling. """
    top_k: Optional[int] = None
    candidate_count: Optional[int] = None
    context: Optional[str] = None
    examples: Optional[List[GoogleCloudChatExample]] = None

    # required for autodoc
    model_config = ConfigDict(protected_namespaces=())


class GoogleCloudChatModel(Model):
    """Model for Google Cloud Chat API."""

    configuration: GoogleCloudChatModelConfiguration  # type: ignore

    # required for autodoc
    model_config = ConfigDict(protected_namespaces=())

    def _authenticate(self) -> None:
        palm.configure(  # type: ignore
            api_key=self.configuration.api_key,
        )

    def _send(self, parameters: Parameters, session: Session) -> Message:
        self._authenticate()
        if not isinstance(parameters, GoogleCloudChatModelParameters):
            raise ParameterValidationError(
                f"{GoogleCloudChatModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
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
        logger.debug(pformat(object=response))  # type: ignore
        if len(response.candidates) == 0:  # type: ignore
            if hasattr(response, "filters") and len(response.filters) > 0:  # type: ignore
                raise ProviderResponseError(f"Blocked: {response.filters}", response=response)  # type: ignore
            raise ProviderResponseError("No candidates returned.", response=response)  # type: ignore

        message = response.candidates[0]  # type: ignore #TODO: More robust error handling
        return Message(content=message["content"], sender=message["author"])  # type: ignore #TODO: More robust error handling

    def validate_session(self, session: Session, is_async: bool) -> None:
        if len(session.messages) == 0:
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Session should be a Session object and have at least one message."
            )
        if any([not isinstance(message.content, str) for message in session.messages]):  # type: ignore
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should be string."
            )
        # TODO: OpenAI allow empty string, but Google Cloud does not. In principle, we should not allow empty string. Should we impose this restriction on OpenAI as well?
        if any([message.content == "" for message in session.messages]):  # type: ignore
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should not be empty string."
            )
        if any([message.sender is None for message in session.messages]):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: All message in a session should have sender."
            )

    def list_models(self) -> List[str]:
        self._authenticate()
        response: List[palm.models] = palm.list_models()  # type: ignore
        return [model.name for model in response]  # type: ignore
