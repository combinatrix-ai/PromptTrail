from logging import getLogger
from pprint import pformat
from typing import List, Optional

import google.generativeai as palm  # type: ignore
import google.generativeai.types as gai_types  # type: ignore
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

    model_name: str = "models/gemini-1.5-flash"
    """ Name of the model to use. Use GoogleCloudChatModel.list_models() to get the list of available models. Default is set to a lightweight model for faster responses. """
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

        model = palm.GenerativeModel(parameters.model_name)
        chat = model.start_chat()

        # Set context if provided
        if parameters.context:
            chat.send_message(parameters.context)

        # Set examples if provided
        if parameters.examples is not None and len(parameters.examples) > 0:
            for example in parameters.examples:
                chat.send_message(example.prompt)
                chat.send_message(example.response)

        # Send all messages in the session
        for message in session.messages[:-1]:  # Send all but the last message
            chat.send_message(message.content)

        # Send the last message with generation config
        response = chat.send_message(
            session.messages[-1].content,
            generation_config=palm.types.GenerationConfig(
                temperature=parameters.temperature,
                candidate_count=parameters.candidate_count,
                top_p=parameters.top_p,
                top_k=parameters.top_k,
                max_output_tokens=parameters.max_tokens,
            ),
        )
        logger.debug(pformat(object=response))
        if response.prompt_feedback.block_reason:
            raise ProviderResponseError(
                f"Blocked: {response.prompt_feedback.block_reason}", response=response
            )
        if not response.text:
            raise ProviderResponseError("No response text returned.", response=response)

        return Message(content=response.text, sender="assistant")

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
        models = palm.list_models()
        return [model.name for model in models]
