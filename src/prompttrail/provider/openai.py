import logging
from typing import Dict, Generator, List, Literal, Optional, Sequence, Tuple

import openai

from prompttrail.agent.hook.core import JumpHook, TransformHook
from prompttrail.agent.template.core import (
    GenerateTemplate,
    MessageTemplate,
    TemplateId,
    TemplateLike,
)
from prompttrail.core import (
    Configuration,
    Message,
    Model,
    Parameters,
    Session,
    TextMessage,
)
from prompttrail.error import ParameterValidationError
from prompttrail.mock import MockModel, MockProvider

logger = logging.getLogger(__name__)


class OpenAIModelConfiguration(Configuration):
    api_key: str
    organization_id: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None


class OpenAIModelParameters(Parameters):
    model_name: str
    temperature: Optional[float] = 0
    max_tokens: int = 1024

    class Config:
        protected_namespaces = ()


class OpenAIChatCompletionModel(Model):
    configuration: OpenAIModelConfiguration

    def _authenticate(self) -> None:
        openai.api_key = self.configuration.api_key  # type: ignore
        openai.organization = self.configuration.organization_id  # type: ignore
        if self.configuration.api_base is not None:
            openai.api_base = self.configuration.api_base  # type: ignore
        if self.configuration.api_version is not None:
            openai.api_version = self.configuration.api_version  # type: ignore

    def before_send(
        self, parameters: Parameters, session: Optional[Session], is_async: bool
    ) -> Tuple[Optional[Configuration], Optional[Parameters], Optional[Session]]:
        self._authenticate()
        return (None, None, None)

    def _send(self, parameters: Parameters, session: Session) -> Message:
        if not isinstance(parameters, OpenAIModelParameters):
            raise ParameterValidationError(
                f"{OpenAIModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )
        # TODO: Add retry logic for http error and max_tokens_exceeded
        response = openai.ChatCompletion.create(  # type: ignore
            model=parameters.model_name,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
            messages=self._session_to_openai_messages(session),
        )
        message = response.choices[0]["message"]  # type: ignore #TODO: More robust error handling
        return TextMessage(content=message["content"], sender=message["role"])  # type: ignore #TODO: More robust error handling

    def _send_async(
        self, parameters: Parameters, session: Session
    ) -> Generator[TextMessage, None, None]:
        if not isinstance(parameters, OpenAIModelParameters):
            raise ParameterValidationError(
                f"{OpenAIModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )
        response: Generator[Dict[str, str], None, None] = openai.ChatCompletion.create(  # type: ignore
            model=parameters.model_name,
            temperature=parameters.temperature,
            max_tokens=parameters.max_tokens,
            messages=self._session_to_openai_messages(session),
            stream=True,
        )
        # response is a generator, and we want response[i]['choices'][0]['delta'].get('content', '')
        all_text: str = ""
        for message in response:
            all_text: str = all_text + message.choices[0]["delta"].get("content", "")  # type: ignore #TODO: More robust error handling
            yield TextMessage(content=all_text, sender=None)  # type: ignore

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
                message.sender not in ["system", "assistant", "user", "prompttrail"]
                # TODO: decide what to do with MetaTemplate (role=prompttrail)
                for message in session.messages
            ]
        ):
            raise ParameterValidationError(
                f"{self.__class__.__name__}: Sender should be one of 'system', 'assistant', 'user' for all message in a session."
            )

    @staticmethod
    def _session_to_openai_messages(session: Session) -> List[Dict[str, str]]:
        # TODO: decide what to do with MetaTemplate (role=prompttrail)
        messages = [
            message for message in session.messages if message.sender != "prompttrail"
        ]
        return [
            {
                "content": message.content,
                "role": message.sender,  # type: ignore
            }
            for message in messages
        ]

    def list_models(self) -> List[str]:
        self._authenticate()
        response = openai.Model.list()  # type: ignore
        return [model.id for model in response.data]  # type: ignore


class OpenAIChatCompletionModelMock(OpenAIChatCompletionModel, MockModel):
    def setup(self, mock_provider: MockProvider) -> None:
        self.mock_provider: MockProvider = mock_provider

    def _send(self, parameters: Parameters, session: Session) -> Message:
        if not isinstance(parameters, OpenAIModelParameters):
            raise ParameterValidationError(
                f"{OpenAIModelParameters.__name__} is expected, but {type(parameters).__name__} is given."
            )
        return self.mock_provider.call(session)


OpenAIrole = Literal["system", "assistant", "user"]


class OpenAIMessageTemplate(MessageTemplate):
    def __init__(
        self,
        content: str,
        role: OpenAIrole,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateLike] = None,
        before_transform: Sequence[TransformHook] = [],
        after_transform: Sequence[TransformHook] = [],
        before_control: Sequence[JumpHook] = [],
        after_control: Sequence[JumpHook] = [],
    ):
        super().__init__(
            content=content,
            template_id=template_id,
            role=role,
            next_template_default=next_template_default,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )


class OpenAIGenerateTemplate(GenerateTemplate):
    def __init__(
        self,
        role: OpenAIrole,
        template_id: Optional[TemplateId] = None,
        next_template_default: Optional[TemplateLike] = None,
        before_transform: Sequence[TransformHook] = [],
        after_transform: Sequence[TransformHook] = [],
        before_control: Sequence[JumpHook] = [],
        after_control: Sequence[JumpHook] = [],
    ):
        super().__init__(
            template_id=template_id,
            role=role,
            next_template_default=next_template_default,
            before_transform=before_transform,
            after_transform=after_transform,
            before_control=before_control,
            after_control=after_control,
        )