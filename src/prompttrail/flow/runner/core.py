import logging
from abc import abstractmethod
from typing import Dict, Optional, Sequence

from prompttrail.core import Model, Parameters
from prompttrail.flow import FlowState, StatefulSession
from prompttrail.flow.templates import Template, TemplateId, TemplateLike
from prompttrail.util import END_TEMPLATE_ID, MAX_TEMPLATE_LOOP

logger = logging.getLogger(__name__)


def get_id(template_like: TemplateLike) -> TemplateId:
    if isinstance(template_like, Template):
        return template_like.template_id
    return template_like


class Runner(object):
    @abstractmethod
    def _run(
        self,
        start_template: Optional[TemplateLike] = None,
        flow_state: Optional[FlowState] = None,
    ) -> FlowState:
        raise NotImplementedError("run method is not implemented")


class CommandLineRunner(Runner):
    def __init__(
        self,
        model: Model,
        parameters: Parameters,
        templates: Sequence["Template"],
        flow_state: Optional[FlowState] = None,
    ):
        # In future it must handle models, manage task runner, and control simultaneous flow. So template itself cannot be sufficient.
        self.model = model
        self.parameters = parameters
        self.templates = templates
        self.flow_state = flow_state
        self.template_dict: Dict[TemplateId, Template] = {}
        visited_templates: Sequence[Template] = []
        for template in templates:
            for next_template in template.walk(visited_templates):
                if next_template.template_id in self.template_dict:
                    raise ValueError(
                        f"Template id {next_template.template_id} is duplicated."
                    )
                self.template_dict[next_template.template_id] = next_template  # type: ignore

    def run(
        self,
        start_template: Optional[TemplateLike] = None,
        flow_state: Optional[FlowState] = None,
    ) -> FlowState:
        if flow_state is None:
            flow_state = FlowState(
                model=self.model,
                parameters=self.parameters,
                data={},
                session_history=StatefulSession(),
                jump=None,
            )
        if start_template is not None:
            if isinstance(start_template, str):
                start_template = self._search_template(start_template)
            elif not isinstance(start_template, Template):  # type: ignore
                raise TypeError("start_template must be Template or TemplateId.")

        template = start_template if start_template is not None else self.templates[0]

        last_template = template
        same_template_count = 0
        next_message_index_to_show = 0
        next_template = None
        while 1:
            # logger_multiline(
            #     logger, f"Current template: {template.template_id}", logging.DEBUG
            # )
            # logger_multiline(
            #     logger, f"flow state (before rendering):\n{flow_state}", logging.DEBUG
            # )
            flow_state = template.render(flow_state)
            # logger_multiline(
            #     logger, f"flow state (after rendering):\n{flow_state}", logging.DEBUG
            # )
            new_messages = flow_state.session_history.messages[
                next_message_index_to_show:
            ]
            for message in new_messages:
                print(message)
            next_message_index_to_show = len(flow_state.session_history.messages)

            # calculate next template
            if flow_state.jump is not None:
                logger.info(msg=f"Jump is set to {flow_state.jump}.")
                if get_id(flow_state.jump) == END_TEMPLATE_ID:
                    logger.info(f"Flow is finished.")
                    break
                if not isinstance(flow_state.jump, Template):
                    next_template = self._search_template(flow_state.jump)
                flow_state.jump = None
            else:
                if flow_state.current_template is not None:
                    current_template = flow_state.current_template
                    next_template_like = self._search_template(
                        current_template
                    ).next_template_default
                    next_template = (
                        self._search_template(next_template_like)
                        if isinstance(next_template_like, TemplateId)
                        else next_template
                    )
                    if next_template is not None:
                        next_template = self._search_template(next_template)
            if next_template is None:
                logger.info(f"No jump is set. Flow is finished.")
                break
            if last_template == next_template:
                same_template_count += 1
                if same_template_count > MAX_TEMPLATE_LOOP:
                    raise RuntimeError(
                        f"Same template is rendered consecutively more than {MAX_TEMPLATE_LOOP} times. This may be caused by infinite loop?"
                    )
            template = next_template
        return flow_state

    def _search_template(self, template_like: TemplateLike) -> "Template":
        if isinstance(template_like, Template):
            return template_like
        if template_like not in self.template_dict:
            raise ValueError(f"Template id {template_like} is not found.")
        return self.template_dict[template_like]
