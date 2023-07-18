import logging
from abc import abstractmethod
from typing import Dict, Optional, Sequence

from src.prompttrail.core import Model, Parameters
from src.prompttrail.flow.core import FlowState, StatefulSession, TemplateId
from src.prompttrail.flow.templates import Template

logger = logging.getLogger(__name__)


class Runner(object):
    @abstractmethod
    def run(
        self,
        start_template: Optional["Template | TemplateId"] = None,
        flow_state: Optional[FlowState] = None,
    ) -> FlowState:
        raise NotImplementedError("run method is not implemented")


class CommanLineRunner(Runner):
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
        for template in self.templates:
            if template.id in self.template_dict:
                raise ValueError(f"Template id {template.id} is duplicated.")
            self.template_dict[template.id] = template

    def run(
        self,
        start_template: Optional["Template | TemplateId"] = None,
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
        while 1:
            flow_state = template.render(flow_state)
            print(flow_state.get_last_message().content)
            from IPython import embed

            embed()
            if flow_state.jump is not None:
                if not isinstance(flow_state.jump, "Template"):
                    template = self._search_template(flow_state.jump)
                flow_state.jump = None
            else:
                logger.info(f"No jump is set. Flow is finished.")
                break
        return flow_state

    def _search_template(self, template_id: TemplateId) -> "Template":
        if template_id not in self.template_dict:
            raise ValueError(f"Template id {template_id} is not found.")
        return self.template_dict[template_id]
