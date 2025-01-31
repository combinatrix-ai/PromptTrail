import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from prompttrail.agent.session_transformers._core import SessionTransformer
from prompttrail.agent.subroutine._base import SubroutineTemplate
from prompttrail.agent.subroutine.session_init_strategy import (
    FilteredInheritStrategy,
    SessionInitStrategy,
)
from prompttrail.agent.subroutine.squash_strategy import (
    LastMessageStrategy,
    SquashStrategy,
)
from prompttrail.agent.templates._core import Template
from prompttrail.agent.tools._base import Tool, ToolArgument, ToolResult
from prompttrail.core import Message, Session


class SubroutineTool(Tool):
    """Tool for executing templates as subroutines.

    This tool wraps a template in a SubroutineTemplate and executes it with the provided
    session initialization and message handling strategies.

    Args:
        name: Name of the tool
        description: Description of the tool's functionality
        template: The template to execute as a subroutine
        session_init_strategy: Strategy for initializing the subroutine session
        squash_strategy: Strategy for handling messages after subroutine execution
        before_transform: Transformers to apply before execution
        after_transform: Transformers to apply after execution
    """

    subroutine: Optional[SubroutineTemplate] = Field(default=None, exclude=True)

    def __init__(
        self,
        name: str,
        description: str,
        template: Template,
        session_init_strategy: Optional[SessionInitStrategy] = None,
        squash_strategy: Optional[SquashStrategy] = None,
        before_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
        after_transform: Optional[
            Union[List[SessionTransformer], SessionTransformer]
        ] = None,
    ):
        super().__init__(
            name=name,
            description=description,
            arguments={
                "input": ToolArgument(
                    name="input",
                    description="Input message for the subroutine",
                    value_type=str,
                    required=True,
                ),
                "system_message": ToolArgument(
                    name="system_message",
                    description="Optional system message for the subroutine",
                    value_type=str,
                    required=False,
                ),
            },
        )

        # Use FilteredInheritStrategy to inherit both system and user messages
        default_init_strategy = FilteredInheritStrategy(
            lambda msg: msg.role in ["system", "user"]
        )

        self.subroutine = SubroutineTemplate(
            template=template,
            template_id=name,
            session_init_strategy=session_init_strategy or default_init_strategy,
            squash_strategy=squash_strategy or LastMessageStrategy(),
            before_transform=before_transform,
            after_transform=after_transform,
        )

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the subroutine template with the provided input.

        Args:
            args: Dictionary containing 'input' and optional 'system_message'

        Returns:
            ToolResult containing the subroutine's output messages
        """
        if self.subroutine is None:
            raise RuntimeError("Subroutine template not initialized")

        # Create parent session with input
        parent_session = Session()

        # Add system message if provided
        if "system_message" in args:
            parent_session.append(
                Message(role="system", content=args["system_message"])
            )

        # Add input message
        parent_session.append(Message(role="user", content=args["input"]))

        # Debug log parent session state
        logging.debug(
            f"SubroutineTool._execute parent session messages: {parent_session.messages}"
        )

        # Execute subroutine with parent session
        messages = []
        for message in self.subroutine.render(parent_session):
            messages.append(message)
            logging.debug(f"SubroutineTool._execute received message: {message}")

        # Return results
        result = ToolResult(
            content=messages[-1].content if messages else None,
            metadata={"messages": messages},
        )
        logging.debug(f"SubroutineTool._execute result: {result}")
        return result


# Rebuild model after class definition
SubroutineTool.model_rebuild()
