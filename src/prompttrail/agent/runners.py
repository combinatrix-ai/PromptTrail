import asyncio
import json
import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, cast
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket

from prompttrail.agent.templates._control import EndTemplate
from prompttrail.agent.templates._core import Event, Template, UserInteractionEvent
from prompttrail.agent.user_interface import UserInterface
from prompttrail.core import Message, MessageRoleType, Model, Session
from prompttrail.core.const import ReachedEndTemplateException
from prompttrail.core.utils import Debuggable

logger = logging.getLogger(__name__)


class Runner(Debuggable, metaclass=ABCMeta):
    def __init__(
        self,
        model: Model,
        template: "Template",
        user_interface: UserInterface | None,
    ):
        """Abstract class for runner. Runner is a class to run the templates. It is responsible for rendering templates and handling user interactions."""
        super().__init__()
        self.model = model
        self.user_interface = user_interface
        self.template = template
        self.template_dict: Dict[str, Template] = {}
        visited_templates: Set[Template] = set()
        for next_template in template.walk(visited_templates):
            if next_template.template_id in self.template_dict:
                raise ValueError(
                    f"Template id {next_template.template_id} is duplicated."
                )
            self.template_dict[next_template.template_id] = next_template

    @abstractmethod
    def run(
        self,
        session: Optional["Session"] = None,
        max_messages: Optional[int] = None,
        debug_mode: bool = False,
    ) -> "Session":
        """All runners should implement this method. This method should run the templates and return the final session."""
        raise NotImplementedError("run method is not implemented")

    def search_template(self, template_like: str) -> "Template":
        """Search template by template id. If template id is not found, raise ValueError."""
        if template_like == EndTemplate.template_id:
            return EndTemplate()
        if template_like not in self.template_dict:
            raise ValueError(f"Template id {template_like} is not found.")
        return self.template_dict[template_like]


def cutify_role(role: MessageRoleType) -> str:
    """Cutify role name based on OpenAI's naming convention."""
    if role == "system":
        return "📝 system"
    if role == "user":
        return "👤 user"
    if role == "assistant":
        return "🤖 assistant"
    if role == "function":
        return "🛠️ function"
    if role == "tool_result":
        return "📊 tool_result"
    return role


def pretty_print_metadata(metadata: Dict[str, Any]) -> str:
    TRUNCATION_THRESHOLD = 20  # Only truncate strings longer than this
    TRUNC_HEAD = 5  # Number of characters to keep at the start
    TRUNC_TAIL = 5  # Number of characters to keep at the end

    def format_value(value: Any) -> str:
        if isinstance(value, str):
            # Truncate string if it is too long.
            if len(value) > TRUNCATION_THRESHOLD:
                value = value[:TRUNC_HEAD] + "..." + value[-TRUNC_TAIL:]
            # Use json.dumps to produce a properly escaped string (with quotes)
            return json.dumps(value)
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, dict):
            # Nested dictionaries use curly braces.
            return format_dict(value)
        elif isinstance(value, list):
            # Format each element recursively for lists.
            return "[" + ", ".join(format_value(item) for item in value) + "]"
        elif value is None:
            return "None"
        else:
            # For custom objects, simply show the class name followed by ()
            return f"{value.__class__.__name__}()"

    def format_dict(d: dict) -> str:
        # Top-level dictionary uses parentheses, nested ones use curly braces.
        open_delim = "{"
        close_delim = "}"
        formatted_items = []
        for key, val in d.items():
            # We assume keys are strings and print them as is.
            formatted_items.append(f'"{key}": {format_value(val)}')
        return open_delim + ", ".join(formatted_items) + close_delim

    return format_dict(metadata)


def print_message(message: Message):
    print("From: " + cutify_role(message.role))
    if message.content:
        print("message: ", message.content)
    if message.tool_use:
        print("tool_use: ", message.tool_use)
    if message.metadata:
        print("metadata: ", pretty_print_metadata(message.metadata))


class CommandLineRunner(Runner):
    def run(
        self,
        session: Optional["Session"] = None,
        max_messages: Optional[int] = 100,
        debug_mode: bool = False,
    ) -> "Session":
        """Command line runner. This runner is for debugging purpose. It prints out the messages to the console.

        Args:
            session (Optional[Session], optional): If set, use the session given. Otherwise, create a new session. Defaults to None.
            max_messages (Optional[int], optional): Maximum number of messages to yield. If number of messages exceeds this number, the conversation is forced to stop. Defaults to 100.
            debug_mode (bool, optional): If set, print out debug messages. Defaults to False.

        Returns:
            Session: Final session of the conversation.
        """

        if self.user_interface is None:
            self.info("User interface is not set. Running in non-interactive mode.")

        # set / update session
        if session is None:
            session = Session(
                runner=self,
                debug_mode=debug_mode,
            )
        else:
            if session.runner is None or session.runner != self:
                session.runner = self
            session.debug_mode = debug_mode or session.debug_mode

        n_messages = 0
        template = self.template
        gen = template.render(session)
        print("===== Start =====")
        while 1:
            # render template until exhausted
            try:
                obj = next(gen)
            except ReachedEndTemplateException:
                self.warning(
                    "End template %s is reached. Exiting the session.",
                    EndTemplate.template_id,
                )
                break
            except StopIteration as e:
                # For generator, type support for return value is not so good.
                session = cast(Session, e.value)
                break
            if isinstance(obj, Message):
                message = obj
                print("From: " + cutify_role(message.role))
                if message.content:
                    print("message: ", message.content)
                if message.tool_use:
                    print("tool_use: ", message.tool_use)
                if message.metadata:
                    print("metadata: ", pretty_print_metadata(message.metadata))
                    n_messages += 1
            elif isinstance(obj, Event):
                event = obj
                if isinstance(event, UserInteractionEvent):
                    if self.user_interface is None:
                        self.error(
                            "User interface is not set. But user interaction event is found."
                        )
                        raise ValueError("User interface is not set.")
                    instruction = event.instruction or "Input: "
                    default = event.default or None
                    content = self.user_interface.ask(session, instruction, default)
                    message = Message(
                        role="user",
                        content=content,
                        metadata=session.metadata,
                    )
                    session.messages.append(message)
                    print_message(message)
                else:
                    self.warning(f"Unknown event type: {type(event)}")
                    raise ValueError(f"Unknown event type: {type(event)}")
            else:
                self.warning(f"Unknown object type: {type(obj)}")

            if max_messages and n_messages >= max_messages:
                self.warning(
                    "Max messages %s is reached. Exiting the session.", max_messages
                )
                break
            print("=================")
        print("====== End ======")
        return session


@dataclass
class SessionState:
    session: Session
    current_event: Optional[UserInteractionEvent] = None
    is_running: bool = False
    websocket: Optional[WebSocket] = None


class APIRunner(Runner):
    def __init__(
        self,
        model: Model,
        template: "Template",
        user_interface: UserInterface,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        super().__init__(model, template, user_interface)
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.sessions: Dict[str, SessionState] = {}
        self._setup_routes()

    def run(
        self,
        session: Optional["Session"] = None,
        max_messages: Optional[int] = 100,
        debug_mode: bool = False,
    ) -> "Session":
        """Run method implementation for APIRunner.

        This method is mainly for compatibility with the Runner interface.
        The actual processing is done through the API endpoints.
        """
        if session is None:
            session = Session(
                runner=self,
                debug_mode=debug_mode,
            )
        else:
            if session.runner is None or session.runner != self:
                session.runner = self
            session.debug_mode = debug_mode or session.debug_mode

        # Create a session state
        session_id = str(uuid4())
        self.sessions[session_id] = SessionState(session=session)

        # Start template execution
        asyncio.create_task(self._run_template(session_id))

        return session

    def _setup_routes(self):
        @self.app.post("/sessions")
        async def create_session(request: Request):
            data = await request.json()
            metadata = data.get("metadata", {})
            session_id = str(uuid4())
            session = Session(runner=self, metadata=metadata)
            self.sessions[session_id] = SessionState(session=session)
            return {
                "session_id": session_id,
                "session": session.to_dict(),  # Use to_dict() method
            }

        @self.app.get("/sessions/{session_id}")
        async def get_session_status(session_id: str):
            state = self._get_session_state(session_id)
            return {
                "is_running": state.is_running,
                "has_event": state.current_event is not None,
                "session": state.session.to_dict(),  # Use to_dict() method
                "current_event": {
                    "instruction": state.current_event.instruction,
                    "default": state.current_event.default,
                }
                if state.current_event
                else None,
            }

        @self.app.post("/sessions/{session_id}/start")
        async def start_session(session_id: str):
            state = self._get_session_state(session_id)
            if state.is_running:
                raise HTTPException(status_code=400, detail="Session already running")
            asyncio.create_task(self._run_template(session_id))
            return {"status": "started"}

        @self.app.post("/sessions/{session_id}/input")
        async def post_input(session_id: str, request: Request):
            data = await request.json()
            user_input = data.get("input")
            if not user_input:
                raise HTTPException(status_code=400, detail="Input is required")

            state = self._get_session_state(session_id)
            if not state.current_event:
                raise HTTPException(status_code=400, detail="No pending event")

            state.session.messages.append(
                Message(
                    role="user",
                    content=user_input,
                    metadata=state.session.metadata,
                )
            )
            state.current_event = None
            return {"status": "success"}

    def _get_session_state(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.sessions[session_id]

    async def _run_template(self, session_id: str):
        state = self._get_session_state(session_id)
        state.is_running = True

        try:
            template = self.template
            gen = template.render(state.session)

            while True:
                try:
                    obj = next(gen)

                    if isinstance(obj, Message):
                        pass
                    elif isinstance(obj, UserInteractionEvent):
                        state.current_event = obj
                        # Wait for input
                        while state.current_event is not None:
                            await asyncio.sleep(0.1)

                except StopIteration as e:
                    state.session = cast(Session, e.value)
                    break
                except ReachedEndTemplateException:
                    break

        finally:
            state.is_running = False
            if state.websocket:
                await state.websocket.send_json({"type": "completed"})

    def start_server(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
