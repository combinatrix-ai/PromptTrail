from logging import Logger
from typing import Any, Dict, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.utils import Debuggable

T = TypeVar("T")


class ToolArgument(BaseModel, Generic[T]):
    """Tool argument class"""

    name: str
    description: str
    value_type: type
    required: bool = True

    def validate_value(self, value: Any) -> bool:
        """Validate argument value"""
        return isinstance(value, self.value_type)


class ToolResult(BaseModel):
    """Tool result class"""

    content: Any = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel, Debuggable):
    """Base tool class"""

    name: str
    description: str
    arguments: Dict[str, ToolArgument[Any]]
    # To meet the Pydantic BaseModel requirements,
    # we need to define the logger attribute as a class attribute.
    logger: Logger = None  # type: ignore
    enable_logging: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)
        self.setup_logger_for_pydantic()

    def validate_arguments(self, args: Dict[str, Any]) -> None:
        """Validate tool arguments"""
        # Check for unknown arguments
        unknown_args = set(args.keys()) - set(self.arguments.keys())
        if unknown_args:
            raise ParameterValidationError(
                f"Unexpected argument: {', '.join(unknown_args)}"
            )

        # Check for required arguments
        for name, arg in self.arguments.items():
            if arg.required and name not in args:
                raise ParameterValidationError(f"Missing required argument: {name}")

        # Validate argument types
        for name, value in args.items():
            arg = self.arguments[name]
            if not isinstance(value, arg.value_type):
                raise ParameterValidationError(
                    f"Invalid type for argument {name}: expected {arg.value_type.__name__}, got {type(value).__name__}"
                )

    def execute(self, **kwargs) -> ToolResult:
        """Execute tool with arguments"""
        # Validate arguments
        self.validate_arguments(kwargs)

        # Execute tool implementation with validated arguments
        return self._execute(kwargs)

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute tool implementation with validated arguments"""
        raise NotImplementedError("Tool._execute() must be implemented by subclass")

    def to_schema(self) -> Dict[str, Any]:
        """Convert tool to schema format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: {
                        "type": "string" if arg.value_type == str else "number",
                        "description": arg.description,
                    }
                    for name, arg in self.arguments.items()
                },
                "required": [
                    name for name, arg in self.arguments.items() if arg.required
                ],
            },
        }
