from logging import Logger
from typing import Any, Dict, Generic, List, Optional, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from prompttrail.core.errors import ParameterValidationError
from prompttrail.core.utils import Debuggable

T = TypeVar("T")


class ToolArgument(BaseModel, Generic[T]):
    """Tool argument definition with name, description, type and required flag."""

    name: str
    description: str
    value_type: type
    required: bool = True

    def validate_value(self, value: Any) -> bool:
        """Validate if the given value matches the argument's type."""
        return isinstance(value, self.value_type)


class ToolResult(BaseModel):
    """Container for a tool's execution result and optional metadata."""

    content: Any = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel, Debuggable):
    """Base class for implementing tools that can be called by LLMs.

    Provides argument validation and schema generation for function calling APIs.
    Subclasses should implement the _execute() method.
    """

    name: str
    description: str
    arguments: List[ToolArgument[Any]]
    logger: Logger = None  # type: ignore
    enable_logging: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_arguments(self) -> Self:
        """Validate that arguments are unique by name."""
        arg_names = [arg.name for arg in self.arguments]
        if len(arg_names) != len(set(arg_names)):
            raise ValueError("Arguments must have unique names")
        return self

    def get_argument(self, name: str) -> Optional[ToolArgument[Any]]:
        """Get argument by name."""
        for arg in self.arguments:
            if arg.name == name:
                return arg
        return None

    def get_arguments_dict(self) -> Dict[str, ToolArgument[Any]]:
        """Get arguments as a dictionary for backward compatibility."""
        return {arg.name: arg for arg in self.arguments}

    def model_post_init(self, *args, **kwargs):
        """Configure logging after initialization."""
        super().model_post_init(*args, **kwargs)
        self.setup_logger_for_pydantic()

    def runtime_check_arguments(self, args: Dict[str, Any]) -> None:
        """Validate that all required arguments are present and have correct types."""
        # Create a mapping of argument names for validation
        arg_map = {arg.name: arg for arg in self.arguments}

        # Check for unknown arguments
        unknown_args = set(args.keys()) - set(arg_map.keys())
        if unknown_args:
            raise ParameterValidationError(
                f"Unexpected argument: {', '.join(unknown_args)}"
            )

        # Check for required arguments
        for arg in self.arguments:
            if arg.required and arg.name not in args:
                raise ParameterValidationError(f"Missing required argument: {arg.name}")

        # Validate argument types
        for name, value in args.items():
            arg = arg_map[name]
            if not isinstance(value, arg.value_type):
                raise ParameterValidationError(
                    f"Invalid type for argument {name}: expected {arg.value_type.__name__}, got {type(value).__name__}"
                )

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool after validating arguments."""
        self.runtime_check_arguments(kwargs)
        return self._execute(kwargs)

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute tool implementation with validated arguments.

        Args:
            args: Dictionary of validated argument values

        Returns:
            ToolResult containing the execution result

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Tool._execute() must be implemented by subclass")

    def to_schema(self) -> Dict[str, Any]:
        """Generate function calling schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    arg.name: {
                        "type": "string" if arg.value_type == str else "number",
                        "description": arg.description,
                    }
                    for arg in self.arguments
                },
                "required": [arg.name for arg in self.arguments if arg.required],
            },
        }
