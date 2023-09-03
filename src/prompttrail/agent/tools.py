import enum
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type, TypeAlias, Union

from pydantic import BaseModel
from typing_inspect import get_args, is_optional_type  # type: ignore

from prompttrail.agent import State

logger = logging.getLogger(__name__)


class ToolResult(BaseModel, metaclass=ABCMeta):
    """A class to represent the result of a tool called.

    ToolResult class is responsible for conveying the result of the tool to LLM (`show` method).
    The user must inherit this class and define `show` as class method.
    ToolResult compose a Tool with ToolArgument.
    """

    @abstractmethod
    def show(self) -> Dict[str, Any]:
        """Return a dictionary to show LLM the result of the tool. The dictionary should be JSON serializable."""
        # TODO: It may be possible to automatically encode this with pydantic
        ...


# TODO: Can be used with Literal if we use typevar?
FunctionCallingArgumentValueType: TypeAlias = Union[
    str,
    Optional[str],
    int,
    Optional[int],
    float,
    Optional[float],
    bool,
    Optional[bool],
    enum.Enum,
    Optional[enum.Enum],
]

FunctionCallingArgumentValueTypeType: TypeAlias = Union[
    Type[str],
    Type[Optional[str]],
    Type[int],
    Type[Optional[int]],
    Type[float],
    Type[Optional[float]],
    Type[bool],
    Type[Optional[bool]],
    Type[enum.Enum],
    Type[Optional[enum.Enum]],
]


class ToolArgument(BaseModel):
    """A class to represent an argument of a tool. User must inherit this class to create a tool.

    ToolArgument class is responsible for conveying the usage of the tool to LLM (`show` method).
    The user must inherit this class and define `description` and `value` as class variables.
    Typing information of `value` is used to generate the JSON schema for the function calling.
    ToolArgument compose a Tool with ToolResult.
    """

    description: str  # This is a little opinionated, but I think it's a good idea to always have a description for each argument.
    """ The description of the argument. For example, if the argument is "place", the description can be "The name of the place". The description should be set when the user inherits this class. """
    value: FunctionCallingArgumentValueType
    """ The value of the argument. When user inherits this class, the type of the value should be specified. For example, if the argument is "place", the type can be str. """

    @classmethod
    def get_name(cls) -> str:
        """The name of the argument is determined by the class name automactically. If the user want to change the name, they can override this method."""
        return cls.__name__.lower()

    @classmethod
    def is_required(cls) -> bool:
        """Return True if the argument is required, False otherwise. Determined by the type annotation of value in Tool. Note that the value cannot be Optional. The optionality of the argument should be determined in Tool."""
        return not is_optional_type(cls)

    @classmethod
    def get_value_type(cls) -> FunctionCallingArgumentValueTypeType:
        """Return the type of the value. Determined by the type annotation of value."""
        return cls.model_fields["value"].annotation  # type: ignore

    @classmethod
    def get_description(cls) -> str:
        """Return the description of the argument. Determined by the default value of description."""
        # We need this because ToolArgument is usually accessed as a class, not an instance.
        return cls.model_fields["description"].default  # type: ignore


class FunctionCallingPartialProperty(BaseModel):
    """A internal class to represent a property of a function calling. This class is used to generate the JSON schema for the function calling."""

    type: str
    required: bool
    enum: Optional[List[str]] = None


class FunctionCallingProperty(FunctionCallingPartialProperty):
    """A class to represent a property of a function calling. This class is used to generate the JSON schema for the function calling."""

    name: str
    description: str


def function_calling_type_to_partial_property(
    T: Type[FunctionCallingArgumentValueType],
) -> FunctionCallingPartialProperty:
    """Convert a type of a function calling to a partial property. This function is used to generate the JSON schema for the function calling."""
    if T == str:
        return FunctionCallingPartialProperty(type="string", required=True)
    elif T == Optional[str]:
        return FunctionCallingPartialProperty(type="string", required=False)
    elif T == int:
        return FunctionCallingPartialProperty(type="int", required=True)
    elif T == Optional[int]:
        return FunctionCallingPartialProperty(type="int", required=False)
    elif T == float:
        return FunctionCallingPartialProperty(
            type="float", required=True
        )  # TODO: Not documented in OpenAI API?
    elif T == Optional[float]:
        return FunctionCallingPartialProperty(
            type="float", required=False
        )  # TODO: Not documented in OpenAI API?
    elif T == bool:
        return FunctionCallingPartialProperty(type="boolean", required=True)
    elif T == Optional[bool]:
        return FunctionCallingPartialProperty(type="boolean", required=False)
    # TODO: Whatever reason, type checking does not work here
    elif is_optional_type(T):
        args = get_args(T)  # type: ignore
        for arg in args:  # type: ignore
            if issubclass(arg, enum.Enum):
                return FunctionCallingPartialProperty(
                    type="string", required=False, enum=[x.value for x in arg]
                )
        raise ValueError("Invalid type", T)
    else:
        if not issubclass(T, enum.Enum):
            raise ValueError("Invalid type", T)

        return FunctionCallingPartialProperty(
            type="string", required=True, enum=[x.value for x in T]
        )


def convert_property_to_api_call_parts(prop: FunctionCallingProperty):
    """Convert a partial property to the API call parts. This function is used to generate the JSON schema for the function calling."""
    result = {
        "type": prop.type,
        "description": prop.description,
        "enum": prop.enum,
    }
    if result["enum"] is None:
        del result["enum"]
    return result


def check_arguments(
    args_made_by_api: Dict[str, Any], args_of_function: Sequence[Type[ToolArgument]]
) -> Sequence[ToolArgument]:
    """Check if the arguments made by API are valid compared to the arguments of the function."""

    result = []
    for arg_passed in args_of_function:
        if arg_passed.is_required():
            if arg_passed.get_name() not in args_made_by_api:
                raise ValueError("Missing required argument:", arg_passed.get_name())
        if arg_passed.get_name() in args_made_by_api:
            result.append(arg_passed(value=args_made_by_api[arg_passed.get_name()]))  # type: ignore
    # Warn if there are unused arguments
    for arg_name in args_made_by_api:
        if not any(
            [arg_name == arg_passed.get_name() for arg_passed in args_of_function]
        ):
            logger.warning("Unused argument:" + arg_name)
    return result  # type: ignore


class Tool(object, metaclass=ABCMeta):
    """A class to represent a tool. User must inherit this class to create a tool.

    Tool class is responsible for conveying the usage of the tool to LLM (`show` method) and actually calling the tool (`call` method) and return the result.
    The user must inherit this class and define `name`, `description`, `argument_types`, and `result_type` as class variables.
    Typing information of `argument_types` and `result_type` is used to generate the JSON schema for the function calling.
    """

    name: str
    description: str
    argument_types: Sequence[Type[ToolArgument]]
    result_type: Type[ToolResult]

    @abstractmethod
    def _call(self, args: Sequence[ToolArgument], state: "State") -> ToolResult:
        """This method should be implemented by the user. This method is the actual function of the tool. Return the result of the tool as the user defined class that inherits `ToolResult`.

        Args:
            args (Sequence[ToolArgument]): The arguments returned by the API.
            state (State): The current state of the agent.

        Returns:
            ToolResult: The result of the tool.
        """

        ...

    def call(self, args: Sequence[ToolArgument], state: "State") -> ToolResult:
        """Call the function and return the result based on the args returned by the API. This method is usually not overriden by the user.

        Args:
            args (Sequence[ToolArgument]): The arguments returned by the API.
            state (State): The current state of the agent.

        Returns:
            ToolResult: The result of the tool.
        """
        for arg in args:
            if not any([isinstance(arg, x) for x in self.argument_types]):
                raise ValueError("Given argument type is not supported", arg.__class__)
        for arg_predef in self.argument_types:
            flag = False
            missing_predefs = []
            if not is_optional_type(arg_predef):
                if not any([isinstance(x, arg_predef) for x in args]):
                    flag = True
                    missing_predefs.append(arg_predef)  # type: ignore
            if flag:
                raise ValueError("Missing required arguments", missing_predefs)
        result = self._call(args, state)
        if not isinstance(result, self.result_type):
            raise ValueError("Invalid result type")
        return result

    def show(self):
        """Return a dictionary to show LLM how to use the tool. The dictionary should be JSON serializable. This method is usually not overriden by the user."""
        partial_properties = [
            function_calling_type_to_partial_property(x.get_value_type())
            for x in self.argument_types
        ]
        properties = [
            FunctionCallingProperty(
                name=arg.get_name(),
                description=arg.get_description(),
                required=arg.is_required(),
                type=part.type,
                enum=part.enum,
            )
            for part, arg in zip(partial_properties, self.argument_types)
        ]
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    prop.name: convert_property_to_api_call_parts(prop)
                    for prop in properties
                },
                "required": [prop.name for prop in properties if prop.required],
            },
        }
