import enum
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type, TypeAlias, Union

from pydantic import BaseModel
from typing_inspect import get_args, is_optional_type  # type: ignore

from prompttrail.agent.core import FlowState

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    ...

    @abstractmethod
    def show(self) -> Dict[str, Any]:
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
    description: str  # This is a little opinionated, but I think it's a good idea to always have a description for each argument.
    value: FunctionCallingArgumentValueType

    @classmethod
    def get_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def is_required(cls) -> bool:
        return not is_optional_type(cls)

    @classmethod
    def get_value_type(cls) -> FunctionCallingArgumentValueTypeType:
        return cls.model_fields["value"].annotation  # type: ignore

    @classmethod
    def get_description(cls) -> str:
        # We need this because ToolArgument is usually accessed as a class, not an instance.
        return cls.model_fields["description"].default  # type: ignore


class FunctionCallingPartialProperty(BaseModel):
    type: str
    required: bool
    enum: Optional[List[str]] = None


class FunctionCallingProperty(FunctionCallingPartialProperty):
    name: str
    description: str


def function_calling_type_to_partial_property(
    T: Type[FunctionCallingArgumentValueType],
) -> FunctionCallingPartialProperty:
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
            logger.warning("Unused argument:", arg_name)
    return result  # type: ignore


class Tool(object):
    name: str
    description: str
    argument_types: Sequence[Type[ToolArgument]]
    result_type: Type[ToolResult]

    @abstractmethod
    def _call(self, args: Sequence[ToolArgument]) -> ToolResult:
        ...

    def call(self, args: Sequence[ToolArgument], flow_state: "FlowState") -> ToolResult:
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
        result = self._call(args)
        if not isinstance(result, self.result_type):
            raise ValueError("Invalid result type")
        return result

    def show(self):
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
