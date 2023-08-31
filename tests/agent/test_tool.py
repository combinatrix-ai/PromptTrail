import enum
from typing import Any, Dict, Optional, Sequence

import pytest

from prompttrail.agent import State
from prompttrail.agent.tools import (
    FunctionCallingPartialProperty,
    FunctionCallingProperty,
    Tool,
    ToolArgument,
    ToolResult,
    convert_property_to_api_call_parts,
    function_calling_type_to_partial_property,
)


class ToolResult1(ToolResult):
    key: str

    def show(self) -> Dict[str, Any]:
        return {"key": self.key}


class ToolArgument1(ToolArgument):
    description: str = "test description"
    value: int


class ToolArgument2(ToolArgument):
    description: str = "test description2"
    value: Optional[int]


class ToolArgumentEnumType(enum.Enum):
    A = "A"
    B = "B"


class ToolArgumentEnum(ToolArgument):
    description: str = "test description3"
    value: ToolArgumentEnumType


class MyTool(Tool):
    name = "mytool"
    description = "A test tool"
    argument_types = [ToolArgument1]
    result_type = ToolResult1

    def _call(self, args: Sequence[ToolArgument], state: State) -> ToolResult:
        return ToolResult1(key="key")


def test_show_method():
    result = ToolResult1(key="value")
    assert result.show() == {"key": "value"}


def test_function_calling_type_to_partial_property():
    # x vs Optional[x]
    assert function_calling_type_to_partial_property(
        str
    ) == FunctionCallingPartialProperty(type="string", required=True)
    assert function_calling_type_to_partial_property(
        Optional[str]
    ) == FunctionCallingPartialProperty(type="string", required=False)
    # enum
    assert function_calling_type_to_partial_property(
        ToolArgumentEnumType
    ) == FunctionCallingPartialProperty(type="string", required=True, enum=["A", "B"])
    assert function_calling_type_to_partial_property(
        Optional[ToolArgumentEnumType]
    ) == FunctionCallingPartialProperty(type="string", required=False, enum=["A", "B"])


def test_convert_property_to_api_call_parts():
    prop = FunctionCallingProperty(
        type="string", required=True, name="name", description="description"
    )
    assert convert_property_to_api_call_parts(prop=prop) == {
        "type": "string",
        "description": "description",
    }
    prop = FunctionCallingProperty(
        type="int", required=False, name="name", description="description"
    )
    assert convert_property_to_api_call_parts(prop) == {
        "type": "int",
        "description": "description",
    }
    prop = FunctionCallingProperty(
        type="string",
        required=True,
        name="name",
        description="description",
        enum=["a", "b"],
    )
    assert convert_property_to_api_call_parts(prop) == {
        "type": "string",
        "description": "description",
        "enum": ["a", "b"],
    }


def test_tool_argument_class():
    assert ToolArgument1.get_name() == "toolargument1"
    assert ToolArgument1.is_required()
    assert ToolArgument1.get_value_type() == int
    assert ToolArgument1.get_description() == "test description"
    with pytest.raises(AttributeError):
        # Value itself cannot be accessed when it is not instantiated
        ToolArgument1.value

    # Optional
    assert ToolArgument2.get_name() == "toolargument2"
    # assert not toolargument2.is_required()
    assert ToolArgument2.get_value_type() == Optional[int]
    assert ToolArgument2.get_description() == "test description2"
    with pytest.raises(AttributeError):
        # Value itself cannot be accessed when it is not instantiated
        ToolArgument2.value


def test_tool_argument_instance():
    arg = ToolArgument1(value=5)
    assert arg.get_name() == "toolargument1"
    assert arg.is_required()
    assert arg.get_value_type() == int
    assert arg.get_description() == "test description"
    assert arg.value == 5

    # Optional
    arg = ToolArgument2(value=None)
    assert arg.get_name() == "toolargument2"
    # assert not arg.is_required()
    assert arg.get_value_type() == Optional[int]
    assert arg.get_description() == "test description2"
    assert arg.value is None

    # Enum
    arg = ToolArgumentEnum(value=ToolArgumentEnumType.A)
    assert arg.get_name() == "toolargumentenum"
    assert arg.is_required()
    assert arg.get_value_type() == ToolArgumentEnumType
    assert arg.get_description() == "test description3"
    assert arg.value == ToolArgumentEnumType.A


def test_tool():
    tool = MyTool()
    args = [ToolArgument1(value=5)]
    result = tool.call(args=args, state=State())
    assert isinstance(result, ToolResult1)
    assert tool.show() == {
        "name": "mytool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {
                "toolargument1": {"type": "int", "description": "test description"}
            },
            "required": ["toolargument1"],
        },
    }


# TODO: Make Test Scenario: Cake chain store

# class Place(ToolArgument):
#     description: str = "The city to search"
#     value: str


# class CackTypeType(enum.Enum):
#     Chocolate = "Chocolate"
#     Strawberry = "Strawberry"
#     Cheese = "Cheese"


# class CakeType(ToolArgument):
#     # No description
#     value: CackTypeType


# class LowerPrice(ToolArgument):
#     value: int


# class UpperPrice(ToolArgument):
#     value: int


# class SpecialMessage(ToolArgument):
#     description: str = "The special message to write on the cake if any"
#     # Optional
#     value: Optional[str]


# class Cake(BaseModel):
#     name: str
#     price: int


# class CakeSearchResult(ToolResult):
#     # No description
#     cakes: List[Cake]

#     def show(self) -> Dict[str, List[Dict[str, Any]]]:
#         return {
#             "cakes": [{"name": cake.name, "price": cake.price} for cake in self.cakes]
#         }
