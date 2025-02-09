# Event Handling Enhancement Proposal for Runner and Template

## Introduction

This document proposes enhancing the existing `Runner` and `Template` classes to support `Event` objects alongside `Message` objects. This addition will allow the system to handle a broader range of interactions and events, enabling more complex workflows and integrations.

## Proposed Changes

### 1. Event Classes

We propose implementing two primary event classes:

```python
@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any]

class UserInteractionEvent(Event):
    instruction: Optional[str] = None
    default: Optional[str] = None

    def __init__(
        self,
        payload: Dict[str, Any] = {},
        instruction: Optional[str] = None,
        default: Optional[str] = None,
    ):
        super().__init__(event_type="user_interaction", payload=payload)
        self.instruction = instruction
        self.default = default
```

The base `Event` class will provide a foundation for all event types, while `UserInteractionEvent` will specifically handle user interaction flows.

### 2. Template Modifications

All template classes will need to be updated to support yielding both `Message` and `Event` objects:

```python
def _render(
    self, 
    session: "Session"
) -> Generator[Union[Message, Event], None, "Session"]:
```

This change will need to be implemented across:
- Base Template class
- MessageTemplate
- GenerateTemplate
- UserTemplate
- AssistantTemplate
- Control templates (LinearTemplate, LoopTemplate, etc.)

### 3. Runner Updates

The `CommandLineRunner` will need to be enhanced to handle both Message and Event objects:

```python
if isinstance(obj, Message):
    # Handle message output
    print("From: " + cutify_role(message.role))
    if message.content:
        print("message: ", message.content)
    # ... additional message handling ...
elif isinstance(obj, Event):
    event = obj
    if isinstance(event, UserInteractionEvent):
        instruction = event.instruction or "Input: "
        default = event.default or None
        content = self.user_interface.ask(session, instruction, default)
        session.messages.append(
            Message(
                role="user",
                content=content,
                metadata=session.metadata,
            )
        )
    else:
        self.warning(f"Unknown event type: {type(event)}")
```

### 4. User Interface Integration

The UserInterface class will need to be updated to support the new event-based interaction model:

```python
def ask(
    self,
    session: Session,
    instruction: Optional[str],
    default: Optional[str] = None,
) -> str:
```

### 5. Subroutine Support

Subroutines will need modifications to properly handle events:

```python
def _render(
    self,
    session: "Session"
) -> Generator[Union[Message, Event], None, "Session"]:
    # ... setup code ...
    for message in self.template.render(temp_session):
        if isinstance(message, Event):
            yield message
            continue
        messages.append(message)
        yield message
```

Initially, SubroutineTool will not support events and will raise an error if it receives one. Support for events in SubroutineTool can be considered as a future enhancement.

## Expected Benefits

- **Interactive Flows**: The system will properly support interactive user input through events
- **Type Safety**: All event handling will be properly typed and checked
- **Backwards Compatibility**: Existing message-based templates will continue to work without modification
- **Extensibility**: New event types can be added by extending the base Event class
- **Clear Separation**: Events and Messages will be clearly separated in the type system

## Implementation Plan

1. **Phase 1: Core Event System**
   - Implement Event and UserInteractionEvent classes
   - Update Template base class to support events
   - Modify Runner to handle basic events

2. **Phase 2: Template Updates**
   - Update all template classes to support events
   - Add event handling to control templates
   - Implement user interaction flow

3. **Phase 3: Testing and Documentation**
   - Add comprehensive tests for event system
   - Update documentation
   - Create example templates using events

## Risks and Mitigations

1. **Complexity**
   - Risk: Adding events increases system complexity
   - Mitigation: Clear documentation and type safety

2. **Backwards Compatibility**
   - Risk: Breaking existing templates
   - Mitigation: Careful type updates and testing

3. **Performance**
   - Risk: Event processing overhead
   - Mitigation: Efficient event handling implementation

## Future Considerations

1. **Additional Event Types**: System should be designed to easily add new event types

2. **Subroutine Event Support**: Future enhancement to add event support to SubroutineTool

3. **Event Handlers**: Potential for more sophisticated event handling system

4. **Async Support**: Possible addition of async event processing

## Migration Guide for Template Authors

1. Update template render methods to include Event in their return type:
```python
def _render(
    self,
    session: "Session"
) -> Generator[Union[Message, Event], None, "Session"]:
```

2. Use UserInteractionEvent for user input instead of direct UI calls:
```python
yield UserInteractionEvent(
    instruction="Please enter your name:",
    default="User"
)
```

3. Test templates with both Message and Event handling

## Timeline

## Conclusion

This proposal outlines a clear path to adding event support to the system while maintaining backwards compatibility and providing a foundation for future enhancements. The proposed changes will significantly improve the system's ability to handle interactive workflows and complex user interactions.