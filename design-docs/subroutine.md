# Implementation Proposal for Subroutine Feature in Templates

## Current Implementation

### Control Flow Templates
```python
class ControlTemplate(Template, metaclass=ABCMeta):
    """Base class for control flow templates."""
    pass

class LinearTemplate(ControlTemplate):
    """Sequential execution of templates"""
    pass

class LoopTemplate(ControlTemplate):
    """Repeated execution with conditions"""
    pass

class IfTemplate(ControlTemplate):
    """Conditional branching"""
    pass
```

Current Status:
- Basic control flow templates implemented
- Clear template hierarchy
- Stack-based execution tracking

## Proposed Enhancements

### 1. Introduction of SessionInitStrategy

```python
class SessionInitStrategy(ABC):
    """Base class defining how to initialize subroutine session"""
    
    @abstractmethod
    def initialize(self, parent_session: Session) -> Session:
        """Create and initialize a session for subroutine
        
        Args:
            parent_session: Parent session that invoked the subroutine
            
        Returns:
            Initialized session for subroutine use
        """
        pass

class CleanSessionStrategy(SessionInitStrategy):
    """Strategy to create a clean session with no messages"""
    def initialize(self, parent_session: Session) -> Session:
        return Session(
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner
        )

class InheritSystemStrategy(SessionInitStrategy):
    """Strategy to inherit system messages from parent"""
    def initialize(self, parent_session: Session) -> Session:
        system_messages = [
            msg for msg in parent_session.messages 
            if msg.role == "system"
        ]
        return Session(
            messages=system_messages.copy(),
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner
        )

class LastNMessagesStrategy(SessionInitStrategy):
    """Strategy to inherit last N messages from parent"""
    def __init__(self, n: int):
        self.n = n
        
    def initialize(self, parent_session: Session) -> Session:
        last_messages = parent_session.messages[-self.n:] if parent_session.messages else []
        return Session(
            messages=last_messages.copy(),
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner
        )

class FilteredInheritStrategy(SessionInitStrategy):
    """Strategy to inherit messages based on custom filter"""
    def __init__(self, filter_fn: Callable[[Message], bool]):
        self.filter_fn = filter_fn
        
    def initialize(self, parent_session: Session) -> Session:
        filtered_messages = [
            msg for msg in parent_session.messages 
            if self.filter_fn(msg)
        ]
        return Session(
            messages=filtered_messages.copy(),
            metadata=parent_session.metadata.copy(),
            runner=parent_session.runner
        )
```

### 2. Introduction of SquashStrategy

```python
class SquashStrategy(ABC):
    """Base class defining message squashing strategy"""
    
    def initialize(self, parent_session: Session, subroutine_session: Session) -> None:
        """Initialize session information"""
        self.parent_session = parent_session
        self.subroutine_session = subroutine_session
    
    @abstractmethod
    def squash(self, messages: List[Message]) -> List[Message]:
        """Execute message squashing process"""
        pass

class LastMessageStrategy(SquashStrategy):
    """Strategy to retain only the last message"""
    def squash(self, messages: List[Message]) -> List[Message]:
        return [messages[-1]] if messages else []

class FilterByRoleStrategy(SquashStrategy):
    """Strategy to retain messages with specific roles"""
    def __init__(self, roles: List[str]):
        super().__init__()
        self.roles = roles
    
    def squash(self, messages: List[Message]) -> List[Message]:
        return [msg for msg in messages if msg.role in self.roles]
```

### 3. SubroutineTemplate Implementation

```python
class SubroutineTemplate(Template):
    def __init__(
        self,
        template: Template,
        template_id: Optional[str] = None,
        session_init_strategy: Optional[SessionInitStrategy] = None,
        squash_strategy: Optional[SquashStrategy] = None,
        before_transform: Optional[Union[List[SessionTransformer], SessionTransformer]] = None,
        after_transform: Optional[Union[List[SessionTransformer], SessionTransformer]] = None,
    ):
        super().__init__(
            template_id=template_id,
            before_transform=before_transform,
            after_transform=after_transform,
        )
        self.template = template
        self.session_init_strategy = session_init_strategy or CleanSessionStrategy()
        self.squash_strategy = squash_strategy or LastMessageStrategy()

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        # Initialize subroutine session
        temp_session = self.session_init_strategy.initialize(session)
        self.squash_strategy.initialize(session, temp_session)
        
        messages: List[Message] = []
        try:
            # Execute subroutine
            async for message in self.template.render(temp_session):
                messages.append(message)
                yield message
                
            # Apply squash strategy
            selected_messages = self.squash_strategy.squash(messages)
            for msg in selected_messages:
                session.append(msg)
                
        finally:
            temp_session.cleanup()
        
        return session
```

## Usage Examples

```python
# Example 1: Clean subroutine with system message
subroutine = SubroutineTemplate(
    template=some_template,
    session_init_strategy=CleanSessionStrategy(),
    squash_strategy=LastMessageStrategy()
)

# Example 2: Inherit system context
subroutine = SubroutineTemplate(
    template=some_template,
    session_init_strategy=InheritSystemStrategy(),
    squash_strategy=FilterByRoleStrategy(roles=["assistant"])
)

# Example 3: Last N messages context
subroutine = SubroutineTemplate(
    template=some_template,
    session_init_strategy=LastNMessagesStrategy(n=3),
    squash_strategy=LastMessageStrategy()
)

# Example 4: Custom filtering
def is_important(msg: Message) -> bool:
    return msg.metadata.get("importance", 0) > 0.5

subroutine = SubroutineTemplate(
    template=some_template,
    session_init_strategy=FilteredInheritStrategy(is_important),
    squash_strategy=LastMessageStrategy()
)
```

## Benefits

1. Message Management
   - Flexible message filtering
   - Clear separation of concerns
   - Improved debugging capabilities

2. Session Control
   - Controlled session initialization
   - Isolated subroutine execution
   - Flexible message inheritance

3. Extensibility
   - Custom initialization strategies
   - Custom squash strategies
   - Reusable components

## Implementation Steps

1. Phase 1: Core Implementation
   - SessionInitStrategy base class
   - SquashStrategy base class
   - Basic strategy implementations
   - SubroutineTemplate core functionality

2. Phase 2: Integration
   - Session state management
   - Message propagation rules
   - Error handling

3. Phase 3: Extensions
   - Additional strategies
   - Performance optimization
   - Documentation and examples

## Considerations

1. Performance
   - Memory usage for temporary sessions
   - Message copying overhead
   - Strategy execution efficiency

2. Error Handling
   - Subroutine failure recovery
   - State consistency
   - Resource cleanup

3. Debugging
   - Message flow visualization
   - Strategy selection logging
   - State inspection tools

## Future Prospects

1. Advanced Features
   - Conditional initialization
   - Dynamic strategy selection
   - State-based filtering

2. Integration
   - Debug UI support
   - Monitoring tools
   - Performance analytics

3. Extensions
   - Custom strategy framework
   - Plugin system
   - State management tools