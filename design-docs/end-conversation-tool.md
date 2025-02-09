# EndConversationTool Design Document

## Overview
The EndConversationTool provides a mechanism to explicitly end a conversation with an optional farewell message. This tool works in conjunction with EndTemplate and ToolingTemplate to ensure proper message handling and conversation termination.

## Design Considerations

### Implementation Approach

The implementation follows these key principles:

1. **Message Handling Separation**:
   - EndTemplate directly yields farewell messages
   - EndConversationTool provides farewell messages through ToolResult
   - ToolingTemplate handles the conversion of tool results to messages

2. **Consistent Termination Flow**:
   - Both EndTemplate and EndConversationTool use ReachedEndTemplateException
   - The exception carries the farewell message
   - Runner handles the exception and ensures proper conversation termination

### Implementation Details

#### EndTemplate
```python
class EndTemplate(Template):
    def __init__(self, farewell_message: str | None = None):
        self.farewell_message = farewell_message

    def _render(self, session: "Session") -> Generator[Message, None, "Session"]:
        if self.farewell_message:
            yield Message(content=self.farewell_message, role="assistant")
        raise ReachedEndTemplateException(farewell_message=self.farewell_message)
```

#### EndConversationTool
```python
class EndConversationTool(Tool):
    name: str = "end_conversation"
    description: str = "End the current conversation"
    arguments: Dict[str, ToolArgument[Any]] = {
        "message": ToolArgument(
            name="message",
            description="Optional farewell message",
            value_type=str,
            required=False,
        )
    }

    def _execute(self, args: Dict[str, Any]) -> ToolResult:
        message = args.get("message", "Conversation ended.")
        result = ToolResult(
            content=json.dumps({
                "status": "success",
                "message": message
            })
        )
        raise ReachedEndTemplateException(farewell_message=message)
```

### Message Flow

1. **EndTemplate Path**:
   ```
   EndTemplate -> yield Message -> raise ReachedEndTemplateException -> Runner handles exception
   ```

2. **EndConversationTool Path**:
   ```
   LLM decides to end -> Tool execution -> raise ReachedEndTemplateException -> ToolingTemplate handles exception -> Runner handles exception
   ```

### Testing Strategy

1. **EndTemplate Tests**:
   - Direct message yielding
   - Exception with farewell message
   - Exception without farewell message

2. **EndConversationTool Tests**:
   - Tool execution with message
   - Tool execution without message
   - Exception handling by ToolingTemplate

3. **Integration Tests**:
   - EndTemplate in LinearTemplate
   - EndConversationTool with LLM decision making
   ```python
   def test_end_conversation_with_llm():
       # Mock LLM that decides to end conversation
       mock_model = MockModel(responses=[
           "I'll end this conversation now. {end_conversation(message='Thanks for chatting!')}"
       ])
       
       template = LinearTemplate([
           SystemTemplate(content="You can end the conversation using the end_conversation tool."),
           UserTemplate(content="Let's end this chat."),
           ToolingTemplate(tools=[EndConversationTool()])
       ])
       
       runner = CommandLineRunner(
           model=mock_model,
           template=template,
           user_interface=EchoMockInterface()
       )
       
       session = runner.run()
       assert len(session.messages) == 2  # system + farewell
       assert session.messages[-1].content == "Thanks for chatting!"
   ```

## Usage Examples

### Using EndTemplate
```python
template = LinearTemplate([
    MessageTemplate(content="Hello"),
    EndTemplate(farewell_message="Goodbye!")
])
```

### Using EndConversationTool
```python
# The tool is made available to the LLM through ToolingTemplate
template = LinearTemplate([
    SystemTemplate(content="""You are a helpful assistant.
    When the user wants to end the conversation, use the end_conversation tool with a polite farewell message."""),
    LoopTemplate([
        UserTemplate(),
        ToolingTemplate(tools=[EndConversationTool()])
    ])
])
```

## Future Considerations

1. **Enhanced Message Control**:
   - Support for structured farewell messages
   - Message formatting options
   - Multi-part farewell sequences

2. **Context Awareness**:
   - Conversation summarization on end
   - Context-dependent farewell messages
   - Session metadata preservation

3. **LLM Integration Improvements**:
   - Better tool usage guidance for LLMs
   - Context-aware farewell message generation
   - Conversation state analysis for appropriate ending