# Template Model Override Design Proposal

## Context

In LLM-based applications, the ability to optimize cost and performance by choosing appropriate models for different tasks is crucial. We propose adding model override capabilities to templates to enable fine-grained control over model selection.

## Proposal

### Core Concept: Model Override

Templates should have the ability to override their model through a `model` attribute:

```python
class GenerateTemplate(MessageTemplate):
    def __init__(
        self,
        role: MessageRoleType,
        template_id: Optional[str] = None,
        before_transform: Optional[Union[List[SessionTransformer], SessionTransformer]] = None,
        after_transform: Optional[Union[List[SessionTransformer], SessionTransformer]] = None,
        enable_logging=True,
        disable_jinja=False,
        model: Optional[Model] = None,  # Model override capability
    ):
        super().__init__(...)
        self.model = model

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        if not session.runner:
            raise ValueError("runner is not set")
        
        # Use overridden model if provided
        model = self.model or session.runner.model
        response = model.send(session)
        ...
```

This enables various optimization strategies:

1. **Cost Optimization**
   - Use cheaper models for simple tasks
   - Reserve expensive models for complex operations
   ```python
   simple_task = GenerateTemplate(role="assistant", model=GPT35Model())  # Cheaper
   complex_task = GenerateTemplate(role="assistant", model=GPT4Model())  # More expensive
   ```

2. **Performance Optimization**
   - Use faster models for time-sensitive operations
   - Use more accurate models for quality-critical tasks
   ```python
   quick_response = GenerateTemplate(role="assistant", model=FastModel())
   high_quality = GenerateTemplate(role="assistant", model=AccurateModel())
   ```

3. **Specialized Model Utilization**
   - Use code-specialized models for code generation
   - Use math-specialized models for mathematical tasks
   ```python
   code_gen = GenerateTemplate(role="assistant", model=CodeModel())
   math_solve = GenerateTemplate(role="assistant", model=MathModel())
   ```

### Special Case: Subroutine Environment Isolation

Subroutine templates have a unique requirement for environment isolation. Therefore, they support both `model` override and complete `runner` override:

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
        runner: Optional[Runner] = None,  # Complete environment override (Subroutine-specific)
        model: Optional[Model] = None,    # Model-only override
    ):
        if runner is not None and model is not None:
            raise ValueError("Cannot set both runner and model - use one or the other")
        
        super().__init__(...)
        self.runner = runner
        self.model = model

    def _render(self, session: Session) -> Generator[Message, None, Session]:
        temp_session = self.session_init_strategy.initialize(session)
        self.squash_strategy.initialize(session, temp_session)
        
        # Environment isolation (Subroutine-specific)
        if self.runner:
            temp_session.runner = self.runner
        elif self.model:
            temp_session.runner = Runner(
                model=self.model,
                user_interface=session.runner.user_interface if session.runner else None
            )
        else:
            # Create copy of parent runner for isolation
            temp_session.runner = copy.deepcopy(session.runner)
```

The runner attribute is specific to Subroutine because:
1. Subroutines need complete environment isolation
2. They manage their own session context
3. They can act as independent execution environments

### Usage Examples

1. **Basic Model Override (Any Template)**
```python
# Override model for specific tasks
template = GenerateTemplate(
    role="assistant",
    model=SpecializedModel()
)
```

2. **Complete Environment Control (Subroutine Only)**
```python
# Override the entire execution environment
custom_runner = Runner(
    model=CustomModel(),
    user_interface=CustomUI()
)
subroutine = SubroutineTemplate(
    template=special_template,
    runner=custom_runner  # Only available in Subroutine
)
```

3. **Default Isolation (Subroutine Only)**
```python
# Environment isolation happens automatically in Subroutine
subroutine = SubroutineTemplate(template=some_template)
# A copy of parent runner will be created automatically
```

## Benefits

1. **Cost Efficiency**
   - Fine-grained control over model usage
   - Use expensive models only where necessary

2. **Performance Optimization**
   - Balance between speed and quality
   - Use appropriate models for different requirements

3. **Environment Isolation** (Subroutine-specific)
   - Complete isolation of execution environment
   - Safe modification of runner settings
   - No side effects on parent environment

## Future Considerations

### ControlTemplate Extension

While it would be technically possible to add model override capabilities to ControlTemplates:
1. ControlTemplates don't generate content directly
2. They would only pass the model to child templates
3. This could be beneficial in some cases (e.g., setting model for a group of templates)
4. However, it would increase system complexity

Therefore, we decide not to implement this feature for ControlTemplates at this time.

## Considerations

### Positive Impact
- Better cost control
- Improved performance optimization
- More flexible LLM utilization
- Complete environment isolation (Subroutine)

### Challenges
- Need for careful model selection
- Potential complexity in configuration
- Resource management considerations
- Memory overhead from runner copying (Subroutine)

This design enables flexible and efficient LLM utilization while maintaining clean and maintainable code structure. The ability to override models at the template level provides powerful optimization capabilities for both cost and performance, with special consideration for Subroutine's unique isolation requirements.