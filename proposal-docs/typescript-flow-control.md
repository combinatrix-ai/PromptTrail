# Control Flow Implementation in Modern TypeScript

This document demonstrates how to implement complex control flow structures like the Python example in a TypeScript-idiomatic way.

## Template System Implementation

```typescript
// Base template interface with type-safe context
interface Template<TContext = Record<string, unknown>> {
  render(context: TContext): AsyncIterableIterator<Message>;
}

// Composable template builders
type TemplateBuilder<TContext> = (context: TContext) => AsyncIterableIterator<Message>;

// Template combinators for flow control
const sequence = <T>(...templates: Template<T>[]): Template<T> => ({
  async *render(context: T) {
    for (const template of templates) {
      yield* template.render(context);
    }
  }
});

const loop = <T>(
  template: Template<T>,
  exitCondition: (context: T) => boolean
): Template<T> => ({
  async *render(context: T) {
    while (!exitCondition(context)) {
      yield* template.render(context);
    }
  }
});

// Template implementations
class SystemTemplate implements Template<Record<string, unknown>> {
  constructor(
    private content: string,
    private options: { interpolate?: boolean } = {}
  ) {}

  async *render(context: Record<string, unknown>) {
    const rendered = this.options.interpolate
      ? interpolateTemplate(this.content, context)
      : this.content;

    yield {
      type: 'system',
      content: rendered
    };
  }
}

class UserTemplate implements Template {
  constructor(
    private description: string,
    private options: { default?: string } = {}
  ) {}

  async *render() {
    yield {
      type: 'user',
      content: await getUserInput(this.description, this.options.default)
    };
  }
}

class ToolingTemplate implements Template {
  constructor(private tools: Tool[]) {}

  async *render(context: Record<string, unknown>) {
    // Implement tool execution logic
    yield {
      type: 'tool_result',
      content: 'Tool execution result',
      result: {}
    };
  }
}
```

// Example Implementation

```typescript
// Modern TypeScript equivalent of the Python example
import { createModel, createRunner, createSession } from '@prompttrail/core';
import { sequence, loop, SystemTemplate, UserTemplate, ToolingTemplate } from '@prompttrail/templates';
import { ExecuteCommand, ReadFile, CreateFile, EditFile } from '@prompttrail/tools';

// Create model with type-safe configuration
const model = createModel({
  type: 'anthropic',
  config: {
    apiKey: process.env.ANTHROPIC_API_KEY,
    modelName: 'claude-3-sonnet',
    maxTokens: 4096,
    tools: [
      new ExecuteCommand(),
      new ReadFile(),
      new CreateFile(),
      new EditFile()
    ]
  }
});

// Create templates with type-safe context
const templates = sequence(
  // System prompt with interpolation
  new SystemTemplate(
    "You're a smart coding agent! Type END if you want to end conversation. Follow rules: {{clinerules}}",
    { interpolate: true }
  ),
  
  // Loop with exit condition
  loop(
    sequence(
      // User input
      new UserTemplate("Input: "),
      
      // Tool integration
      new ToolingTemplate([
        new ExecuteCommand(),
        new ReadFile(),
        new CreateFile(),
        new EditFile()
      ])
    ),
    // Exit condition as a pure function
    (session) => session.lastMessage?.content === "END"
  )
);

// Create session with type-safe metadata
const session = createSession({
  metadata: {
    clinerules: await readFile('.clinerules')
  }
});

// Create runner with dependency injection
const runner = createRunner({
  model,
  templates,
  interface: new CLIInterface()
});

// Run with async/await
await runner.run(session);
```

## Advanced Flow Control Features

### 1. Conditional Templates

```typescript
const conditional = <T>(
  condition: (context: T) => boolean,
  ifTemplate: Template<T>,
  elseTemplate?: Template<T>
): Template<T> => ({
  async *render(context: T) {
    if (condition(context)) {
      yield* ifTemplate.render(context);
    } else if (elseTemplate) {
      yield* elseTemplate.render(context);
    }
  }
});

// Usage
const template = conditional(
  (ctx) => ctx.isAdvancedUser,
  new SystemTemplate("Advanced mode enabled..."),
  new SystemTemplate("Basic mode enabled...")
);
```

### 2. Parallel Template Execution

```typescript
const parallel = <T>(...templates: Template<T>[]): Template<T> => ({
  async *render(context: T) {
    const results = await Promise.all(
      templates.map(t => collect(t.render(context)))
    );
    for (const messages of results) {
      yield* messages;
    }
  }
});
```

### 3. Error Handling

```typescript
const withErrorHandling = <T>(template: Template<T>): Template<T> => ({
  async *render(context: T) {
    try {
      yield* template.render(context);
    } catch (error) {
      yield {
        type: 'system',
        content: `Error: ${error.message}`
      };
    }
  }
});
```

### 4. State Management

```typescript
const withState = <T, S>(
  template: Template<T & { state: S }>,
  initialState: S
): Template<T> => ({
  async *render(context: T) {
    const contextWithState = {
      ...context,
      state: initialState
    };
    yield* template.render(contextWithState);
  }
});
```

### 5. Middleware Support

```typescript
type Middleware<T> = (
  template: Template<T>
) => Template<T>;

const createMiddleware = <T>(
  before?: (context: T) => Promise<void>,
  after?: (context: T) => Promise<void>
): Middleware<T> => 
  (template) => ({
    async *render(context: T) {
      if (before) await before(context);
      yield* template.render(context);
      if (after) await after(context);
    }
  });
```

## Benefits of This Approach

1. **Type Safety**
   - Full type inference for templates and context
   - Compile-time checks for template composition
   - Type-safe tool integration

2. **Composability**
   - Templates can be easily combined and nested
   - Pure functions for flow control
   - Middleware pattern for cross-cutting concerns

3. **Testability**
   - Pure functions are easy to test
   - Mock support built-in
   - Isolated template testing

4. **Extensibility**
   - Easy to add new template types
   - Pluggable tool system
   - Customizable flow control

This implementation provides the same capabilities as the Python version but with added type safety and modern TypeScript patterns. The template system is more flexible and composable while maintaining the intuitive DSL-like interface.