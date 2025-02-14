# Modern TypeScript Implementation Proposal for PromptTrail

## Overview

This document outlines an alternative, modern TypeScript-first implementation of PromptTrail that leverages contemporary TypeScript patterns and features. While the core functionality remains similar, the implementation takes advantage of TypeScript's type system and modern JavaScript features to create a more idiomatic TypeScript library.

## Core Design Principles

1. **Type-First Development**
   - Leverage TypeScript's powerful type system
   - Use discriminated unions and type guards
   - Implement strict type checking
   - Provide excellent type inference

2. **Functional Core, Imperative Shell**
   - Pure functions for core logic
   - Immutable data structures
   - Side effects at the edges
   - Composition over inheritance

3. **Modern JavaScript Features**
   - ESM modules
   - Async iterators
   - Decorators
   - Private class fields
   - Optional chaining and nullish coalescing

4. **Developer Experience**
   - Excellent TypeScript IntelliSense
   - Zero runtime overhead from types
   - Clear error messages
   - Comprehensive documentation

## Architecture

### 1. Core Types

```typescript
// Discriminated union for message types
type Message = 
  | { type: 'system'; content: string; metadata?: Record<string, unknown> }
  | { type: 'user'; content: string; metadata?: Record<string, unknown> }
  | { type: 'assistant'; content: string; metadata?: Record<string, unknown> }
  | { type: 'tool_result'; content: string; result: unknown; metadata?: Record<string, unknown> };

// Type guard example
const isSystemMessage = (message: Message): message is Extract<Message, { type: 'system' }> =>
  message.type === 'system';

// Immutable session state
interface Session {
  readonly messages: readonly Message[];
  readonly metadata: Readonly<Record<string, unknown>>;
}

// Configuration using branded types for type-safety
type Temperature = number & { readonly __brand: unique symbol };
const createTemperature = (value: number): Temperature => {
  if (value < 0 || value > 2) throw new Error('Temperature must be between 0 and 2');
  return value as Temperature;
};

interface ModelConfig {
  readonly modelName: string;
  readonly temperature: Temperature;
  readonly maxTokens?: number;
  readonly tools?: readonly Tool[];
}
```

### 2. Template System

```typescript
// Template definition using tagged template literals
const template = createTemplate`
  You are an AI assistant.
  Current time: ${(ctx) => ctx.currentTime}
  User request: ${(ctx) => ctx.userInput}
`;

// Composable template functions
const withSystemPrompt = (prompt: string) => 
  <T extends Record<string, unknown>>(template: Template<T>): Template<T> =>
    async (context: T) => {
      const systemMessage: Message = { type: 'system', content: prompt };
      const result = await template(context);
      return [systemMessage, ...result];
    };

// Higher-order template composition
const chatTemplate = pipe(
  withSystemPrompt('You are a helpful assistant'),
  withTemperature(0.7),
  withMaxTokens(1000)
)(baseTemplate);
```

### 3. Tools System

```typescript
// Tool definition using TypeScript type inference
interface Tool<TInput, TOutput> {
  readonly name: string;
  readonly description: string;
  readonly schema: ZodSchema<TInput>;
  execute(input: TInput): Promise<TOutput>;
}

// Tool factory with automatic type inference
const createTool = <TInput, TOutput>(config: {
  name: string;
  description: string;
  schema: ZodSchema<TInput>;
  execute: (input: TInput) => Promise<TOutput>;
}): Tool<TInput, TOutput> => config;

// Example tool with full type inference
const executeCommand = createTool({
  name: 'execute_command',
  description: 'Execute a shell command',
  schema: z.object({
    command: z.string(),
    cwd: z.string().optional(),
  }),
  async execute({ command, cwd }) {
    // Implementation
  },
});
```

### 4. Model Integration

```typescript
// Abstract model interface using generics
interface LLMModel<TConfig extends ModelConfig = ModelConfig> {
  readonly config: TConfig;
  generate(prompt: string): AsyncIterableIterator<string>;
  complete(messages: readonly Message[]): Promise<Message>;
}

// Functional approach to model creation
const createModel = <TConfig extends ModelConfig>(
  config: TConfig,
  implementation: ModelImplementation<TConfig>
): LLMModel<TConfig> => ({
  config,
  generate: implementation.generate,
  complete: implementation.complete,
});

// Example usage
const model = createModel({
  modelName: 'gpt-4',
  temperature: createTemperature(0.7),
  maxTokens: 1000,
}, openAIImplementation);
```

## Implementation Strategy

### 1. Package Structure

```
prompttrail-modern/
├── packages/
│   ├── core/                 # Core types and utilities
│   │   ├── src/
│   │   │   ├── types.ts     # Core type definitions
│   │   │   ├── session.ts   # Session management
│   │   │   └── utils.ts     # Utility functions
│   │   └── package.json
│   ├── template/            # Template system
│   │   ├── src/
│   │   │   ├── builder.ts   # Template builders
│   │   │   └── compiler.ts  # Template compilation
│   │   └── package.json
│   ├── tools/               # Tool system
│   │   ├── src/
│   │   │   ├── registry.ts  # Tool registry
│   │   │   └── builtin.ts   # Built-in tools
│   │   └── package.json
│   └── models/              # Model implementations
│       ├── src/
│       │   ├── openai.ts    # OpenAI integration
│       │   └── anthropic.ts # Anthropic integration
│       └── package.json
├── examples/
│   ├── chat-bot/
│   └── code-assistant/
└── package.json
```

### 2. Development Approach

1. **Type-Driven Development**
   - Define types first
   - Use TypeScript's strict mode
   - Leverage type inference
   - Document with TSDoc

2. **Testing Strategy**
   - Unit tests with Jest
   - Type testing with dtslint
   - Integration tests
   - Property-based testing

3. **Documentation**
   - TypeDoc for API docs
   - Storybook for examples
   - Interactive playground

## Example Usage

```typescript
import { createAgent, createModel, withTools } from '@prompttrail/core';
import { executeCommand, readFile } from '@prompttrail/tools';

// Create a model with type-safe configuration
const model = createModel({
  modelName: 'gpt-4',
  temperature: createTemperature(0.7),
  maxTokens: 1000,
});

// Create a template with type checking
const template = createTemplate`
  System: You are a helpful assistant.
  User: ${(ctx: { input: string }) => ctx.input}
  Assistant: Let me help you with that.
`;

// Compose agent with tools
const agent = pipe(
  withTools([executeCommand, readFile]),
  withMemory(createConversationMemory()),
  withErrorHandling()
)(createAgent(model, template));

// Use the agent
const response = await agent.run({ input: 'Hello!' });
```

## Key Features

1. **Type Safety**
   - Full type inference
   - No type assertions needed
   - Compile-time checks
   - Discriminated unions

2. **Functional Approach**
   - Immutable data structures
   - Pure functions
   - Function composition
   - Higher-order functions

3. **Modern Patterns**
   - Builder pattern with fluent interface
   - Dependency injection
   - Factory functions
   - Decorators for cross-cutting concerns

4. **Developer Experience**
   - Rich type information
   - Clear error messages
   - Autocomplete support
   - Documentation

## Next Steps

1. Set up monorepo with modern tools
   ```bash
   pnpm init
   pnpm add -D typescript @types/node vitest tsup
   ```

2. Create core package
   ```bash
   cd packages
   pnpm create @prompttrail/core
   ```

3. Implement core functionality
4. Add template system
5. Integrate with LLM providers
6. Add comprehensive tests

If you approve of this modern TypeScript approach, we can switch to Code mode to begin implementation.