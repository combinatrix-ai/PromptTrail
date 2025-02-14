# TypeScript Implementation Proposal for PromptTrail

## Overview

This document outlines a plan to implement PromptTrail in TypeScript, creating a developer-friendly library that provides a unified LLM interface and DSL for agents. The implementation will take inspiration from both the Python version and LangChain.js while leveraging TypeScript's type system for enhanced developer experience.

## Core Architecture

### 1. Core Module (`@prompttrail/core`)

#### Base Types and Interfaces

```typescript
// Core message types
type MessageRoleType = 'system' | 'user' | 'assistant' | 'tool_result' | 'control';

interface Message {
  content: string;
  role: MessageRoleType;
  toolUse?: Record<string, any>;
  metadata: Metadata;
}

// Enhanced type-safe metadata with generics
class Metadata<T extends Record<string, any> = Record<string, any>> extends Map<keyof T, T[keyof T]> {
  constructor(data?: T) {
    super();
    if (data) {
      Object.entries(data).forEach(([key, value]) => {
        this.set(key as keyof T, value);
      });
    }
  }

  toJSON(): T {
    return Object.fromEntries(this) as T;
  }
}

// Configuration with strong typing
interface ModelConfig {
  modelName: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  tools?: Tool[];
  cacheProvider?: CacheProvider;
  mockProvider?: MockProvider;
}

// Base Model interface
interface Model<TConfig extends ModelConfig = ModelConfig> {
  configuration: TConfig;
  send(session: Session): Promise<Message>;
  sendStream(session: Session): AsyncGenerator<Message>;
  formatTool?(tool: Tool): Record<string, any>;
  formatToolResult?(result: ToolResult): Record<string, any>;
}
```

### 2. Agent Module (`@prompttrail/agent`)

#### Template System

```typescript
// Base Template interface
interface Template {
  templateId: string;
  beforeTransform?: SessionTransformer[];
  afterTransform?: SessionTransformer[];
  render(session: Session): AsyncGenerator<Message | Event>;
}

// Template implementations
class MessageTemplate implements Template {
  constructor(
    private content: string,
    private role: MessageRoleType,
    public templateId: string = uuid(),
    private options: {
      beforeTransform?: SessionTransformer[];
      afterTransform?: SessionTransformer[];
      disableJinja?: boolean;
    } = {}
  ) {}

  async *render(session: Session): AsyncGenerator<Message | Event> {
    // Implementation
  }
}

// Specialized templates
class SystemTemplate extends MessageTemplate {
  constructor(content: string, options?: MessageTemplateOptions) {
    super(content, 'system', options);
  }
}

class UserTemplate extends MessageTemplate {
  constructor(
    content?: string,
    description?: string,
    options?: MessageTemplateOptions & {
      default?: string;
    }
  ) {
    super(content || '', 'user', options);
  }
}
```

### 3. Tools Module (`@prompttrail/tools`)

```typescript
interface Tool {
  name: string;
  description: string;
  parameters: JSONSchema7;
  execute(args: Record<string, any>): Promise<ToolResult>;
}

interface ToolResult {
  content: string;
  metadata?: Record<string, any>;
}

// Built-in tools
class ExecuteCommand implements Tool {
  name = 'execute_command';
  description = 'Execute a shell command';
  parameters = {
    type: 'object',
    properties: {
      command: { type: 'string' }
    },
    required: ['command']
  };

  async execute(args: { command: string }): Promise<ToolResult> {
    // Implementation
  }
}
```

## Implementation Strategy

1. **Package Structure**
```
prompttrail-ts/
├── packages/
│   ├── core/           # Core functionality
│   ├── agent/          # Agent and template system
│   ├── tools/          # Built-in tools
│   └── models/         # Model implementations
├── examples/           # Usage examples
└── tests/             # Test suite
```

2. **Build System**
- Use `pnpm` for package management (better monorepo support)
- TypeScript configuration with strict mode enabled
- ESM modules for better tree-shaking
- Jest for testing
- ESLint + Prettier for code quality

3. **Development Phases**

Phase 1: Core Implementation
- Implement core types and interfaces
- Basic message and session handling
- Model interface and base implementation

Phase 2: Template System
- Implement template base classes
- Message template system with Jinja-like functionality
- Control flow templates

Phase 3: Tools & Integration
- Implement tool interface
- Built-in tools
- Model integrations (OpenAI, Anthropic, etc.)

Phase 4: Testing & Documentation
- Comprehensive test suite
- API documentation
- Usage examples
- Migration guide from Python version

## Key Differences from Python Version

1. **Type System**
- Leverage TypeScript's type system for better IDE support
- Generic types for metadata and configurations
- Strict typing for template parameters

2. **Async Handling**
- Use native Promise and AsyncGenerator
- Better streaming support
- Improved error handling

3. **Modern JavaScript Features**
- ESM modules
- Class fields
- Optional chaining
- Nullish coalescing

4. **Developer Experience**
- Better IDE integration
- Type inference
- Autocomplete support
- Runtime type checking

## Example Usage

```typescript
import { CommandLineRunner, LinearTemplate, SystemTemplate } from '@prompttrail/agent';
import { AnthropicModel } from '@prompttrail/models';
import { ExecuteCommand, ReadFile } from '@prompttrail/tools';

// Configure model
const model = new AnthropicModel({
  apiKey: process.env.ANTHROPIC_API_KEY,
  modelName: 'claude-3-sonnet',
  maxTokens: 4096,
  tools: [new ExecuteCommand(), new ReadFile()]
});

// Create template
const template = new LinearTemplate([
  new SystemTemplate('You are a helpful assistant'),
  new UserTemplate('Input: '),
  new AssistantTemplate()
]);

// Create runner
const runner = new CommandLineRunner({
  model,
  template,
  metadata: {
    rules: await readFile('.clinerules')
  }
});

// Run agent
await runner.run();
```

## Next Steps

1. Set up monorepo structure with pnpm
2. Implement core types and interfaces
3. Create basic template system
4. Add model implementations starting with OpenAI
5. Implement built-in tools
6. Add tests and documentation

## Implementation Decisions

1. **API Compatibility**
   - Maintain exact API compatibility with the Python version
   - Keep method names, parameters, and behavior consistent
   - Use TypeScript types to enhance the developer experience without changing the API surface
   - Example:
     ```typescript
     // Python: session.append(message)
     // TypeScript: Same method name and behavior
     class Session {
       append(message: Message): void {
         this.messages.push(message);
       }
     }
     ```

2. **Template Rendering**
   - Use [Nunjucks](https://mozilla.github.io/nunjucks/) as the Jinja2 alternative
   - Nunjucks is a rich and mature templating engine that's very similar to Jinja2
   - Maintains similar syntax and features:
     ```typescript
     import nunjucks from 'nunjucks';

     class MessageTemplate implements Template {
       private env: nunjucks.Environment;

       constructor() {
         this.env = nunjucks.configure({ autoescape: true });
       }

       protected renderTemplate(content: string, context: Record<string, any>): string {
         return this.env.renderString(content, context);
       }
     }
     ```

3. **Class-based Architecture**
   - Continue using class-based approach to maintain consistency with Python version
   - Leverage TypeScript's OOP features for better type safety and inheritance
   - Benefits:
     - Direct mapping to Python implementation
     - Clear inheritance hierarchies
     - Easier state management
     - Better encapsulation
     - More intuitive for developers coming from Python version

4. **Monorepo Structure**
   ```
   prompttrail-ts/
   ├── packages/
   │   ├── core/              # Core functionality
   │   │   ├── src/
   │   │   │   ├── message.ts
   │   │   │   ├── session.ts
   │   │   │   └── model.ts
   │   │   └── package.json
   │   ├── agent/             # Agent system
   │   │   ├── src/
   │   │   │   ├── templates/
   │   │   │   └── runners/
   │   │   └── package.json
   │   ├── tools/             # Built-in tools
   │   │   ├── src/
   │   │   │   ├── execute.ts
   │   │   │   └── file.ts
   │   │   └── package.json
   │   └── models/            # Model implementations
   │       ├── src/
   │       │   ├── anthropic.ts
   │       │   └── openai.ts
   │       └── package.json
   ├── examples/              # Usage examples
   │   ├── basic/
   │   └── advanced/
   ├── python/                # Python source (for reference)
   │   └── src/
   ├── pnpm-workspace.yaml
   └── package.json
   ```

## Next Steps

1. Set up monorepo with pnpm
   ```bash
   pnpm init
   pnpm add -D typescript @types/node
   ```

2. Create core package structure
   ```bash
   cd packages
   pnpm create @prompttrail/core
   pnpm create @prompttrail/agent
   pnpm create @prompttrail/tools
   pnpm create @prompttrail/models
   ```

3. Implement core functionality with exact API compatibility
4. Add template system with Nunjucks integration
5. Implement model integrations
6. Add comprehensive tests matching Python version

If you approve of these decisions, we can switch to Code mode to begin implementation.