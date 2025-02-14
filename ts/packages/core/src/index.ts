/**
 * Core functionality for PromptTrail
 */

// Export types
export type {
  Message,
  SystemMessage,
  UserMessage,
  AssistantMessage,
  ToolResultMessage,
  ControlMessage,
  Session,
  Tool,
  ModelConfig,
  Temperature
} from './types';

// Export type guards
export {
  isSystemMessage,
  isUserMessage,
  isAssistantMessage,
  isToolResultMessage,
  isControlMessage,
  createTemperature
} from './types';

// Export error classes
export {
  PromptTrailError,
  ValidationError,
  ConfigurationError
} from './types';

// Export metadata functionality
export {
  Metadata,
  createMetadata
} from './metadata';

// Export session functionality
export {
  SessionImpl,
  createSession
} from './session';