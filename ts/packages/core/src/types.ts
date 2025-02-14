/**
 * Core type definitions for PromptTrail
 */
import type { Metadata } from './metadata';

/**
 * Represents the role of a message in a conversation
 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool_result' | 'control';

/**
 * Discriminated union type for different message types
 */
export type Message = 
  | SystemMessage
  | UserMessage
  | AssistantMessage
  | ToolResultMessage
  | ControlMessage;

/**
 * Base interface for message properties
 */
interface BaseMessage {
  content: string;
  metadata?: Record<string, unknown>;
}

export interface SystemMessage extends BaseMessage {
  type: 'system';
}

export interface UserMessage extends BaseMessage {
  type: 'user';
}

export interface AssistantMessage extends BaseMessage {
  type: 'assistant';
}

export interface ToolResultMessage extends BaseMessage {
  type: 'tool_result';
  result: unknown;
}

export interface ControlMessage extends BaseMessage {
  type: 'control';
  control: {
    action: string;
    parameters?: Record<string, unknown>;
  };
}

/**
 * Type guard functions for message types
 */
export const isSystemMessage = (message: Message): message is SystemMessage =>
  message.type === 'system';

export const isUserMessage = (message: Message): message is UserMessage =>
  message.type === 'user';

export const isAssistantMessage = (message: Message): message is AssistantMessage =>
  message.type === 'assistant';

export const isToolResultMessage = (message: Message): message is ToolResultMessage =>
  message.type === 'tool_result';

export const isControlMessage = (message: Message): message is ControlMessage =>
  message.type === 'control';

/**
 * Branded type for temperature to ensure type safety
 */
export type Temperature = number & { readonly __brand: unique symbol };

export const createTemperature = (value: number): Temperature => {
  if (value < 0 || value > 2) {
    throw new Error('Temperature must be between 0 and 2');
  }
  return value as Temperature;
};

/**
 * Model configuration interface
 */
export interface ModelConfig {
  readonly modelName: string;
  readonly temperature: Temperature;
  readonly maxTokens?: number;
  readonly topP?: number;
  readonly topK?: number;
  readonly repetitionPenalty?: number;
  readonly tools?: readonly Tool[];
}

/**
 * Tool interface for function calling
 */
export interface Tool<TInput = unknown, TOutput = unknown> {
  readonly name: string;
  readonly description: string;
  readonly schema: unknown; // Will be replaced with actual schema type
  execute(input: TInput): Promise<TOutput>;
}

/**
 * Session interface for maintaining conversation state
 */
export interface Session<T extends Record<string, unknown> = Record<string, unknown>> {
  readonly messages: readonly Message[];
  readonly metadata: Metadata<T>;
}

/**
 * Error types
 */
export class PromptTrailError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = 'PromptTrailError';
  }
}

export class ValidationError extends PromptTrailError {
  constructor(message: string) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
  }
}

export class ConfigurationError extends PromptTrailError {
  constructor(message: string) {
    super(message, 'CONFIGURATION_ERROR');
    this.name = 'ConfigurationError';
  }
}