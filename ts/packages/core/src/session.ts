import type { Message, Session } from './types';
import { Metadata } from './metadata';
import { ValidationError } from './types';

/**
 * Immutable session implementation
 */
export class SessionImpl<T extends Record<string, unknown> = Record<string, unknown>> implements Session<T> {
  constructor(
    public readonly messages: readonly Message[] = [],
    public readonly metadata: Metadata<T> = new Metadata<T>()
  ) {}

  /**
   * Create a new session with additional message
   */
  addMessage(message: Message): SessionImpl<T> {
    return new SessionImpl<T>(
      [...this.messages, message],
      this.metadata.clone()
    );
  }

  /**
   * Create a new session with updated metadata
   */
  updateMetadata<U extends Record<string, unknown>>(metadata: U): SessionImpl<T & U> {
    return new SessionImpl<T & U>(
      this.messages,
      this.metadata.merge(metadata)
    );
  }

  /**
   * Get the last message in the session
   */
  getLastMessage(): Message | undefined {
    return this.messages[this.messages.length - 1];
  }

  /**
   * Get all messages of a specific type
   */
  getMessagesByType<U extends Message['type']>(type: U): Extract<Message, { type: U }>[] {
    return this.messages.filter((msg): msg is Extract<Message, { type: U }> => 
      msg.type === type
    );
  }

  /**
   * Validate session state
   */
  validate(): void {
    // Check for empty session
    if (this.messages.length === 0) {
      throw new ValidationError('Session must have at least one message');
    }

    // Check for empty messages
    if (this.messages.some(msg => !msg.content)) {
      throw new ValidationError('Empty messages are not allowed');
    }

    // Check system message position
    const systemMessages = this.getMessagesByType('system');
    if (systemMessages.length > 1) {
      throw new ValidationError('Only one system message is allowed');
    }
    if (systemMessages.length === 1 && this.messages[0].type !== 'system') {
      throw new ValidationError('System message must be at the beginning');
    }
  }

  /**
   * Create a JSON representation of the session
   */
  toJSON(): Record<string, unknown> {
    return {
      messages: this.messages,
      metadata: this.metadata.toJSON()
    };
  }

  /**
   * Create a string representation of the session
   */
  toString(): string {
    return JSON.stringify(this.toJSON(), null, 2);
  }

  /**
   * Create a new session from a JSON representation
   */
  static fromJSON<U extends Record<string, unknown>>(json: Record<string, unknown>): SessionImpl<U> {
    if (!json.messages || !Array.isArray(json.messages)) {
      throw new ValidationError('Invalid session JSON: messages must be an array');
    }

    return new SessionImpl<U>(
      json.messages as Message[],
      new Metadata<U>(json.metadata as U)
    );
  }

  /**
   * Create an empty session
   */
  static empty<U extends Record<string, unknown>>(): SessionImpl<U> {
    return new SessionImpl<U>();
  }

  /**
   * Create a session with initial messages
   */
  static withMessages<U extends Record<string, unknown>>(messages: Message[]): SessionImpl<U> {
    return new SessionImpl<U>(messages);
  }

  /**
   * Create a session with initial metadata
   */
  static withMetadata<U extends Record<string, unknown>>(metadata: U): SessionImpl<U> {
    return new SessionImpl<U>([], new Metadata<U>(metadata));
  }

  /**
   * Create a session with both messages and metadata
   */
  static create<U extends Record<string, unknown>>(
    messages: Message[],
    metadata: U
  ): SessionImpl<U> {
    return new SessionImpl<U>(messages, new Metadata<U>(metadata));
  }
}

/**
 * Create a new session with type inference
 */
export function createSession<T extends Record<string, unknown>>(options: {
  messages?: Message[];
  metadata?: T;
} = {}): SessionImpl<T> {
  return new SessionImpl<T>(
    options.messages,
    options.metadata ? new Metadata<T>(options.metadata) : undefined
  );
}