# Quickstart

Welcome to PromptTrail! Choose your path:

## Quick Navigation

1. **Just want to call LLM APIs?**
   - Start with [Preparation](quickstart-preparation.md) to set up your environment
   - Then read [Models](quickstart-models.md) to learn how to make API calls
   - For production:
     - Learn about [Caching](quickstart-core.md#cache) to optimize API usage
     - Explore [Tool/Function Calling](quickstart-agents.md#tool-function-calling) for advanced features

2. **Want to build complex conversational agents?**
   - Follow path 1 first to understand the basics
   - Then dive into [Agents](quickstart-agents.md) for building conversational flows

## Table of Contents

```{toctree}
quickstart-preparation.md
quickstart-models.md
quickstart-agents.md
quickstart-core.md
```

## Project Structure

PromptTrail is organized into several core modules:

- `prompttrail.models`: Unified interfaces to various LLM providers
- `prompttrail.agent`: Framework for building complex LLM applications
- `prompttrail.core`: Core utilities and developer tools

Each module has its own quickstart guide above.