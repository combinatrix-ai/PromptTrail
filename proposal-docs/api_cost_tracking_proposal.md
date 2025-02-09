# API Cost Tracking Proposal

## Overview
This proposal outlines an approach to track API costs from model responses and store them in message/session metadata. This will help users monitor and manage their API usage costs across different models.

## Core Components

### 1. Response Usage Types

```python
class OpenAITokenDetails(TypedDict):
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int

class OpenAIUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[OpenAITokenDetails]

class AnthropicUsage(TypedDict):
    input_tokens: int
    output_tokens: int
```

### 2. Model-Specific Pricing Configuration

Add to existing Config classes:

```python
class ModelPricing(TypedDict):
    input_price_per_1k: float     # Price per 1K input/prompt tokens
    output_price_per_1k: float    # Price per 1K output/completion tokens
    currency: str                 # Currency code (e.g., "USD")

class Config:
    # Add to existing config
    pricing: Optional[ModelPricing] = None
```

Example configurations:
```python
GPT4_PRICING = {
    "input_price_per_1k": 0.03,
    "output_price_per_1k": 0.06,
    "currency": "USD"
}

CLAUDE3_PRICING = {
    "input_price_per_1k": 0.015,
    "output_price_per_1k": 0.075,
    "currency": "USD"
}
```

### 3. Cost Tracking Implementation

#### 3.1 Model Class Changes

Extend the base Model class to handle cost tracking:

```python
class Model:
    def _calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.configuration.pricing:
            return None
        
        pricing = self.configuration.pricing
        input_cost = (input_tokens / 1000) * pricing["input_price_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_price_per_1k"]
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "currency": pricing["currency"],
            "model": model_name or self.configuration.model_name
        }

    def after_send(self, session: Session, message: Message) -> Message:
        # Extract usage data from response
        usage_data = message.metadata.get("_response_data", {}).get("usage")
        if not usage_data:
            return message

        # Calculate cost based on model type
        if isinstance(usage_data, OpenAIUsage):
            cost_info = self._calculate_cost(
                usage_data["prompt_tokens"],
                usage_data["completion_tokens"]
            )
        elif isinstance(usage_data, AnthropicUsage):
            cost_info = self._calculate_cost(
                usage_data["input_tokens"],
                usage_data["output_tokens"]
            )
            
        if cost_info:
            # Store cost info in message metadata
            message.metadata["usage"] = cost_info
            
            # Update session total cost
            session.metadata.setdefault("usage_stats", {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "currency": cost_info["currency"],
                "usage_by_model": {}
            })
            
            stats = session.metadata["usage_stats"]
            stats["total_input_tokens"] += cost_info["input_tokens"]
            stats["total_output_tokens"] += cost_info["output_tokens"]
            stats["total_cost"] += cost_info["total_cost"]
            
            # Track usage by model
            model_name = cost_info["model"]
            if model_name not in stats["usage_by_model"]:
                stats["usage_by_model"][model_name] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0
                }
            
            model_stats = stats["usage_by_model"][model_name]
            model_stats["total_input_tokens"] += cost_info["input_tokens"]
            model_stats["total_output_tokens"] += cost_info["output_tokens"]
            model_stats["total_cost"] += cost_info["total_cost"]
            
        return message
```

#### 3.2 Model-Specific Implementations

##### OpenAI Implementation
```python
class OpenAIModel(Model):
    def _send(self, session: Session) -> Message:
        response = openai.chat.completions.create(**create_params)
        message = Message(
            content=response.choices[0].message.content,
            role="assistant"
        )
        
        # Store complete response data for usage extraction
        message.metadata["_response_data"] = {
            "id": response.id,
            "model": response.model,
            "usage": response.usage
        }
            
        return message
```

##### Anthropic Implementation
```python
class AnthropicModel(Model):
    def _send(self, session: Session) -> Message:
        response = self.client.messages.create(**create_params)
        
        # Extract text content from blocks
        content = "".join(
            block.text for block in response.content 
            if hasattr(block, "text")
        )
        
        message = Message(content=content, role="assistant")
        
        # Store complete response data for usage extraction
        message.metadata["_response_data"] = {
            "id": response.id,
            "model": response.model,
            "usage": response.usage
        }
            
        return message
```

### 4. Usage Examples

```python
# Configure model with pricing
model = OpenAIModel(
    configuration=OpenAIConfig(
        api_key="...",
        model_name="gpt-4",
        pricing=GPT4_PRICING
    )
)

# Send message
session = Session()
session.append(Message(content="Hello", role="user"))
response = model.send(session)

# Access usage information for specific message
usage = response.metadata["usage"]
print(f"Message cost: ${usage['total_cost']:.4f}")
print(f"Input tokens: {usage['input_tokens']}")
print(f"Output tokens: {usage['output_tokens']}")

# Access session-wide statistics
stats = session.metadata["usage_stats"]
print(f"Total session cost: ${stats['total_cost']:.4f}")
print(f"Total input tokens: {stats['total_input_tokens']}")
print(f"Total output tokens: {stats['total_output_tokens']}")

# Access per-model statistics
for model_name, model_stats in stats["usage_by_model"].items():
    print(f"\nModel: {model_name}")
    print(f"Total cost: ${model_stats['total_cost']:.4f}")
    print(f"Total tokens: {model_stats['total_input_tokens'] + model_stats['total_output_tokens']}")
```

## Benefits

1. **Direct Integration**: Uses actual response formats from OpenAI and Anthropic
2. **Detailed Tracking**: Captures both message-level and session-level statistics
3. **Model-Specific Stats**: Tracks usage separately for each model
4. **Flexible Pricing**: Supports different input/output token rates
5. **Transparent Storage**: All data available in message and session metadata

## Implementation Steps

1. Add usage type definitions and pricing configuration
2. Update Model class with cost calculation and tracking
3. Modify OpenAI and Anthropic implementations to store response data
4. Add tests for different response formats and pricing scenarios
5. Update documentation with usage examples
6. Add cost tracking to CLI interface (optional)

## Future Enhancements

1. Budget limits and alerts
2. Cost optimization suggestions
3. Usage reporting and visualization
4. Support for more complex pricing models
5. Token usage estimation before sending requests
6. Cost comparison between different models
7. Integration with billing systems
8. Usage analytics and trends
9. Handle Anthropic's explicit cache writing operation
   - Anthropic requires explicit cache writing operations
   - Need to implement cache handling that preserves token usage information
   - Consider impact on cost tracking when responses are cached
   - Ensure cached responses maintain accurate usage statistics