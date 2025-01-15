from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from prompttrail.core import Message, Session
from prompttrail.models.transformers import (
    TransformersConfig,
    TransformersModel,
    TransformersParam,
)

# Configuration for a small language model to run on CPU
config = TransformersConfig(device="cpu")

# Load the pre-trained model and tokenizer
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", low_cpu_mem_usage=True
)

# Initialize the TransformersModel with the configuration, model, and tokenizer
transformers_model = TransformersModel(
    configuration=config, model=model, tokenizer=tokenizer
)

# Create a new session
session = Session(messages=[Message(content="Hello", role="user")])

# Set parameters for text generation
params = TransformersParam(max_tokens=5)

# Send the message to the model and get the response
response = transformers_model.send(parameters=params, session=session)

# Print the assistant's response
print(f"Assistant: {response.content}")
