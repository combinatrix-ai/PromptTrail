import pytest
from unittest.mock import MagicMock

from prompttrail.core import Message, Session
from prompttrail.core.errors import ParameterValidationError
from prompttrail.models.transformers import (
    TransformersModel,
    TransformersModelConfiguration,
    TransformersModelParameters,
)

@pytest.fixture
def mock_model():
    # Set up mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # Model configuration
    config = TransformersModelConfiguration(device="cpu")
    
    # Create TransformersModel instance and inject mocks
    model = TransformersModel(config, mock_model, mock_tokenizer)
    
    return model

def test_send(mock_model):
    # Test session and parameters
    session = Session(messages=[
        Message(content="Hello", sender="user")
    ])
    params = TransformersModelParameters(max_tokens=10)
    
    # Set up mocks
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_model.tokenizer.return_tensors = "pt"
    mock_model.tokenizer.return_value = MagicMock()
    mock_model.tokenizer.return_value.to.return_value = mock_tensor
    mock_model.model.generate.return_value = mock_tensor
    mock_model.tokenizer.decode.return_value = "Mock response"
    
    # Execute method
    response = mock_model.send(parameters=params, session=session)
    
    # Assertions
    assert response.content == "Mock response"
    mock_model.tokenizer.assert_called_once_with("user: Hello", return_tensors="pt")
    mock_model.model.generate.assert_called_once()

def test_send_async(mock_model):
    # Test session and parameters
    session = Session(messages=[
        Message(content="Hello", sender="user")
    ])
    params = TransformersModelParameters(max_tokens=10)
    
    # Set up mocks
    mock_model._create_streamer = MagicMock()
    mock_model._streamer_messages = [Message(content="Mock stream", sender="assistant")]
    
    # Execute method
    responses = list(mock_model.send_async(parameters=params, session=session))
    
    # Assertions
    assert len(responses) == 1
    assert responses[0].content == "Mock stream"
    mock_model._create_streamer.assert_called_once_with("new")

def test_validate_session(mock_model):
    # Valid session
    valid_session = Session(messages=[
        Message(content="Valid", sender="user")
    ])
    mock_model.validate_session(valid_session, is_async=False)
    
    # Invalid session (no messages)
    with pytest.raises(ParameterValidationError):
        empty_session = Session(messages=[])
        mock_model.validate_session(empty_session, is_async=False)
    
    # Invalid session (no sender)
    with pytest.raises(ParameterValidationError):
        invalid_session = Session(messages=[
            Message(content="Invalid", sender=None)
        ])
        mock_model.validate_session(invalid_session, is_async=False)

def test_small_llm_on_cpu():
    """Test using a small LLM (sshleifer/tiny-gpt2) running on CPU"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load model and tokenizer (low memory settings)
    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    
    # Model configuration
    config = TransformersModelConfiguration(device="cpu")
    transformers_model = TransformersModel(config, model, tokenizer)
    
    # Test session and parameters
    session = Session(messages=[
        Message(content="Hello", sender="user")
    ])
    params = TransformersModelParameters(max_tokens=5)  # Reduce token count
    
    # Execute method
    response = transformers_model.send(parameters=params, session=session)
    
    # Assertions
    assert isinstance(response.content, str)
    assert len(response.content) > 0