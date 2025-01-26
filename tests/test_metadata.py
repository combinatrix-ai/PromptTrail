from prompttrail.core import Message, Metadata, Session


def test_metadata_basic_operations():
    """Test basic metadata operations"""
    metadata = Metadata()

    # Test setting and getting values
    metadata["key"] = "value"
    assert metadata["key"] == "value"

    # Test get with default
    assert metadata.get("nonexistent") is None
    assert metadata.get("nonexistent", "default") == "default"

    # Test update
    metadata.update({"new_key": "new_value"})
    assert metadata["new_key"] == "new_value"

    # Test copy
    copied = metadata.copy()
    assert copied["key"] == "value"
    assert copied["new_key"] == "new_value"

    # Verify copies are independent
    copied["another_key"] = "another_value"
    assert "another_key" not in metadata


def test_message_metadata():
    """Test metadata in Message class"""
    message = Message(content="test", role="user")

    # Test metadata operations
    message.metadata["timestamp"] = "2024-01-26"
    assert message.metadata["timestamp"] == "2024-01-26"

    # Test metadata update
    message.metadata.update({"source": "test"})
    assert message.metadata["source"] == "test"


def test_session_metadata():
    """Test metadata in Session class"""
    session = Session()

    # Test direct metadata operations
    session.metadata["user_id"] = "123"
    assert session.metadata["user_id"] == "123"

    # Test metadata update
    session.metadata.update({"language": "ja"})
    assert session.metadata["language"] == "ja"


def test_metadata_complex_values():
    """Test metadata with complex value types"""
    metadata = Metadata()

    # Test with different value types
    metadata["string"] = "text"
    metadata["number"] = 42
    metadata["list"] = [1, 2, 3]
    metadata["dict"] = {"key": "value"}
    metadata["boolean"] = True

    assert metadata["string"] == "text"
    assert metadata["number"] == 42
    assert metadata["list"] == [1, 2, 3]
    assert metadata["dict"] == {"key": "value"}
    assert metadata["boolean"] is True


def test_metadata_copy_independence():
    """Test that copied metadata is independent"""
    original = Metadata()
    original["key"] = "value"

    # Test copy
    copied = original.copy()
    copied["key"] = "new_value"

    assert original["key"] == "value"
    assert copied["key"] == "new_value"

    # Test model_copy
    model_copied = original.model_copy()
    model_copied["key"] = "another_value"

    assert original["key"] == "value"
    assert model_copied["key"] == "another_value"
