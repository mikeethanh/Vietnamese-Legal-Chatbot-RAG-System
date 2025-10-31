"""
Simple test to verify testing infrastructure works
"""
import pytest
import json
import os
import sys

def test_basic_functionality():
    """Test basic Python functionality"""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"

def test_json_operations():
    """Test JSON operations"""
    data = {"key": "value", "number": 123}
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    assert parsed == data

def test_file_operations():
    """Test file operations"""
    import tempfile
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_path = f.name
    
    try:
        # Read file
        with open(temp_path, 'r') as f:
            content = f.read()
        assert content == "test content"
    finally:
        # Cleanup
        os.unlink(temp_path)

def test_environment_variables():
    """Test environment variable handling"""
    # Set test environment variable
    os.environ['TEST_VAR'] = 'test_value'
    
    # Read it back
    value = os.environ.get('TEST_VAR')
    assert value == 'test_value'
    
    # Cleanup
    del os.environ['TEST_VAR']

def test_string_operations():
    """Test string operations"""
    text = "Vietnamese Legal Chatbot"
    
    assert "Vietnamese" in text
    assert text.lower() == "vietnamese legal chatbot"
    assert text.replace("Vietnamese", "English") == "English Legal Chatbot"

@pytest.mark.parametrize("input_val,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("test", "TEST"),
])
def test_parametrized_string_upper(input_val, expected):
    """Test parametrized string operations"""
    assert input_val.upper() == expected

def test_list_operations():
    """Test list operations"""
    data = [1, 2, 3, 4, 5]
    
    assert len(data) == 5
    assert sum(data) == 15
    assert max(data) == 5
    assert min(data) == 1

def test_dictionary_operations():
    """Test dictionary operations"""
    data = {
        "name": "Legal Chatbot",
        "version": "1.0",
        "features": ["RAG", "Q&A", "Legal Search"]
    }
    
    assert data["name"] == "Legal Chatbot"
    assert len(data["features"]) == 3
    assert "RAG" in data["features"]