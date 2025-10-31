"""
Test cases for backend utilities
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

try:
    from utils import extract_text_content, validate_file_type, safe_json_loads
except ImportError:
    # Skip if utils module not available
    pytestmark = pytest.mark.skip("utils module not available")

class TestUtils:
    """Test utility functions"""
    
    def test_safe_json_loads_valid_json(self):
        """Test safe JSON loading with valid JSON"""
        valid_json = '{"key": "value", "number": 123}'
        result = safe_json_loads(valid_json)
        assert result == {"key": "value", "number": 123}
    
    def test_safe_json_loads_invalid_json(self):
        """Test safe JSON loading with invalid JSON"""
        invalid_json = '{"key": "value", invalid}'
        result = safe_json_loads(invalid_json, default={})
        assert result == {}
    
    def test_safe_json_loads_with_default(self):
        """Test safe JSON loading with custom default"""
        invalid_json = 'not json'
        default_value = {"error": True}
        result = safe_json_loads(invalid_json, default=default_value)
        assert result == default_value
    
    def test_validate_file_type_valid_extensions(self):
        """Test file type validation with valid extensions"""
        valid_files = [
            "document.pdf",
            "data.json", 
            "text.txt",
            "data.jsonl"
        ]
        
        for filename in valid_files:
            assert validate_file_type(filename) is True
    
    def test_validate_file_type_invalid_extensions(self):
        """Test file type validation with invalid extensions"""
        invalid_files = [
            "script.exe",
            "image.jpg",
            "video.mp4",
            "file.unknown"
        ]
        
        for filename in invalid_files:
            assert validate_file_type(filename) is False
    
    def test_extract_text_content_simple(self):
        """Test text content extraction"""
        sample_text = "This is a sample text with some content."
        result = extract_text_content(sample_text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "sample" in result.lower()
    
    def test_extract_text_content_empty(self):
        """Test text content extraction with empty input"""
        result = extract_text_content("")
        assert result == ""
    
    def test_extract_text_content_whitespace(self):
        """Test text content extraction with whitespace"""
        result = extract_text_content("   \n\t   ")
        assert result.strip() == ""