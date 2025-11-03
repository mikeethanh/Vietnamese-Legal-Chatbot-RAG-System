"""
Test cases for backend utilities - testing actual backend functions
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

try:
    from utils import generate_random_string, generate_request_id, ColoredFormatter
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    pytestmark = pytest.mark.skip("backend utils module not available")

@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Backend utils not available")
class TestBackendUtils:
    """Test actual backend utility functions"""
    
    def test_generate_random_string_default_length(self):
        """Test random string generation with default length"""
        result = generate_random_string()
        assert isinstance(result, str)
        assert len(result) == 16  # Default length
    
    def test_generate_random_string_custom_length(self):
        """Test random string generation with custom length"""
        length = 32
        result = generate_random_string(length)
        assert isinstance(result, str)
        assert len(result) == length
    
    def test_generate_random_string_uniqueness(self):
        """Test that random strings are unique"""
        result1 = generate_random_string()
        result2 = generate_random_string()
        assert result1 != result2
    
    def test_generate_request_id_default(self):
        """Test request ID generation with default max length"""
        result = generate_request_id()
        assert isinstance(result, str)
        assert len(result) <= 33  # max_length + 1
    
    def test_generate_request_id_custom_length(self):
        """Test request ID generation with custom max length"""
        max_length = 16
        result = generate_request_id(max_length)
        assert isinstance(result, str)
        assert len(result) <= max_length + 1
    
    def test_generate_request_id_uniqueness(self):
        """Test that request IDs are unique"""
        result1 = generate_request_id()
        result2 = generate_request_id()
        assert result1 != result2
    
    def test_colored_formatter_creation(self):
        """Test ColoredFormatter can be created"""
        formatter = ColoredFormatter()
        assert formatter is not None
        assert hasattr(formatter, 'COLORS')
        assert hasattr(formatter, 'EMOJI')
    
    def test_colored_formatter_colors(self):
        """Test ColoredFormatter has expected colors"""
        formatter = ColoredFormatter()
        expected_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in expected_levels:
            assert level in formatter.COLORS
            assert level in formatter.EMOJI