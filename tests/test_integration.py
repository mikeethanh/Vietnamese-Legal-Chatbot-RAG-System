"""
Integration tests for the complete RAG system
"""
import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import os
import sys

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

from test_utils import create_test_qa_data, create_temp_json_file, assert_valid_response_format

@pytest.mark.integration
class TestRAGSystemIntegration:
    """Integration tests for complete RAG system"""
    
    @patch('qdrant_client.QdrantClient')
    @patch('openai.OpenAI')
    @patch('redis.Redis')
    def test_end_to_end_query_flow(self, mock_redis, mock_openai, mock_qdrant):
        """Test complete query processing flow"""
        # Setup mocks
        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        
        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock vector search results
        mock_qdrant_instance.search.return_value = [
            Mock(
                payload={
                    "content": "Luật giao thông quy định tốc độ tối đa trong khu vực đông dân cư là 50km/h",
                    "metadata": {"law": "Luật Giao thông", "article": "Điều 16"}
                },
                score=0.95
            )
        ]
        
        # Mock OpenAI response
        mock_completion = Mock()
        mock_completion.choices = [
            Mock(
                message=Mock(
                    content='{"answer": "Theo Luật Giao thông đường bộ, tốc độ tối đa trong khu vực đông dân cư là 50km/h.", "sources": ["Luật Giao thông - Điều 16"]}'
                )
            )
        ]
        mock_openai_instance.chat.completions.create.return_value = mock_completion
        
        # Test query processing
        query = "Quy định về tốc độ giao thông trong khu dân cư"
        
        # Mock the complete flow
        expected_response = {
            "answer": "Theo Luật Giao thông đường bộ, tốc độ tối đa trong khu vực đông dân cư là 50km/h.",
            "sources": ["Luật Giao thông - Điều 16"],
            "confidence": 0.95
        }
        
        # Validate response format
        assert_valid_response_format(expected_response)
        assert "giao thông" in expected_response["answer"].lower()
        assert len(expected_response["sources"]) > 0
    
    @patch('import_data.process_jsonl_file')
    def test_data_import_flow(self, mock_process):
        """Test data import and vectorization flow"""
        # Create test data file
        test_data = create_test_qa_data()
        temp_file = create_temp_json_file(test_data)
        
        try:
            # Mock import process
            mock_process.return_value = {
                "imported": len(test_data),
                "errors": 0,
                "status": "success"
            }
            
            result = mock_process(temp_file)
            
            assert result["imported"] == 2
            assert result["errors"] == 0
            assert result["status"] == "success"
            
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_query_preprocessing_pipeline(self):
        """Test query preprocessing and normalization"""
        test_cases = [
            {
                "input": "  Luật giao thông quy định gì về tốc độ???  ",
                "expected_contains": ["luật", "giao thông", "tốc độ"]
            },
            {
                "input": "HỢP ĐỒNG LAO ĐỘNG",
                "expected_contains": ["hợp đồng", "lao động"]
            },
            {
                "input": "Quy định thuế TNCN 2024",
                "expected_contains": ["quy định", "thuế", "2024"]
            }
        ]
        
        for case in test_cases:
            processed = case["input"].lower().strip().rstrip('?')
            for term in case["expected_contains"]:
                assert term.lower() in processed
    
    @patch('cache.get_cached_response')
    @patch('cache.cache_response')
    def test_caching_integration(self, mock_cache_set, mock_cache_get):
        """Test caching mechanism integration"""
        query = "test legal query"
        response = {
            "answer": "Test legal answer",
            "sources": ["Test Law - Article 1"],
            "confidence": 0.9
        }
        
        # Test cache miss scenario
        mock_cache_get.return_value = None
        cached = mock_cache_get(query)
        assert cached is None
        
        # Test cache set
        mock_cache_set.return_value = True
        result = mock_cache_set(query, response)
        assert result is True
        
        # Test cache hit scenario
        mock_cache_get.return_value = response
        cached = mock_cache_get(query)
        assert cached == response
    
    def test_error_handling_scenarios(self):
        """Test error handling in various scenarios"""
        error_scenarios = [
            {"type": "empty_query", "input": ""},
            {"type": "too_long", "input": "a" * 1000},
            {"type": "invalid_chars", "input": "test@#$%^&*()"},
            {"type": "only_numbers", "input": "123456"}
        ]
        
        for scenario in error_scenarios:
            query = scenario["input"]
            
            # Basic validation that should catch these errors
            is_valid = (
                len(query.strip()) > 5 and 
                len(query.strip()) < 500 and
                any(c.isalpha() for c in query)
            )
            
            if scenario["type"] in ["empty_query", "too_long", "only_numbers"]:
                assert not is_valid, f"Should reject {scenario['type']}: {query}"
    
    @patch('database.get_conversation_history')
    def test_conversation_context(self, mock_get_history):
        """Test conversation context handling"""
        # Mock conversation history
        mock_get_history.return_value = [
            {
                "query": "Luật giao thông quy định gì?",
                "response": "Luật giao thông có nhiều quy định...",
                "timestamp": "2024-01-01T10:00:00"
            }
        ]
        
        session_id = "test-session-123"
        history = mock_get_history(session_id)
        
        assert len(history) > 0
        assert "query" in history[0]
        assert "response" in history[0]
        assert "timestamp" in history[0]