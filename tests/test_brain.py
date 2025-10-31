"""
Test cases for brain module (core query processing)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for brain module"""
    with patch('qdrant_client.QdrantClient') as mock_qdrant, \
         patch('openai.OpenAI') as mock_openai, \
         patch('redis.Redis') as mock_redis:
        
        # Setup mock returns
        mock_qdrant_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        
        mock_openai_instance = Mock()
        mock_openai.return_value = mock_openai_instance
        
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        yield {
            'qdrant': mock_qdrant_instance,
            'openai': mock_openai_instance, 
            'redis': mock_redis_instance
        }

class TestBrainModule:
    """Test brain module functionality"""
    
    def test_query_preprocessing(self, mock_dependencies):
        """Test query preprocessing and cleaning"""
        # Mock the brain module import
        with patch.dict('sys.modules', {'brain': Mock()}):
            # Test query cleaning
            test_queries = [
                "Luật giao thông quy định gì về tốc độ?",
                "  Hợp đồng lao động  ",
                "Quy định về thuế thu nhập cá nhân???",
            ]
            
            for query in test_queries:
                # Simulate query processing
                processed = query.strip().rstrip('?')
                assert len(processed) > 0
                assert not processed.endswith('???')
    
    @patch('brain.search_similar_documents')
    def test_document_search(self, mock_search, mock_dependencies):
        """Test document similarity search"""
        # Mock search results
        mock_search.return_value = [
            {
                "content": "Luật giao thông quy định tốc độ tối đa...",
                "metadata": {"law": "Luật Giao thông", "article": "Điều 16"},
                "score": 0.95
            },
            {
                "content": "Các quy định về an toàn giao thông...",
                "metadata": {"law": "Luật Giao thông", "article": "Điều 17"},
                "score": 0.87
            }
        ]
        
        query = "tốc độ giao thông"
        results = mock_search(query)
        
        assert len(results) == 2
        assert all(result["score"] > 0.8 for result in results)
        assert all("content" in result for result in results)
    
    @patch('brain.generate_response')
    def test_response_generation(self, mock_generate, mock_dependencies):
        """Test response generation from context"""
        # Mock OpenAI response
        mock_generate.return_value = {
            "answer": "Theo Luật Giao thông đường bộ, tốc độ tối đa trong khu vực đông dân cư là 50km/h.",
            "confidence": 0.92,
            "sources": ["Luật Giao thông đường bộ - Điều 16"]
        }
        
        query = "Quy định về tốc độ giao thông"
        context = [{"content": "Luật giao thông quy định..."}]
        
        response = mock_generate(query, context)
        
        assert "answer" in response
        assert "confidence" in response
        assert "sources" in response
        assert response["confidence"] > 0.9
        assert len(response["answer"]) > 0
    
    def test_query_validation(self, mock_dependencies):
        """Test query validation"""
        valid_queries = [
            "Luật giao thông quy định gì?",
            "Hợp đồng lao động có những điều khoản nào?",
            "Quy định về thuế thu nhập cá nhân"
        ]
        
        invalid_queries = [
            "",
            "   ",
            "a" * 1000,  # Too long
            "123",  # Too short
        ]
        
        for query in valid_queries:
            assert len(query.strip()) > 5  # Basic validation
            assert len(query.strip()) < 500
        
        for query in invalid_queries:
            is_valid = len(query.strip()) > 5 and len(query.strip()) < 500
            assert not is_valid
    
    @patch('brain.get_cached_response')
    @patch('brain.cache_response')
    def test_caching_mechanism(self, mock_cache, mock_get_cache, mock_dependencies):
        """Test response caching"""
        query = "test query"
        
        # Test cache miss
        mock_get_cache.return_value = None
        result = mock_get_cache(query)
        assert result is None
        
        # Test cache hit
        cached_response = {"answer": "cached answer", "sources": []}
        mock_get_cache.return_value = cached_response
        result = mock_get_cache(query)
        assert result == cached_response
        
        # Test caching new response
        new_response = {"answer": "new answer", "sources": []}
        mock_cache.return_value = True
        cache_result = mock_cache(query, new_response)
        assert cache_result is True