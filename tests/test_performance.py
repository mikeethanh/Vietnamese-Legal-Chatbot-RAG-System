"""
Performance tests for the legal chatbot system
"""
import pytest
import time
import asyncio
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

@pytest.mark.performance
class TestPerformance:
    """Performance test suite"""
    
    def test_query_processing_speed(self):
        """Test query processing time"""
        with patch('brain.process_query') as mock_process:
            mock_process.return_value = {
                "answer": "Test answer",
                "sources": ["Test source"],
                "confidence": 0.9
            }
            
            start_time = time.time()
            result = mock_process("Test query")
            end_time = time.time()
            
            processing_time = end_time - start_time
            assert processing_time < 2.0  # Should process in under 2 seconds
            assert result is not None
    
    def test_concurrent_queries(self):
        """Test handling multiple concurrent queries"""
        with patch('brain.process_query') as mock_process:
            mock_process.return_value = {
                "answer": "Concurrent test answer",
                "sources": ["Test source"],
                "confidence": 0.9
            }
            
            queries = [f"Test query {i}" for i in range(10)]
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(mock_process, query) for query in queries]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            total_time = end_time - start_time
            
            assert len(results) == 10
            assert total_time < 5.0  # Should handle 10 concurrent queries in under 5 seconds
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate processing multiple queries
        with patch('brain.process_query') as mock_process:
            mock_process.return_value = {
                "answer": "Memory test answer",
                "sources": ["Test source"],
                "confidence": 0.9
            }
            
            for i in range(100):
                mock_process(f"Query {i}")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
    
    @pytest.mark.asyncio
    async def test_async_query_processing(self):
        """Test asynchronous query processing"""
        async def mock_async_process(query):
            await asyncio.sleep(0.1)  # Simulate processing time
            return {
                "answer": f"Async answer for: {query}",
                "sources": ["Async source"],
                "confidence": 0.9
            }
        
        queries = [f"Async query {i}" for i in range(5)]
        
        start_time = time.time()
        tasks = [mock_async_process(query) for query in queries]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        assert len(results) == 5
        # Async processing should be faster than sequential
        assert total_time < 1.0  # Should complete in under 1 second
    
    def test_cache_performance(self):
        """Test cache hit performance"""
        with patch('cache.get_cached_response') as mock_cache_get, \
             patch('cache.cache_response') as mock_cache_set:
            
            # Test cache miss (slower)
            mock_cache_get.return_value = None
            start_time = time.time()
            mock_cache_get("test query")
            cache_miss_time = time.time() - start_time
            
            # Test cache hit (faster)
            cached_response = {"answer": "Cached answer", "sources": []}
            mock_cache_get.return_value = cached_response
            start_time = time.time()
            result = mock_cache_get("test query")
            cache_hit_time = time.time() - start_time
            
            # Cache hit should be significantly faster
            assert cache_hit_time < cache_miss_time
            assert result == cached_response