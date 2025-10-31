"""
Test cases for FastAPI app endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add backend src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

# Mock dependencies before importing app
with patch('redis.Redis'), \
     patch('qdrant_client.QdrantClient'), \
     patch('sqlalchemy.create_engine'):
    try:
        from app import app
        client = TestClient(app)
        APP_AVAILABLE = True
    except ImportError:
        APP_AVAILABLE = False
        pytestmark = pytest.mark.skip("FastAPI app not available")

@pytest.mark.skipif(not APP_AVAILABLE, reason="FastAPI app not available")
class TestFastAPIApp:
    """Test FastAPI application endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    @patch('brain.process_query')
    def test_chat_endpoint_valid_query(self, mock_process):
        """Test chat endpoint with valid query"""
        mock_process.return_value = {
            "answer": "Đây là câu trả lời về luật pháp Việt Nam",
            "sources": ["Luật ABC điều 123"],
            "confidence": 0.95
        }
        
        response = client.post("/chat", json={
            "query": "Luật giao thông quy định gì về tốc độ?",
            "session_id": "test-session"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["answer"]) > 0
    
    def test_chat_endpoint_missing_query(self):
        """Test chat endpoint with missing query"""
        response = client.post("/chat", json={
            "session_id": "test-session"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_endpoint_empty_query(self):
        """Test chat endpoint with empty query"""
        response = client.post("/chat", json={
            "query": "",
            "session_id": "test-session"
        })
        
        assert response.status_code == 400
    
    @patch('import_data.import_documents')
    def test_import_data_endpoint(self, mock_import):
        """Test data import endpoint"""
        mock_import.return_value = {"status": "success", "imported": 10}
        
        response = client.post("/data/import", json={
            "collection": "test_collection",
            "batch_size": 50
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/chat")
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]