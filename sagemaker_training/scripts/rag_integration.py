#!/usr/bin/env python3
"""
Script để integrate trained SageMaker embedding model với backend hiện tại
"""

import boto3
import json
import numpy as np
from typing import List, Dict, Any
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

logger = logging.getLogger(__name__)

class SageMakerEmbeddingClient:
    """Client để gọi SageMaker embedding endpoint"""
    
    def __init__(self, endpoint_name: str, region: str = 'ap-southeast-1'):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
        
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts thành embeddings"""
        try:
            payload = {
                'texts': texts
            }
            
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            result = json.loads(response['Body'].read().decode())
            return result.get('embeddings', [])
            
        except Exception as e:
            logger.error(f"SageMaker encoding failed: {e}")
            raise

class QdrantEmbeddingService:
    """Service để quản lý embeddings trong Qdrant"""
    
    def __init__(self, host: str = "localhost", port: int = 6333, collection_name: str = "vietnamese_legal_docs"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        
    def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search similar documents"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            return [
                {
                    'id': result.payload.get('document_id'),
                    'text': result.payload.get('text'),
                    'score': result.score
                }
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get thông tin collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance': info.config.params.vectors.distance.value
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

class LegalRAGService:
    """Main service cho Legal RAG system với SageMaker embeddings"""
    
    def __init__(self, sagemaker_endpoint: str, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.embedding_client = SageMakerEmbeddingClient(sagemaker_endpoint)
        self.vector_service = QdrantEmbeddingService(qdrant_host, qdrant_port)
        
    def search_legal_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search legal documents based on query"""
        try:
            # Generate embedding cho query
            query_embeddings = self.embedding_client.encode([query])
            
            if not query_embeddings:
                return []
            
            query_embedding = query_embeddings[0]
            
            # Search trong Qdrant
            similar_docs = self.vector_service.search_similar(query_embedding, limit)
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Legal document search failed: {e}")
            return []
    
    def get_system_status(self) -> Dict:
        """Get status của toàn bộ system"""
        try:
            qdrant_info = self.vector_service.get_collection_info()
            
            # Test SageMaker endpoint
            test_embedding = self.embedding_client.encode(["test"])
            sagemaker_status = "online" if test_embedding else "offline"
            
            return {
                'sagemaker_endpoint': {
                    'status': sagemaker_status,
                    'endpoint_name': self.embedding_client.endpoint_name
                },
                'qdrant_collection': qdrant_info,
                'system_status': 'healthy' if sagemaker_status == 'online' and qdrant_info else 'degraded'
            }
            
        except Exception as e:
            logger.error(f"System status check failed: {e}")
            return {'system_status': 'error', 'error': str(e)}

# Example usage cho Flask/FastAPI integration
def create_legal_rag_api(app, sagemaker_endpoint: str):
    """Tạo API endpoints cho Flask app"""
    
    rag_service = LegalRAGService(sagemaker_endpoint)
    
    @app.route('/api/search', methods=['POST'])
    def search_documents():
        try:
            data = request.get_json()
            query = data.get('query', '')
            limit = data.get('limit', 10)
            
            if not query:
                return {'error': 'Query is required'}, 400
            
            results = rag_service.search_legal_documents(query, limit)
            
            return {
                'query': query,
                'results': results,
                'count': len(results)
            }
            
        except Exception as e:
            return {'error': str(e)}, 500
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        status = rag_service.get_system_status()
        return status
    
    return app

# Example usage
if __name__ == '__main__':
    # Test với SageMaker endpoint
    endpoint_name = "vietnamese-legal-embedding-endpoint-2024-10-20-12-00-00"
    
    try:
        rag_service = LegalRAGService(endpoint_name)
        
        # Test search
        results = rag_service.search_legal_documents("luật lao động", limit=5)
        
        print(f"Search results: {len(results)} documents found")
        for i, doc in enumerate(results[:3]):
            print(f"{i+1}. Score: {doc['score']:.3f}")
            print(f"   Text: {doc['text'][:100]}...")
            print()
        
        # System status
        status = rag_service.get_system_status()
        print(f"System status: {status}")
        
    except Exception as e:
        print(f"Error: {e}")