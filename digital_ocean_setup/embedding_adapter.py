"""
Adapter để tích hợp Digital Ocean Embedding Service với backend hiện tại
"""

import requests
import logging
import os
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class DigitalOceanEmbeddingAdapter:
    """Adapter để sử dụng embedding service trên Digital Ocean"""
    
    def __init__(self, 
                 api_url: str = None,
                 local_model_path: str = None,
                 timeout: int = 30):
        """
        Initialize adapter với option để dùng API hoặc local model
        
        Args:
            api_url: URL của embedding API trên Digital Ocean
            local_model_path: Path của model local (backup option)
            timeout: Timeout cho API calls
        """
        self.api_url = api_url or os.getenv('EMBEDDING_API_URL', 'http://localhost:5000')
        self.timeout = timeout
        self.local_model = None
        
        # Load local model như backup
        if local_model_path and os.path.exists(local_model_path):
            try:
                self.local_model = SentenceTransformer(local_model_path)
                logger.info(f"Loaded local backup model from {local_model_path}")
            except Exception as e:
                logger.warning(f"Could not load local model: {e}")
                
    def _check_api_health(self) -> bool:
        """Kiểm tra API có hoạt động không"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
            
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts thành embeddings
        Tự động fallback sang local model nếu API không hoạt động
        """
        # Thử API trước
        if self._check_api_health():
            try:
                return self._encode_via_api(texts)
            except Exception as e:
                logger.warning(f"API encoding failed: {e}, falling back to local model")
                
        # Fallback sang local model
        if self.local_model:
            try:
                return self._encode_via_local(texts)
            except Exception as e:
                logger.error(f"Local encoding failed: {e}")
                raise
                
        raise RuntimeError("Both API and local model are unavailable")
        
    def _encode_via_api(self, texts: List[str]) -> np.ndarray:
        """Encode qua API"""
        payload = {"texts": texts}
        response = requests.post(
            f"{self.api_url}/embed",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API error: {response.status_code} - {response.text}")
            
        result = response.json()
        return np.array(result['embeddings'])
        
    def _encode_via_local(self, texts: List[str]) -> np.ndarray:
        """Encode qua local model"""
        if not self.local_model:
            raise RuntimeError("Local model not available")
            
        return self.local_model.encode(texts, convert_to_numpy=True)
        
    def compute_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Tính similarity giữa 2 sets of texts"""
        if self._check_api_health():
            try:
                payload = {
                    "texts1": texts1,
                    "texts2": texts2
                }
                response = requests.post(
                    f"{self.api_url}/similarity",
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return np.array(result['similarities'])
                    
            except Exception as e:
                logger.warning(f"API similarity failed: {e}, computing locally")
                
        # Fallback: compute locally
        embeddings1 = self.encode(texts1)
        embeddings2 = self.encode(texts2)
        
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embeddings1, embeddings2)

# Global instance
_embedding_adapter = None

def get_embedding_adapter() -> DigitalOceanEmbeddingAdapter:
    """Singleton pattern để get embedding adapter"""
    global _embedding_adapter
    
    if _embedding_adapter is None:
        api_url = os.getenv('EMBEDDING_API_URL')
        local_model = os.getenv('LOCAL_EMBEDDING_MODEL_PATH')
        
        _embedding_adapter = DigitalOceanEmbeddingAdapter(
            api_url=api_url,
            local_model_path=local_model
        )
        
    return _embedding_adapter

# Compatibility functions để thay thế SentenceTransformer
def encode_texts(texts: List[str]) -> np.ndarray:
    """Drop-in replacement cho SentenceTransformer.encode()"""
    adapter = get_embedding_adapter()
    return adapter.encode(texts)