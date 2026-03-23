"""
Custom Embedding Service - Sử dụng model riêng thay vì OpenAI
"""

import logging
import os
from functools import lru_cache
from typing import List, Union

import requests

logger = logging.getLogger(__name__)

# Configuration
CUSTOM_EMBEDDING_API_URL = os.environ.get(
    "CUSTOM_EMBEDDING_API_URL", "http://3.80.119.178:5001"
)
CUSTOM_EMBEDDING_ENABLED = (
    os.environ.get("CUSTOM_EMBEDDING_ENABLED", "true").lower() == "true"
)

# Fallback to OpenAI if custom embedding fails
USE_OPENAI_FALLBACK = os.environ.get("USE_OPENAI_FALLBACK", "true").lower() == "true"


class CustomEmbeddingService:
    """Service để generate embeddings từ custom model"""

    def __init__(self, api_url: str = CUSTOM_EMBEDDING_API_URL):
        self.api_url = api_url.rstrip("/")
        self.embedding_endpoint = f"{self.api_url}/embed"
        self.health_endpoint = f"{self.api_url}/health"
        self.timeout = 30  # seconds

        # Check service health on init
        self._check_health()

    def _check_health(self):
        """Check if custom embedding service is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"✅ Custom embedding service is healthy: "
                    f"device={data.get('device')}, "
                    f"dim={data.get('embedding_dim')}, "
                    f"model_loaded={data.get('model_loaded')}"
                )
                return True
            else:
                logger.warning(
                    f"⚠️ Custom embedding service health check failed: "
                    f"status={response.status_code}"
                )
                return False
        except Exception as e:
            logger.warning(f"⚠️ Cannot connect to custom embedding service: {e}")
            return False

    def get_embedding(
        self, text: Union[str, List[str]], batch_size: int = 32
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Args:
            text: Single text string or list of texts
            batch_size: Batch size for processing

        Returns:
            Single embedding vector or list of embedding vectors
        """
        # Normalize input
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        if not texts:
            raise ValueError("Input text cannot be empty")

        # Clean texts
        texts = [t.replace("\n", " ").strip() for t in texts]

        try:
            # Call API
            response = requests.post(
                self.embedding_endpoint,
                json={"texts": texts, "batch_size": batch_size},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise Exception(
                    f"API returned status {response.status_code}: " f"{response.text}"
                )

            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings:
                raise Exception("No embeddings returned from API")

            logger.info(
                f"✅ Generated {len(embeddings)} embeddings "
                f"(dim={len(embeddings[0])}, "
                f"time={data.get('processing_time', 0):.3f}s)"
            )

            # Return single embedding or list
            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"❌ Error getting embedding from custom service: {e}")
            raise

    def get_similarity(self, texts1: List[str], texts2: List[str]) -> List[List[float]]:
        """
        Calculate similarity between two sets of texts

        Args:
            texts1: First set of texts
            texts2: Second set of texts

        Returns:
            Similarity matrix [len(texts1), len(texts2)]
        """
        try:
            response = requests.post(
                f"{self.api_url}/similarity",
                json={"texts1": texts1, "texts2": texts2},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise Exception(
                    f"API returned status {response.status_code}: " f"{response.text}"
                )

            data = response.json()
            similarities = data.get("similarities", [])

            logger.info(
                f"✅ Calculated similarities "
                f"(shape={data.get('shape')}, "
                f"time={data.get('processing_time', 0):.3f}s)"
            )

            return similarities

        except Exception as e:
            logger.error(f"❌ Error calculating similarity: {e}")
            raise


# Global instance
_embedding_service = None


def get_embedding_service() -> CustomEmbeddingService:
    """Get singleton instance of embedding service"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = CustomEmbeddingService()
    return _embedding_service


def get_custom_embedding(
    text: Union[str, List[str]], batch_size: int = 32
) -> Union[List[float], List[List[float]]]:
    """
    Convenience function to get embeddings from custom model

    Args:
        text: Single text or list of texts
        batch_size: Batch size for processing

    Returns:
        Embedding vector(s)
    """
    service = get_embedding_service()
    return service.get_embedding(text, batch_size)


def get_custom_similarity(texts1: List[str], texts2: List[str]) -> List[List[float]]:
    """
    Convenience function to calculate similarity

    Args:
        texts1: First set of texts
        texts2: Second set of texts

    Returns:
        Similarity matrix
    """
    service = get_embedding_service()
    return service.get_similarity(texts1, texts2)


# For backward compatibility and easy switching
if __name__ == "__main__":
    # Test the service
    import sys

    print("🧪 Testing Custom Embedding Service...")
    print(f"API URL: {CUSTOM_EMBEDDING_API_URL}")
    print(f"Enabled: {CUSTOM_EMBEDDING_ENABLED}")
    print()

    try:
        # Test single text
        print("Test 1: Single text embedding")
        text = "Luật Dân sự năm 2015 quy định về quyền sở hữu"
        embedding = get_custom_embedding(text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
        print()

        # Test batch
        print("Test 2: Batch embeddings")
        texts = [
            "Luật Dân sự năm 2015",
            "Bộ luật Hình sự năm 2017",
            "Luật Đất đai năm 2013",
        ]
        embeddings = get_custom_embedding(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print()

        # Test similarity
        print("Test 3: Similarity calculation")
        texts1 = ["Quyền sở hữu tài sản"]
        texts2 = ["Tài sản chung", "Luật Hình sự", "Quyền thừa kế"]
        similarities = get_custom_similarity(texts1, texts2)
        print(f"✅ Similarities: {similarities}")
        print()

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
