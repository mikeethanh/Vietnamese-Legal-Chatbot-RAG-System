"""
Configuration and fixtures for pytest
"""

import os
import sys
from typing import Generator
from unittest.mock import Mock, patch

import pytest

# Add backend src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "src"))


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch("redis.Redis") as mock:
        yield mock


@pytest.fixture
def mock_openai():
    """Mock OpenAI client"""
    with patch("openai.OpenAI") as mock:
        yield mock


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant client"""
    with patch("qdrant_client.QdrantClient") as mock:
        yield mock


@pytest.fixture
def mock_database():
    """Mock database connection"""
    with patch("sqlalchemy.create_engine") as mock:
        yield mock


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "OPENAI_API_KEY": "test-key",
        "REDIS_URL": "redis://localhost:6379",
        "QDRANT_URL": "http://localhost:6333",
        "DATABASE_URL": "mysql://test:test@localhost/test",
        "DEBUG": True,
    }


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment variables"""
    os.environ.update(
        {
            "OPENAI_API_KEY": "test-key",
            "REDIS_URL": "redis://localhost:6379",
            "QDRANT_URL": "http://localhost:6333",
            "DATABASE_URL": "mysql://test:test@localhost/test",
            "DEBUG": "true",
        }
    )
