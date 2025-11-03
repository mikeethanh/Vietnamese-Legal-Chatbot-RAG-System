"""
Configuration file for search parameters and tuning
"""

# Hybrid Search Configuration
HYBRID_SEARCH_CONFIG = {
    # Search weights (should sum to 1.0)
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    
    # BM25 parameters
    "bm25_k1": 1.5,  # Term frequency saturation point
    "bm25_b": 0.75,  # Length normalization
    
    # Search limits
    "default_limit": 10,
    "vector_search_limit": 15,
    "keyword_search_limit": 15,
    "max_query_expansions": 5,
    
    # Scoring thresholds
    "min_similarity_score": 0.3,
    "min_keyword_score": 0.1,
    "hybrid_score_threshold": 0.2,
}

# Query Expansion Configuration
QUERY_EXPANSION_CONFIG = {
    "enabled": True,
    "max_synonyms_per_term": 4,
    "min_term_length": 2,
    
    # Legal domain specific settings
    "use_legal_synonyms": True,
    "use_procedural_expansion": True,  # Expand procedural terms
    "use_temporal_expansion": False,   # Expand time-related terms
}

# Vector Search Configuration
VECTOR_SEARCH_CONFIG = {
    "collection_name": "legal_docs_v1",
    "vector_size": 1024,
    "distance_metric": "DOT",  # DOT, COSINE, EUCLIDEAN
    "batch_size": 100,
    
    # Filtering options
    "enable_metadata_filtering": True,
    "enable_content_type_filtering": True,
    "enable_date_filtering": False,
}

# Retrieval Pipeline Configuration
RETRIEVAL_CONFIG = {
    "use_hybrid_search": True,
    "use_multi_query": True,
    "num_query_variations": 3,
    "use_query_expansion": True,
    
    # Fallback configuration
    "enable_fallback": True,
    "fallback_to_vector_search": True,
    "fallback_timeout": 5.0,  # seconds
    
    # Reranking
    "enable_reranking": True,
    "rerank_top_k": 5,
    "rerank_model": "rerank-multilingual-v3.0",
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    # Caching
    "enable_query_cache": True,
    "query_cache_ttl": 3600,  # 1 hour
    "enable_result_cache": True,
    "result_cache_ttl": 1800,  # 30 minutes
    
    # Timeouts
    "search_timeout": 10.0,
    "embedding_timeout": 5.0,
    "rerank_timeout": 8.0,
    
    # Concurrency
    "max_concurrent_searches": 3,
    "enable_parallel_search": True,
}

# Legal Domain Configuration
LEGAL_DOMAIN_CONFIG = {
    # Document types priority (higher = more important)
    "document_type_weights": {
        "law": 1.0,        # Luật
        "decree": 0.9,     # Nghị định
        "circular": 0.8,   # Thông tư
        "decision": 0.7,   # Quyết định
        "contract": 0.6,   # Hợp đồng
        "case_law": 0.5,   # Án lệ
        "general": 0.4,    # Tài liệu khác
    },
    
    # Content freshness weights (newer = higher weight)
    "freshness_decay_days": 365,  # How fast old content loses relevance
    "enable_freshness_boost": False,
    
    # Authority weighting
    "authority_weights": {
        "constitutional": 1.0,    # Hiến pháp
        "law": 0.9,              # Luật
        "ordinance": 0.8,        # Pháp lệnh
        "decree": 0.7,           # Nghị định
        "decision": 0.6,         # Quyết định
        "circular": 0.5,         # Thông tư
        "guidance": 0.4,         # Hướng dẫn
    }
}

# Debug and Monitoring Configuration
DEBUG_CONFIG = {
    "log_search_queries": True,
    "log_search_results": True,
    "log_timing": True,
    "log_scores": True,
    
    # Metrics collection
    "collect_search_metrics": True,
    "metrics_sample_rate": 0.1,  # 10% of searches
}


def get_config(config_name: str) -> dict:
    """
    Get configuration by name
    
    Args:
        config_name: Name of the configuration
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "hybrid_search": HYBRID_SEARCH_CONFIG,
        "query_expansion": QUERY_EXPANSION_CONFIG,
        "vector_search": VECTOR_SEARCH_CONFIG,
        "retrieval": RETRIEVAL_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "legal_domain": LEGAL_DOMAIN_CONFIG,
        "debug": DEBUG_CONFIG,
    }
    
    return configs.get(config_name, {})


def update_config(config_name: str, updates: dict) -> bool:
    """
    Update configuration values
    
    Args:
        config_name: Name of the configuration to update
        updates: Dictionary of values to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_map = {
            "hybrid_search": HYBRID_SEARCH_CONFIG,
            "query_expansion": QUERY_EXPANSION_CONFIG,
            "vector_search": VECTOR_SEARCH_CONFIG,
            "retrieval": RETRIEVAL_CONFIG,
            "performance": PERFORMANCE_CONFIG,
            "legal_domain": LEGAL_DOMAIN_CONFIG,
            "debug": DEBUG_CONFIG,
        }
        
        if config_name in config_map:
            config_map[config_name].update(updates)
            return True
        
        return False
        
    except Exception:
        return False