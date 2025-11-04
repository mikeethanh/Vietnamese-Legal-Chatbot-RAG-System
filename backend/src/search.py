"""
Enhanced Search Module for Vietnamese Legal Chatbot
Implements hybrid search combining semantic vector search + BM25 keyword search using LlamaIndex
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever

from brain import get_embedding
from configs import DEFAULT_COLLECTION_NAME
from vectorize import search_vector

logger = logging.getLogger(__name__)

# Global search components
_docstore = None
_bm25_retriever = None
_search_engine_initialized = False


def initialize_search_index(documents: List[Dict]) -> bool:
    """
    Initialize BM25 search index from documents
    
    Args:
        documents: List of documents with keys: question, content, source, doc_id
        
    Returns:
        bool: True if successful, False otherwise
    """
    global _docstore, _bm25_retriever, _search_engine_initialized
    
    try:
        logger.info(f"Initializing search index with {len(documents)} documents")
        
        # Convert documents to LlamaIndex format
        llama_docs = []
        for doc in documents:
            # Combine question and content for better search
            text = f"{doc['question']} {doc['content']}"
            llama_doc = Document(
                text=text,
                metadata={
                    "question": doc['question'],
                    "content": doc['content'],
                    "source": doc.get('source', 'unknown'),
                    "doc_id": doc.get('doc_id', 0)
                }
            )
            llama_docs.append(llama_doc)
        
        # Split documents into nodes
        splitter = SentenceSplitter(chunk_size=512)
        nodes = splitter.get_nodes_from_documents(llama_docs)
        
        logger.info(f"Created {len(nodes)} nodes from {len(llama_docs)} documents")
        
        # Initialize docstore
        _docstore = SimpleDocumentStore()
        _docstore.add_documents(nodes)
        
        # Initialize BM25 retriever without stemmer for simplicity
        _bm25_retriever = BM25Retriever.from_defaults(
            docstore=_docstore,
            similarity_top_k=5,
        )
        
        _search_engine_initialized = True
        logger.info("Search index initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize search index: {e}")
        return False


def hybrid_search(query: str, limit: int = 5) -> List[Dict]:
    """
    Perform hybrid search combining BM25 keyword search and vector semantic search
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of documents with hybrid scores
    """
    if not _search_engine_initialized or not _bm25_retriever:
        logger.warning("Search engine not initialized, falling back to vector search only")
        return vector_search_fallback(query, limit)
    
    try:
        # 1. BM25 keyword search
        bm25_results = _bm25_retriever.retrieve(query)
        logger.info(f"BM25 search returned {len(bm25_results)} results")
        
        # 2. Vector semantic search
        vector = get_embedding(query)
        vector_results = search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        # 3. Combine and score results
        combined_results = combine_search_results(bm25_results, vector_results, query)
        
        # 4. Sort by hybrid score and limit results
        combined_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        logger.info(f"Hybrid search returned {len(combined_results[:limit])} final results")
        return combined_results[:limit]
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}, falling back to vector search")
        return vector_search_fallback(query, limit)


def vector_search_fallback(query: str, limit: int = 5) -> List[Dict]:
    """
    Fallback to pure vector search when BM25 is not available
    
    Args:
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of documents
    """
    try:
        vector = get_embedding(query)
        results = search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
        
        # Add search method metadata
        for result in results:
            result["search_method"] = "vector_fallback"
            result["hybrid_score"] = result.get("similarity_score", 0)
        
        return results
        
    except Exception as e:
        logger.error(f"Vector search fallback failed: {e}")
        return []


def combine_search_results(bm25_results, vector_results, query: str) -> List[Dict]:
    """
    Combine BM25 and vector search results with hybrid scoring
    
    Args:
        bm25_results: Results from BM25 search
        vector_results: Results from vector search
        query: Original query for scoring
        
    Returns:
        List of combined documents with hybrid scores
    """
    # Convert BM25 results to dict format
    bm25_docs = {}
    for node in bm25_results:
        content = node.node.text if hasattr(node.node, 'text') else str(node.node)
        content_hash = hash(content)
        
        bm25_docs[content_hash] = {
            "content": content,
            "question": node.node.metadata.get("question", ""),
            "source": node.node.metadata.get("source", "unknown"),
            "doc_id": node.node.metadata.get("doc_id", 0),
            "bm25_score": node.score,
            "search_method": "bm25"
        }
    
    # Convert vector results to dict format
    vector_docs = {}
    for doc in vector_results:
        content = doc.get("content", "")
        content_hash = hash(content)
        
        vector_docs[content_hash] = {
            "content": content,
            "question": doc.get("question", ""),
            "source": doc.get("source", "unknown"),
            "doc_id": doc.get("doc_id", 0),
            "vector_score": doc.get("similarity_score", 0),
            "search_method": "vector"
        }
    
    # Combine results
    all_docs = {}
    
    # Add BM25 results
    for content_hash, doc in bm25_docs.items():
        all_docs[content_hash] = doc
    
    # Add vector results and merge if overlap
    for content_hash, doc in vector_docs.items():
        if content_hash in all_docs:
            # Merge scores for documents found by both methods
            all_docs[content_hash]["vector_score"] = doc["vector_score"]
            all_docs[content_hash]["search_method"] = "hybrid"
        else:
            all_docs[content_hash] = doc
    
    # Calculate hybrid scores
    for doc in all_docs.values():
        bm25_score = doc.get("bm25_score", 0)
        vector_score = doc.get("vector_score", 0)
        
        # Simple hybrid scoring: weighted combination
        # Give equal weight to both methods, boost if found by both
        if doc["search_method"] == "hybrid":
            doc["hybrid_score"] = (bm25_score * 0.5) + (vector_score * 0.5) + 0.1  # Bonus for being in both
        elif doc["search_method"] == "bm25":
            doc["hybrid_score"] = bm25_score * 0.6  # Slightly lower weight for BM25 only
        else:  # vector only
            doc["hybrid_score"] = vector_score * 0.6  # Slightly lower weight for vector only
    
    return list(all_docs.values())


def search_engine() -> bool:
    """
    Alias for backward compatibility
    """
    return _search_engine_initialized


def get_search_stats() -> Dict:
    """
    Get search engine statistics
    
    Returns:
        Dict with search engine status and stats
    """
    return {
        "initialized": _search_engine_initialized,
        "has_docstore": _docstore is not None,
        "has_bm25": _bm25_retriever is not None,
        "docstore_size": len(_docstore.docs) if _docstore else 0
    }


