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
        logger.info(f"🔧 Initializing search index with {len(documents)} documents")
        
        if not documents:
            logger.warning("⚠️ No documents provided for search index initialization")
            return False
        
        # Convert documents to LlamaIndex format
        llama_docs = []
        for i, doc in enumerate(documents):
            if not doc.get('question') and not doc.get('content'):
                logger.warning(f"⚠️ Document {i} has no question or content, skipping")
                continue
                
            # Combine question and content for better search
            text = f"{doc.get('question', '')} {doc.get('content', '')}"
            llama_doc = Document(
                text=text,
                metadata={
                    "question": doc.get('question', ''),
                    "content": doc.get('content', ''),
                    "source": doc.get('source', 'unknown'),
                    "doc_id": doc.get('doc_id', i)
                }
            )
            llama_docs.append(llama_doc)
        
        if not llama_docs:
            logger.error("❌ No valid documents after conversion")
            return False
        
        logger.info(f"📄 Converted {len(llama_docs)} valid documents")
        
        # Split documents into nodes with larger chunk size to accommodate metadata
        splitter = SentenceSplitter(chunk_size=1024)  # Increased from 512 to 1024
        nodes = splitter.get_nodes_from_documents(llama_docs)
        
        logger.info(f"🔍 Created {len(nodes)} nodes from {len(llama_docs)} documents")
        
        # Initialize docstore
        _docstore = SimpleDocumentStore()
        _docstore.add_documents(nodes)
        logger.info(f"📚 Docstore initialized with {len(nodes)} nodes")
        
        # Initialize BM25 retriever without stemmer for simplicity
        _bm25_retriever = BM25Retriever.from_defaults(
            docstore=_docstore,
            similarity_top_k=5,
        )
        logger.info("🔍 BM25 retriever initialized successfully")
        
        _search_engine_initialized = True
        logger.info("✅ Search index initialized successfully!")
        
        # Verify initialization
        stats = get_search_stats()
        logger.info(f"📊 Search stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search index: {e}")
        logger.exception("Full error traceback:")
        _search_engine_initialized = False
        return False


def hybrid_search(query: str, limit: int = 10) -> List[Dict]:
    """
    Perform hybrid search combining BM25 keyword search and vector semantic search
    
    Args:
        query: Search query
        limit: Maximum number of results to return
        
    Returns:
        List of documents with hybrid scores
    """
    if not _search_engine_initialized or not _bm25_retriever:
        logger.warning("⚠️ Search engine not initialized, checking if we can force initialize...")
        force_initialize_if_needed()
        
        if not _search_engine_initialized or not _bm25_retriever:
            logger.warning("⚠️ Search engine still not available, falling back to vector search only")
            return vector_search_fallback(query, limit)
    
    try:
        # 1. BM25 keyword search
        bm25_results = _bm25_retriever.retrieve(query)
        logger.info(f"🔍 BM25 search returned {len(bm25_results)} results")
        
        # 2. Vector semantic search
        vector = get_embedding(query)
        vector_results = search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
        logger.info(f"🔍 Vector search returned {len(vector_results)} results")
        
        # 3. Combine and score results
        combined_results = combine_search_results(bm25_results, vector_results, query)
        
        # 4. Sort by hybrid score and limit results
        combined_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        logger.info(f"✅ Hybrid search returned {len(combined_results[:limit])} final results")
        return combined_results[:limit]
        
    except Exception as e:
        logger.error(f"❌ Hybrid search failed: {e}, falling back to vector search")
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
    
    for i, node in enumerate(bm25_results):
        content = node.node.text if hasattr(node.node, 'text') else str(node.node)
        content_hash = hash(content)
        question = node.node.metadata.get("question", "")
        
        logger.info(f"   BM25[{i+1}]: Score={node.score:.3f}, Q='{question[:50]}...'")
        
        bm25_docs[content_hash] = {
            "content": content,
            "question": question,
            "source": node.node.metadata.get("source", "unknown"),
            "doc_id": node.node.metadata.get("doc_id", 0),
            "bm25_score": node.score,
            "search_method": "bm25"
        }
    
    # Convert vector results to dict format
    vector_docs = {}
    logger.info(f"📝 Processing {len(vector_results)} Vector results...")
    
    for i, doc in enumerate(vector_results):
        content = doc.get("content", "")
        content_hash = hash(content)
        question = doc.get("question", "")
        
        logger.info(f"   Vector[{i+1}]: Score={doc.get('similarity_score', 0):.3f}, Q='{question[:50]}...'")
        
        vector_docs[content_hash] = {
            "content": content,
            "question": question,
            "source": doc.get("source", "unknown"),
            "doc_id": doc.get("doc_id", 0),
            "vector_score": doc.get("similarity_score", 0),
            "search_method": "vector"
        }
    
    # Combine results
    all_docs = {}
    overlap_count = 0
    
    logger.info(f"🔗 Combining results...")
    
    # Add BM25 results
    for content_hash, doc in bm25_docs.items():
        all_docs[content_hash] = doc
    logger.info(f"   Added {len(bm25_docs)} BM25 documents")
    
    # Add vector results and merge if overlap
    for content_hash, doc in vector_docs.items():
        if content_hash in all_docs:
            # Merge scores for documents found by both methods
            all_docs[content_hash]["vector_score"] = doc["vector_score"]
            all_docs[content_hash]["search_method"] = "hybrid"
            overlap_count += 1
            logger.info(f"   ✨ Found overlap: '{doc['question'][:40]}...' (BM25 + Vector)")
        else:
            all_docs[content_hash] = doc
    
    logger.info(f"   Added {len(vector_docs)} Vector documents ({overlap_count} overlaps)")
    logger.info(f"   Total unique documents: {len(all_docs)}")
    
    # Calculate hybrid scores
    logger.info(f"⚖️ Calculating hybrid scores...")
    hybrid_count = bm25_only_count = vector_only_count = 0
    
    for doc in all_docs.values():
        bm25_score = doc.get("bm25_score", 0)
        vector_score = doc.get("vector_score", 0)
        
        # Simple hybrid scoring: weighted combination
        # Give equal weight to both methods, boost if found by both
        if doc["search_method"] == "hybrid":
            doc["hybrid_score"] = (bm25_score * 0.5) + (vector_score * 0.5) + 0.1  # Bonus for being in both
            hybrid_count += 1
        elif doc["search_method"] == "bm25":
            doc["hybrid_score"] = bm25_score * 0.6  # Slightly lower weight for BM25 only
            bm25_only_count += 1
        else:  # vector only
            doc["hybrid_score"] = vector_score * 0.6  # Slightly lower weight for vector only
            vector_only_count += 1
    
    logger.info(f"   Scoring breakdown: Hybrid={hybrid_count}, BM25-only={bm25_only_count}, Vector-only={vector_only_count}")
    
    # Sort by score for logging top results
    sorted_docs = sorted(all_docs.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)
    
    logger.info(f"🏆 Top 3 combined results:")
    for i, doc in enumerate(sorted_docs[:3], 1):
        question = doc.get("question", "N/A")
        score = doc.get("hybrid_score", 0)
        method = doc.get("search_method", "unknown")
        bm25_s = doc.get("bm25_score", 0)
        vector_s = doc.get("vector_score", 0)
        logger.info(f"   {i}. {question[:50]}... (Score: {score:.3f}, Method: {method}, BM25: {bm25_s:.3f}, Vec: {vector_s:.3f})")
    
    logger.info(f"✅ Combined search results: {len(all_docs)} total documents")
    
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


def force_initialize_if_needed() -> bool:
    """
    Force initialize search engine if not already done
    This is a helper function to ensure search engine is ready
    """
    global _search_engine_initialized
    
    if _search_engine_initialized:
        logger.info("✅ Search engine already initialized")
        return True
        
    logger.warning("⚠️ Search engine not initialized, attempting to force initialize...")
    
    # Try to get some sample documents from the database/vector store
    try:
        from vectorize import get_collection_stats
        from configs import DEFAULT_COLLECTION_NAME
        
        logger.info(f"🔍 Checking collection: {DEFAULT_COLLECTION_NAME}")
        stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
        logger.info(f"📊 Collection stats: {stats}")
        
        if stats and not stats.get('error'):
            # Check both vectors_count and points_count as fallback
            vectors_count = stats.get('vectors_count') or 0
            points_count = stats.get('points_count') or 0
            
            logger.info(f"🔢 Parsed counts - vectors: {vectors_count}, points: {points_count}")
            
            if vectors_count > 0 or points_count > 0:
                logger.info(f"📊 Found {vectors_count or points_count} documents in collection (vectors: {vectors_count}, points: {points_count})")
                logger.info("🔄 Attempting to initialize search index from existing data...")
                
                # Try to initialize from existing vector data
                success = initialize_from_vector_store()
                if success:
                    logger.info("✅ Successfully initialized search index from vector store!")
                    return True
                else:
                    logger.warning("💡 Please run import_data.py to initialize the search index properly")
            else:
                logger.warning(f"📋 No documents found in collection. vectors_count={vectors_count}, points_count={points_count}")
        else:
            error_msg = stats.get('error', 'Unknown error') if stats else 'Collection not found'
            logger.warning(f"📋 Could not access collection: {error_msg}")
            
    except Exception as e:
        logger.error(f"❌ Error checking collection stats: {e}")
    
    return False


def initialize_from_vector_store(limit: int = 1000) -> bool:
    """
    Initialize search index from existing vector store data
    
    Args:
        limit: Maximum number of documents to load
        
    Returns:
        bool: True if successful
    """
    try:
        from vectorize import client
        from configs import DEFAULT_COLLECTION_NAME
        
        logger.info(f"🔄 Loading documents from vector store (limit: {limit})")
        
        # Get documents directly from Qdrant using scroll
        scroll_result = client.scroll(
            collection_name=DEFAULT_COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        points = scroll_result[0]  # First element is the list of points
        
        if not points:
            logger.warning("📋 No documents found in vector store")
            return False
        
        logger.info(f"📄 Loaded {len(points)} documents from vector store")
        
        # Convert vector store results to the format expected by initialize_search_index
        documents = []
        for point in points:
            payload = point.payload
            doc = {
                'question': payload.get('question', ''),
                'content': payload.get('content', ''),
                'source': payload.get('source', 'vector_store'),
                'doc_id': payload.get('doc_id', point.id)
            }
            documents.append(doc)
        
        logger.info(f"📝 Converted {len(documents)} documents")
        
        # Initialize search index with these documents
        return initialize_search_index(documents)
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize from vector store: {e}")
        logger.exception("Full error traceback:")
        return False


