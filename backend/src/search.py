"""
Enhanced Search Module for Vietnamese Legal Chatbot
Implements hybrid search combining semantic vector search + keyword search
"""

import logging
import re
import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from brain import get_embedding
from vectorize import search_vector
from configs import DEFAULT_COLLECTION_NAME

logger = logging.getLogger(__name__)


class BM25:
    """
    BM25 implementation for keyword search
    Optimized for Vietnamese legal documents
    """
    
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if corpus else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_count = len(corpus)
        
        # Build index
        self._build_index()
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess Vietnamese text for better keyword matching
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep Vietnamese characters
        text = re.sub(r'[^\w\sáàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Remove single characters and common stop words
        vietnamese_stopwords = {
            'và', 'của', 'có', 'được', 'trong', 'với', 'để', 'theo', 'từ', 'về',
            'này', 'đó', 'các', 'một', 'những', 'khi', 'nếu', 'thì', 'hoặc',
            'phải', 'không', 'là', 'sẽ', 'đã', 'cho', 'tại', 'trên', 'dưới'
        }
        
        words = [word for word in words if len(word) > 1 and word not in vietnamese_stopwords]
        
        return words
    
    def _build_index(self):
        """Build inverted index and calculate IDF scores"""
        df = defaultdict(int)
        
        for doc in self.corpus:
            words = self._preprocess_text(doc)
            word_freqs = Counter(words)
            self.doc_freqs.append(word_freqs)
            
            # Count document frequency for each word
            for word in set(words):
                df[word] += 1
        
        # Calculate IDF
        for word, freq in df.items():
            self.idf[word] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search documents using BM25 scoring
        
        Returns:
            List of (doc_index, score) sorted by score descending
        """
        query_words = self._preprocess_text(query)
        scores = []
        
        for i, doc_freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[i]
            
            for word in query_words:
                if word in doc_freqs:
                    tf = doc_freqs[word]
                    idf = self.idf.get(word, 0)
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores.append((i, score))
        
        # Sort by score descending and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridSearchEngine:
    """
    Hybrid search engine combining semantic vector search with keyword BM25 search
    Optimized for Vietnamese legal documents
    """
    
    def __init__(self):
        self.bm25_index = None
        self.documents_cache = []
        self.collection_name = DEFAULT_COLLECTION_NAME
        
        # Search weights
        self.vector_weight = 0.7  # Semantic search weight
        self.keyword_weight = 0.3  # Keyword search weight
        
        logger.info("Initialized HybridSearchEngine")
    
    def build_keyword_index(self, documents: List[Dict]):
        """
        Build BM25 index from documents
        
        Args:
            documents: List of document dicts with 'content' and 'question' fields
        """
        self.documents_cache = documents
        
        # Create corpus for BM25 (combine question + content)
        corpus = []
        for doc in documents:
            text = f"{doc.get('question', '')} {doc.get('content', '')}"
            corpus.append(text)
        
        # Build BM25 index
        self.bm25_index = BM25(corpus)
        logger.info(f"Built BM25 index for {len(documents)} documents")
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with legal synonyms and related terms
        """
        legal_synonyms = {
            'hợp đồng': ['hợp đồng', 'giao kèo', 'thỏa thuận'],
            'vi phạm': ['vi phạm', 'phạm', 'trái'],
            'phạt': ['phạt', 'xử phạt', 'tiền phạt'],
            'thừa kế': ['thừa kế', 'kế thừa', 'gia tài'],
            'ly hôn': ['ly hôn', 'li hôn', 'chấm dứt hôn nhân'],
            'thuế': ['thuế', 'lệ phí', 'phí'],
            'kiện tụng': ['kiện tụng', 'tranh chấp', 'tranh tụng'],
            'bồi thường': ['bồi thường', 'đền bù', 'bồi hoàn'],
        }
        
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            expanded_terms.append(word)
            # Add synonyms if found
            for key, synonyms in legal_synonyms.items():
                if word in key:
                    expanded_terms.extend(synonyms)
        
        return ' '.join(expanded_terms)
    
    def vector_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform semantic vector search
        """
        try:
            # Get embedding
            vector = get_embedding(query)
            
            # Search using Qdrant
            results = search_vector(self.collection_name, vector, limit)
            
            # Add vector scores (Qdrant returns results sorted by similarity)
            for i, doc in enumerate(results):
                doc['vector_score'] = 1.0 - (i * 0.1)  # Decreasing score
                doc['search_type'] = 'vector'
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Perform keyword search using BM25
        """
        if not self.bm25_index or not self.documents_cache:
            logger.warning("BM25 index not built, skipping keyword search")
            return []
        
        try:
            # Expand query
            expanded_query = self.expand_query(query)
            
            # Search using BM25
            bm25_results = self.bm25_index.search(expanded_query, limit)
            
            # Convert to document format
            results = []
            for doc_idx, score in bm25_results:
                if score > 0:  # Only include results with positive scores
                    doc = self.documents_cache[doc_idx].copy()
                    doc['keyword_score'] = score
                    doc['search_type'] = 'keyword'
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def combine_results(self, vector_results: List[Dict], keyword_results: List[Dict], 
                       top_k: int = 10) -> List[Dict]:
        """
        Combine and rank results from vector and keyword search
        """
        # Create a map to merge results by content
        combined_docs = {}
        
        # Process vector results
        for doc in vector_results:
            content_key = doc.get('content', '')[:100]  # Use first 100 chars as key
            if content_key not in combined_docs:
                combined_docs[content_key] = doc.copy()
                combined_docs[content_key]['vector_score'] = doc.get('vector_score', 0)
                combined_docs[content_key]['keyword_score'] = 0
            else:
                combined_docs[content_key]['vector_score'] = max(
                    combined_docs[content_key]['vector_score'], 
                    doc.get('vector_score', 0)
                )
        
        # Process keyword results
        for doc in keyword_results:
            content_key = doc.get('content', '')[:100]
            if content_key not in combined_docs:
                combined_docs[content_key] = doc.copy()
                combined_docs[content_key]['vector_score'] = 0
                combined_docs[content_key]['keyword_score'] = doc.get('keyword_score', 0)
            else:
                combined_docs[content_key]['keyword_score'] = max(
                    combined_docs[content_key]['keyword_score'], 
                    doc.get('keyword_score', 0)
                )
        
        # Normalize scores and calculate hybrid score
        max_vector_score = max([doc['vector_score'] for doc in combined_docs.values()], default=1.0)
        max_keyword_score = max([doc['keyword_score'] for doc in combined_docs.values()], default=1.0)
        
        for doc in combined_docs.values():
            # Normalize scores
            norm_vector_score = doc['vector_score'] / max_vector_score if max_vector_score > 0 else 0
            norm_keyword_score = doc['keyword_score'] / max_keyword_score if max_keyword_score > 0 else 0
            
            # Calculate hybrid score
            doc['hybrid_score'] = (
                self.vector_weight * norm_vector_score + 
                self.keyword_weight * norm_keyword_score
            )
            
            # Add metadata for debugging
            doc['norm_vector_score'] = norm_vector_score
            doc['norm_keyword_score'] = norm_keyword_score
        
        # Sort by hybrid score and return top_k
        results = list(combined_docs.values())
        results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return results[:top_k]
    
    def search(self, query: str, limit: int = 10, 
               vector_limit: int = 15, keyword_limit: int = 15) -> List[Dict]:
        """
        Main hybrid search method
        
        Args:
            query: Search query
            limit: Final number of results to return
            vector_limit: Number of results from vector search
            keyword_limit: Number of results from keyword search
            
        Returns:
            List of documents ranked by hybrid score
        """
        logger.info(f"Hybrid search for query: {query}")
        
        # Perform both searches in parallel conceptually
        vector_results = self.vector_search(query, vector_limit)
        keyword_results = self.keyword_search(query, keyword_limit)
        
        logger.info(f"Vector search returned {len(vector_results)} results")
        logger.info(f"Keyword search returned {len(keyword_results)} results")
        
        # Combine results
        combined_results = self.combine_results(vector_results, keyword_results, limit)
        
        logger.info(f"Hybrid search returning {len(combined_results)} results")
        
        # Log top result for debugging
        if combined_results:
            top_result = combined_results[0]
            logger.info(f"Top result - Hybrid: {top_result.get('hybrid_score', 0):.3f}, "
                       f"Vector: {top_result.get('norm_vector_score', 0):.3f}, "
                       f"Keyword: {top_result.get('norm_keyword_score', 0):.3f}")
        
        return combined_results


# Global search engine instance
search_engine = HybridSearchEngine()


def hybrid_search(query: str, limit: int = 10) -> List[Dict]:
    """
    Convenience function for hybrid search
    """
    return search_engine.search(query, limit)


def initialize_search_index(documents: List[Dict]):
    """
    Initialize the search index with documents
    Call this when loading documents into the system
    """
    search_engine.build_keyword_index(documents)


def update_search_weights(vector_weight: float, keyword_weight: float):
    """
    Update search weights for tuning
    """
    total = vector_weight + keyword_weight
    search_engine.vector_weight = vector_weight / total
    search_engine.keyword_weight = keyword_weight / total
    
    logger.info(f"Updated search weights - Vector: {search_engine.vector_weight:.2f}, "
               f"Keyword: {search_engine.keyword_weight:.2f}")