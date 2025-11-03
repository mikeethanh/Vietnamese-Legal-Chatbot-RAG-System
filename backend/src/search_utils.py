"""
Search testing and monitoring utilities
"""

import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from search import hybrid_search, search_engine
from vectorize import search_vector, get_collection_stats
from query_rewriter import rewrite_query_to_multi_queries, expand_legal_query
from search_config import get_config

logger = logging.getLogger(__name__)


class SearchPerformanceMonitor:
    """
    Monitor and analyze search performance
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.query_cache = {}
        
    def log_search_performance(self, query: str, method: str, 
                             results: List[Dict], execution_time: float):
        """
        Log search performance metrics
        """
        metric = {
            "query": query,
            "method": method,
            "num_results": len(results),
            "execution_time": execution_time,
            "timestamp": time.time(),
        }
        
        if results:
            # Calculate average score
            scores = [r.get('hybrid_score', r.get('similarity_score', 0)) for r in results]
            metric["avg_score"] = sum(scores) / len(scores)
            metric["max_score"] = max(scores)
            metric["min_score"] = min(scores)
        
        self.metrics[method].append(metric)
        
        if get_config("debug").get("log_timing", False):
            logger.info(f"Search performance - Method: {method}, "
                       f"Query: {query[:50]}..., "
                       f"Results: {len(results)}, "
                       f"Time: {execution_time:.3f}s")
    
    def get_performance_summary(self, method: Optional[str] = None) -> Dict:
        """
        Get performance summary for search methods
        """
        if method:
            metrics = self.metrics.get(method, [])
        else:
            metrics = []
            for method_metrics in self.metrics.values():
                metrics.extend(method_metrics)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        execution_times = [m["execution_time"] for m in metrics]
        num_results = [m["num_results"] for m in metrics]
        
        return {
            "total_searches": len(metrics),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "avg_results": sum(num_results) / len(num_results),
            "methods": list(self.metrics.keys()),
        }


# Global performance monitor
performance_monitor = SearchPerformanceMonitor()


def benchmark_search_methods(test_queries: List[str], top_k: int = 5) -> Dict:
    """
    Benchmark different search methods
    
    Args:
        test_queries: List of test queries
        top_k: Number of results to retrieve
        
    Returns:
        Benchmark results
    """
    results = {
        "hybrid_search": [],
        "vector_search": [],
        "query_expansion": [],
    }
    
    logger.info(f"Benchmarking {len(test_queries)} queries with {top_k} results each")
    
    for query in test_queries:
        logger.info(f"Testing query: {query}")
        
        # Test hybrid search
        start_time = time.time()
        try:
            hybrid_results = hybrid_search(query, limit=top_k)
            hybrid_time = time.time() - start_time
            results["hybrid_search"].append({
                "query": query,
                "results": len(hybrid_results),
                "time": hybrid_time,
                "success": True
            })
            performance_monitor.log_search_performance(
                query, "hybrid", hybrid_results, hybrid_time
            )
        except Exception as e:
            results["hybrid_search"].append({
                "query": query,
                "error": str(e),
                "success": False
            })
        
        # Test pure vector search
        start_time = time.time()
        try:
            from brain import get_embedding
            from configs import DEFAULT_COLLECTION_NAME
            
            vector = get_embedding(query)
            vector_results = search_vector(DEFAULT_COLLECTION_NAME, vector, top_k)
            vector_time = time.time() - start_time
            
            results["vector_search"].append({
                "query": query,
                "results": len(vector_results),
                "time": vector_time,
                "success": True
            })
            performance_monitor.log_search_performance(
                query, "vector", vector_results, vector_time
            )
        except Exception as e:
            results["vector_search"].append({
                "query": query,
                "error": str(e),
                "success": False
            })
        
        # Test query expansion
        start_time = time.time()
        try:
            expanded_query = expand_legal_query(query)
            expanded_queries = rewrite_query_to_multi_queries(expanded_query, 3)
            expansion_time = time.time() - start_time
            
            results["query_expansion"].append({
                "query": query,
                "expanded_queries": expanded_queries,
                "time": expansion_time,
                "success": True
            })
        except Exception as e:
            results["query_expansion"].append({
                "query": query,
                "error": str(e),
                "success": False
            })
    
    # Calculate summary statistics
    for method, method_results in results.items():
        successful_results = [r for r in method_results if r.get("success", False)]
        if successful_results:
            times = [r["time"] for r in successful_results]
            results[f"{method}_summary"] = {
                "successful_queries": len(successful_results),
                "avg_time": sum(times) / len(times),
                "max_time": max(times),
                "min_time": min(times),
            }
    
    return results


def test_search_quality(queries_and_expected: List[tuple], top_k: int = 5) -> Dict:
    """
    Test search quality with expected results
    
    Args:
        queries_and_expected: List of (query, expected_keywords) tuples
        top_k: Number of results to retrieve
        
    Returns:
        Quality metrics
    """
    quality_metrics = {
        "total_queries": len(queries_and_expected),
        "query_results": [],
        "overall_precision": 0.0,
        "overall_recall": 0.0,
    }
    
    total_precision = 0.0
    total_recall = 0.0
    
    for query, expected_keywords in queries_and_expected:
        try:
            # Get search results
            results = hybrid_search(query, limit=top_k)
            
            # Check how many results contain expected keywords
            relevant_results = 0
            for result in results:
                content = result.get("content", "").lower()
                if any(keyword.lower() in content for keyword in expected_keywords):
                    relevant_results += 1
            
            # Calculate precision and recall
            precision = relevant_results / len(results) if results else 0.0
            recall = relevant_results / len(expected_keywords) if expected_keywords else 0.0
            
            quality_metrics["query_results"].append({
                "query": query,
                "expected_keywords": expected_keywords,
                "total_results": len(results),
                "relevant_results": relevant_results,
                "precision": precision,
                "recall": recall,
            })
            
            total_precision += precision
            total_recall += recall
            
        except Exception as e:
            logger.error(f"Quality test failed for query '{query}': {e}")
            quality_metrics["query_results"].append({
                "query": query,
                "error": str(e),
                "success": False,
            })
    
    # Calculate overall metrics
    successful_queries = [r for r in quality_metrics["query_results"] 
                         if r.get("success", True)]
    
    if successful_queries:
        quality_metrics["overall_precision"] = total_precision / len(successful_queries)
        quality_metrics["overall_recall"] = total_recall / len(successful_queries)
    
    return quality_metrics


def diagnose_search_issues(query: str) -> Dict:
    """
    Diagnose potential search issues for a query
    
    Args:
        query: Query to diagnose
        
    Returns:
        Diagnostic information
    """
    diagnosis = {
        "query": query,
        "issues": [],
        "suggestions": [],
        "metrics": {},
    }
    
    try:
        # Test query expansion
        expanded_query = expand_legal_query(query)
        if expanded_query == query:
            diagnosis["issues"].append("No legal terms found for expansion")
        else:
            diagnosis["metrics"]["expanded_query"] = expanded_query
        
        # Test query rewriting
        rewritten_queries = rewrite_query_to_multi_queries(query, 3)
        if len(set(rewritten_queries)) == 1:
            diagnosis["issues"].append("Query rewriting produced no variations")
        else:
            diagnosis["metrics"]["query_variations"] = len(set(rewritten_queries))
        
        # Test vector embedding
        start_time = time.time()
        from brain import get_embedding
        vector = get_embedding(query)
        embedding_time = time.time() - start_time
        
        if embedding_time > 2.0:
            diagnosis["issues"].append("Slow embedding generation")
        
        diagnosis["metrics"]["embedding_time"] = embedding_time
        diagnosis["metrics"]["embedding_dimension"] = len(vector)
        
        # Test search methods
        start_time = time.time()
        hybrid_results = hybrid_search(query, limit=5)
        hybrid_time = time.time() - start_time
        
        diagnosis["metrics"]["hybrid_search_time"] = hybrid_time
        diagnosis["metrics"]["hybrid_results_count"] = len(hybrid_results)
        
        if not hybrid_results:
            diagnosis["issues"].append("No results found with hybrid search")
            diagnosis["suggestions"].append("Try more general terms or check document index")
        
        if hybrid_time > 5.0:
            diagnosis["issues"].append("Slow hybrid search performance")
            diagnosis["suggestions"].append("Consider optimizing search parameters")
        
        # Check collection statistics
        try:
            from configs import DEFAULT_COLLECTION_NAME
            stats = get_collection_stats(DEFAULT_COLLECTION_NAME)
            diagnosis["metrics"]["collection_stats"] = stats
            
            if stats.get("vectors_count", 0) == 0:
                diagnosis["issues"].append("Empty vector database")
                diagnosis["suggestions"].append("Index documents before searching")
                
        except Exception as e:
            diagnosis["issues"].append(f"Cannot access vector database: {e}")
    
    except Exception as e:
        diagnosis["issues"].append(f"Critical error during diagnosis: {e}")
    
    return diagnosis


# Common test queries for Vietnamese legal domain
LEGAL_TEST_QUERIES = [
    "Thủ tục ly hôn như thế nào?",
    "Hợp đồng lao động có thời hạn",
    "Phạt vi phạm giao thông",
    "Quyền thừa kế của con cái",
    "Đăng ký kinh doanh cần giấy tờ gì?",
    "Thuế thu nhập cá nhân",
    "Bồi thường thiệt hại do vi phạm hợp đồng",
    "Trách nhiệm của người sử dụng lao động",
    "Quyền sở hữu tài sản chung vợ chồng",
    "Thủ tục khởi kiện dân sự",
]


def run_comprehensive_search_test():
    """
    Run comprehensive search system test
    """
    logger.info("Starting comprehensive search test")
    
    # Benchmark performance
    benchmark_results = benchmark_search_methods(LEGAL_TEST_QUERIES[:5])
    logger.info("Benchmark completed")
    
    # Test search quality (simplified - you'd need actual expected results)
    quality_test_data = [
        ("Thủ tục ly hôn", ["ly hôn", "hôn nhân", "chấm dứt"]),
        ("Hợp đồng lao động", ["lao động", "hợp đồng", "việc làm"]),
        ("Phạt giao thông", ["giao thông", "phạt", "vi phạm"]),
    ]
    
    quality_results = test_search_quality(quality_test_data)
    logger.info("Quality test completed")
    
    # Get performance summary
    performance_summary = performance_monitor.get_performance_summary()
    
    return {
        "benchmark": benchmark_results,
        "quality": quality_results,
        "performance": performance_summary,
        "timestamp": time.time(),
    }