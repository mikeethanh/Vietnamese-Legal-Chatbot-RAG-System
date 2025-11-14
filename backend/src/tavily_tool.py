"""
Tavily AI Search integration for Vietnamese legal chatbot
Provides better web search capabilities with AI-powered summarization
"""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


def tavily_search(
    query: str, max_results: int = 5, search_depth: str = "basic"
) -> Dict:
    """
    Search using Tavily AI - provides AI-optimized search results.
    Falls back to simple implementation if tavily-python not available.

    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)
        search_depth: "basic" or "advanced" (default: "basic")

    Returns:
        Dict with search results
    """
    try:
        # Try importing tavily
        try:
            from tavily import TavilyClient

            tavily_available = True
        except ImportError:
            logger.warning("tavily-python not installed, using fallback")
            tavily_available = False

        if not TAVILY_API_KEY:
            logger.warning("TAVILY_API_KEY not found")
            return {
                "error": "TAVILY_API_KEY not configured",
                "query": query,
                "results": [],
            }

        if not tavily_available:
            # Simple fallback without actual API call
            return {
                "query": query,
                "results": [],
                "note": "Tavily library not installed. Install with: pip install tavily-python",
            }

        # Use Tavily client
        client = TavilyClient(api_key=TAVILY_API_KEY)

        # Perform search
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_answer=True,  # Get AI-generated answer
            include_raw_content=False,  # Don't need full page content
        )

        # Format results
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                }
            )

        output = {
            "query": query,
            "answer": response.get("answer", ""),  # AI-generated summary answer
            "results": formatted_results,
            "num_results": len(formatted_results),
        }

        logger.info(
            f"[TAVILY] Search completed: {len(formatted_results)} results for '{query}'"
        )
        return output

    except Exception as e:
        logger.error(f"[TAVILY] Search error: {e}")
        return {"error": str(e), "query": query, "results": []}


def tavily_search_legal(query: str, max_results: int = 5) -> str:
    """
    Tavily search specifically optimized for Vietnamese legal queries.
    Returns formatted string suitable for agent tool.

    Args:
        query: Legal search query
        max_results: Maximum results

    Returns:
        Formatted search results as string
    """
    try:
        # Add Vietnamese legal context to query
        enhanced_query = f"Việt Nam pháp luật: {query}"

        results = tavily_search(
            enhanced_query, max_results=max_results, search_depth="advanced"
        )

        if "error" in results:
            return f"Lỗi tìm kiếm: {results['error']}"

        # Format output
        output = f"Kết quả tìm kiếm cho: {query}\n\n"

        # Add AI answer if available
        if results.get("answer"):
            output += f"Tóm tắt AI:\n{results['answer']}\n\n"

        # Add individual results
        output += "Chi tiết:\n"
        for idx, result in enumerate(results.get("results", []), 1):
            output += f"{idx}. {result['title']}\n"
            output += f"   URL: {result['url']}\n"
            output += f"   {result['content'][:200]}...\n"
            output += f"   Độ liên quan: {result.get('score', 0):.2f}\n\n"

        if not results.get("results"):
            output += "Không tìm thấy kết quả.\n"

        return output

    except Exception as e:
        logger.error(f"Error in tavily_search_legal: {e}")
        return f"Lỗi tìm kiếm: {str(e)}"


def tavily_qna(question: str) -> str:
    """
    Quick Q&A using Tavily - gets direct answer to a question.
    Useful for factual questions that need web search.

    Args:
        question: Question to answer

    Returns:
        Answer string
    """
    try:
        results = tavily_search(question, max_results=3, search_depth="advanced")

        if "error" in results:
            return f"Không thể trả lời: {results['error']}"

        # Return AI-generated answer if available
        if results.get("answer"):
            return results["answer"]

        # Otherwise summarize from results
        if results.get("results"):
            summary = "Dựa trên tìm kiếm web:\n"
            for result in results["results"][:2]:
                summary += f"- {result['content'][:150]}...\n"
            return summary

        return "Không tìm thấy thông tin."

    except Exception as e:
        logger.error(f"Error in tavily_qna: {e}")
        return f"Lỗi: {str(e)}"
