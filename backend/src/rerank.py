import os
from time import time
from typing import List

import cohere
import numpy as np
import requests

# Set up your cohere client
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
DEFAULT_RANK_MODEL = "rerank-multilingual-v3.0"
co = cohere.Client(COHERE_API_KEY)


def rerank_documents(docs, query, top_n=3, rank_model=DEFAULT_RANK_MODEL):
    """
    Rerank documents based on the query
    """
    # Kiểm tra nếu docs trống hoặc query trống thì return luôn
    if not docs:
        print("[RERANK] Docs list is empty, skipping rerank")
        return []

    if not query or not query.strip():
        print("[RERANK] Query is empty, returning original docs without rerank")
        return docs[:top_n]

    # Kiểm tra API key
    if not COHERE_API_KEY:
        print("[RERANK] COHERE_API_KEY not found, returning original docs")
        return docs[:top_n]

    try:
        # Tạo process_docs và lọc ra documents rỗng
        process_docs = []
        valid_doc_indices = []

        for idx, doc in enumerate(docs):
            title = doc.get("title", "") or ""
            content = doc.get("content", "") or ""
            combined = f"{title} {content}".strip()

            if combined:  # Chỉ thêm nếu có nội dung
                process_docs.append(combined)
                valid_doc_indices.append(idx)

        # Nếu không có document hợp lệ nào
        if not process_docs:
            print("[RERANK] No valid documents to rerank")
            return docs[:top_n]

        print(
            f"[RERANK] Reranking {len(process_docs)} documents with query: '{query[:50]}...'"
        )

        results = co.rerank(
            query=query,
            documents=process_docs,
            top_n=min(top_n, len(process_docs)),
            model=rank_model,
        )

        # Map lại index từ process_docs về docs gốc
        ranked_docs = []
        for item in results.results:
            original_idx = valid_doc_indices[item.index]
            doc = docs[original_idx].copy()
            doc["relevance_score"] = item.relevance_score
            ranked_docs.append(doc)
            print(
                f"[RERANK] Doc {original_idx}: {doc.get('title', 'No title')[:50]} - Score: {item.relevance_score:.5f}"
            )

        return ranked_docs

    except Exception as e:
        print(f"[RERANK] Error during reranking: {e}")
        print(f"[RERANK] Returning original docs without rerank")
        return docs[:top_n]
