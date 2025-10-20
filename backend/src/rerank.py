import cohere
import requests
import numpy as np
import os
from time import time
from typing import List


# Set up your cohere client
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
DEFAULT_RANK_MODEL = 'rerank-multilingual-v3.0'
co = cohere.Client(COHERE_API_KEY)


def rerank_documents(docs, query, top_n=3, rank_model=DEFAULT_RANK_MODEL):
    """
    Rerank documents based on the query
    """
    process_docs = [
        doc['title'] + ' ' + doc['content']
        for doc in docs
    ]
    results = co.rerank(
        query=query,
        documents=process_docs,
        top_n=top_n,
        model=rank_model
    )
    for item in results.results:
        print(f"Document Index: {item.index}")
        print(f"Document: {docs[item.index]}")
        print(f"Relevance Score: {item.relevance_score:.5f}")

    ranked_docs = [docs[item.index] for item in results.results]

    return ranked_docs