import logging
import os

from custom_embedding import get_custom_embedding
from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, TokenTextSplitter

logger = logging.getLogger(__name__)


# Custom embedding wrapper for Vietnamese legal model
class CustomEmbeddingWrapper(BaseEmbedding):
    """Wrapper for custom embeddings to use with SemanticSplitter"""

    def __init__(self):
        super().__init__()

    def _get_query_embedding(self, query: str):
        """Get embedding for query"""
        text = query.replace("\n", " ")
        return get_custom_embedding(text)

    def _get_text_embedding(self, text: str):
        """Get embedding for text"""
        text = text.replace("\n", " ")
        return get_custom_embedding(text)

    async def _aget_query_embedding(self, query: str):
        """Async get embedding for query"""
        return self._get_query_embedding(query)


def split_document(text, metadata={"course": "LLM"}, use_semantic=True):
    """
    Split document into chunks using semantic splitting for better context preservation.

    Args:
        text: The document text to split
        metadata: Metadata to attach to the document
        use_semantic: Whether to use semantic splitting (recommended for legal documents)

    Returns:
        List of nodes with semantic-aware boundaries
    """
    doc = Document(text=text, metadata=metadata)

    if use_semantic:
        # Use semantic splitter with custom Vietnamese legal embeddings
        # This ensures chunks break at semantic boundaries rather than arbitrary tokens
        embed_model = CustomEmbeddingWrapper()
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,  # Number of sentences to group together when evaluating semantic similarity
            breakpoint_percentile_threshold=95,  # Threshold for semantic similarity to create a split
            embed_model=embed_model,
        )
        logger.info(
            "Using semantic splitter with custom embedding model for document chunking"
        )
    else:
        # Fallback to token-based splitting
        splitter = TokenTextSplitter(
            chunk_size=512,  # Increased for better context
            chunk_overlap=50,
            separator=" ",
        )
        logger.info("Using token splitter for document chunking")

    nodes = splitter.get_nodes_from_documents([doc])
    logger.info(f"Split document into {len(nodes)} chunks")
    return nodes
