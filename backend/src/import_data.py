"""
Script to import data from train_qa_format.jsonl into Qdrant vector database
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from brain import get_embedding
from configs import DEFAULT_COLLECTION_NAME
from search import initialize_search_index
from splitter import split_document
from utils import setup_logging
from vectorize import add_vector, create_collection

setup_logging()
logger = logging.getLogger(__name__)

# Path to the data file
DATA_FILE_PATH = "/usr/src/app/data/train.jsonl"


def import_qa_data(
    data_file_path=DATA_FILE_PATH,
    collection_name=DEFAULT_COLLECTION_NAME,
    batch_size=50,
    limit=None,
):
    """
    Import Q&A data from JSONL file into Qdrant vector database

    Args:
        data_file_path: Path to the train.jsonl file (RAG format)
        collection_name: Name of the Qdrant collection
        batch_size: Number of vectors to process in each batch
        limit: Maximum number of records to process (None for all)
    """

    # Check if file exists
    if not os.path.exists(data_file_path):
        logger.error(f"❌ Data file not found: {data_file_path}")
        return False

    logger.info(f"✅ Data file found: {data_file_path}")
    
    # Get file size for progress tracking
    file_size = os.path.getsize(data_file_path)
    logger.info(f"📊 File size: {file_size / (1024*1024):.2f} MB")

    logger.info(
        f"🚀 Starting import from {data_file_path} to collection {collection_name}"
    )
    if limit:
        logger.info(f"Limiting import to {limit} records")

    # Try to create collection (will fail if already exists, which is fine)
    try:
        create_collection(collection_name)
        logger.info(f"✅ Created collection: {collection_name}")
    except Exception as e:
        logger.info(f"📋 Collection {collection_name} might already exist: {e}")

    logger.info("🔄 Starting to read JSONL file...")
    
    # Read and process the JSONL file
    success_count = 0
    error_count = 0
    vectors_batch = {}  # Collect vectors for batch processing
    total_vectors_processed = 0  # Track total vectors processed
    documents_for_search = []  # Collect documents for search index

    logger.info(f"📖 Opening file: {data_file_path}")
    
    with open(data_file_path, "r", encoding="utf-8") as f:
        logger.info("📄 File opened successfully, starting line-by-line processing...")
        
        for idx, line in enumerate(f):
            # Add progress logging every 50 lines
            if idx % 50 == 0:
                logger.info(f"📊 Processing line {idx + 1}...")
                
            # Check limit
            if limit and idx >= limit:
                logger.info(f"🛑 Reached limit of {limit} records, stopping")
                break

            try:
                # Parse JSON line
                data = json.loads(line.strip())
                question = data.get("question", "")
                context = data.get("context", "")  # Note: "context" thay vì "answer"

                if not question or not context:
                    logger.warning(
                        f"⚠️ Line {idx + 1}: Missing question or context, skipping"
                    )
                    error_count += 1
                    continue

                # Debug first few records
                if idx < 3:
                    logger.info(f"📝 Sample record {idx + 1}: Q='{question[:50]}...', C='{context[:50]}...'")

                # Store document for search index
                documents_for_search.append({
                    "question": question,
                    "content": context,
                    "source": "train",
                    "doc_id": idx
                })

                # Combine question and context for embedding
                text = f"{question} {context}"

                # Split document into chunks if needed
                nodes = split_document(text)

                # Process each chunk
                for chunk_idx, node in enumerate(nodes):
                    # Generate unique ID for this chunk (must be integer for Qdrant)
                    # Using formula: idx * 1000 + chunk_idx to ensure uniqueness
                    point_id = idx * 1000 + chunk_idx

                    # Get embedding
                    vector = get_embedding(node.text)

                    # Add to batch
                    vectors_batch[point_id] = {
                        "vector": vector,
                        "payload": {
                            "question": question,
                            "content": node.text,
                            "source": "train",  # Fixed source name
                            "doc_id": idx,
                        },
                    }

                success_count += 1

                # Process batch when it reaches batch_size
                if len(vectors_batch) >= batch_size:
                    logger.info(f"🔄 Processing batch of {len(vectors_batch)} vectors...")
                    add_vector(
                        collection_name=collection_name,
                        vectors=vectors_batch,
                        batch_size=batch_size,
                    )
                    total_vectors_processed += len(vectors_batch)
                    vectors_batch = {}  # Reset batch
                    
                    logger.info(
                        f"✅ Processed {total_vectors_processed} vectors total - Records: {success_count}, Errors: {error_count}"
                    )

            except json.JSONDecodeError as e:
                logger.error(f"Line {idx + 1}: JSON decode error - {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Line {idx + 1}: Unexpected error - {e}")
                error_count += 1

    # Process remaining vectors in the final batch
    if vectors_batch:
        logger.info(f"🔄 Processing final batch of {len(vectors_batch)} vectors...")
        add_vector(
            collection_name=collection_name,
            vectors=vectors_batch,
            batch_size=batch_size,
        )
        total_vectors_processed += len(vectors_batch)

    logger.info("🎯 REACHED END OF PROCESSING LOOP!")
    logger.info(
        f"📊 Import completed! Total vectors: {total_vectors_processed}, Records: {success_count}, Errors: {error_count}"
    )

    # Initialize search index with collected documents
    if documents_for_search:
        logger.info(f"🔍 STARTING SEARCH INDEX INITIALIZATION with {len(documents_for_search)} documents...")
        try:
            success = initialize_search_index(documents_for_search)
            if success:
                logger.info("✅ Search index initialized successfully!")
            else:
                logger.error("❌ Search index initialization returned False")
        except Exception as e:
            logger.error(f"❌ Failed to initialize search index: {e}")
            logger.exception("Full error traceback:")
    else:
        logger.warning("⚠️ No documents collected for search index!")

    logger.info("🏁 IMPORT FUNCTION COMPLETED!")
    return True


def main():
    """Main function to run the import"""
    import argparse

    parser = argparse.ArgumentParser(description="Import Q&A data into Qdrant")
    parser.add_argument(
        "--data-file",
        type=str,
        default=DATA_FILE_PATH,
        help="Path to train.jsonl file (RAG format)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=50, 
        help="Batch size for vector processing"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of records to process (for testing)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting RAG Data Import")
    logger.info("=" * 60)
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Limit: {args.limit or 'No limit'}")
    logger.info("=" * 60)

    success = import_qa_data(
        data_file_path=args.data_file,
        collection_name=args.collection,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    if success:
        logger.info("Import completed successfully!")
        sys.exit(0)
    else:
        logger.error("Import failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
