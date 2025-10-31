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
from vectorize import add_vector, create_collection
from configs import DEFAULT_COLLECTION_NAME
from splitter import split_document
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Path to the data file
DATA_FILE_PATH = "/usr/src/app/../data_pipeline/data/finetune_data/train_qa_format.jsonl"


def import_qa_data(data_file_path=DATA_FILE_PATH, collection_name=DEFAULT_COLLECTION_NAME, batch_size=100):
    """
    Import Q&A data from JSONL file into Qdrant vector database
    
    Args:
        data_file_path: Path to the train_qa_format.jsonl file
        collection_name: Name of the Qdrant collection
        batch_size: Number of records to process before logging progress
    """
    
    # Check if file exists
    if not os.path.exists(data_file_path):
        logger.error(f"Data file not found: {data_file_path}")
        return False
    
    logger.info(f"Starting import from {data_file_path} to collection {collection_name}")
    
    # Try to create collection (will fail if already exists, which is fine)
    try:
        create_collection(collection_name)
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.info(f"Collection {collection_name} might already exist: {e}")
    
    # Read and process the JSONL file
    success_count = 0
    error_count = 0
    
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                question = data.get('question', '')
                answer = data.get('answer', '')
                
                if not question or not answer:
                    logger.warning(f"Line {idx + 1}: Missing question or answer, skipping")
                    error_count += 1
                    continue
                
                # Combine question and answer for embedding
                text = f"{question} {answer}"
                
                # Split document into chunks if needed
                nodes = split_document(text)
                
                # Process each chunk
                for chunk_idx, node in enumerate(nodes):
                    # Generate unique ID for this chunk (must be integer for Qdrant)
                    # Using formula: idx * 1000 + chunk_idx to ensure uniqueness
                    point_id = idx * 1000 + chunk_idx
                    
                    # Get embedding
                    vector = get_embedding(node.text)
                    
                    # Add to Qdrant
                    add_vector(
                        collection_name=collection_name,
                        vectors={
                            point_id: {
                                "vector": vector,
                                "payload": {
                                    "question": question,
                                    "content": node.text,
                                    "source": "train_qa_format",
                                    "doc_id": idx
                                }
                            }
                        }
                    )
                
                success_count += 1
                
                # Log progress every batch_size records
                if (idx + 1) % batch_size == 0:
                    logger.info(f"Processed {idx + 1} records - Success: {success_count}, Errors: {error_count}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Line {idx + 1}: JSON decode error - {e}")
                error_count += 1
            except Exception as e:
                logger.error(f"Line {idx + 1}: Unexpected error - {e}")
                error_count += 1
    
    logger.info(f"Import completed! Total success: {success_count}, Total errors: {error_count}")
    return True


def main():
    """Main function to run the import"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import Q&A data into Qdrant')
    parser.add_argument('--data-file', type=str, default=DATA_FILE_PATH,
                        help='Path to train_qa_format.jsonl file')
    parser.add_argument('--collection', type=str, default=DEFAULT_COLLECTION_NAME,
                        help='Qdrant collection name')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size for logging progress')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Starting Q&A Data Import")
    logger.info("="*60)
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60)
    
    success = import_qa_data(
        data_file_path=args.data_file,
        collection_name=args.collection,
        batch_size=args.batch_size
    )
    
    if success:
        logger.info("Import completed successfully!")
        sys.exit(0)
    else:
        logger.error("Import failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
