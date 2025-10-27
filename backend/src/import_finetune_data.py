"""
Script to import finetune data from JSONL file into Qdrant vector database.

This script reads the combined_finetune_data.jsonl file and indexes each entry
into Qdrant using the index_document_v2 function.

Usage:
    python import_finetune_data.py --file <path_to_jsonl> --collection <collection_name> --batch-size <batch_size>

Example:
    python import_finetune_data.py --file ../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl
"""

import json
import logging
import argparse
from pathlib import Path
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from tasks import index_document_v2
from configs import DEFAULT_COLLECTION_NAME
from utils import setup_logging
from vectorize import create_collection

setup_logging()
logger = logging.getLogger(__name__)


def read_jsonl_file(file_path: str):
    """
    Generator function to read JSONL file line by line.
    
    Args:
        file_path: Path to the JSONL file
        
    Yields:
        Dictionary containing the parsed JSON data from each line
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON at line {line_num}: {e}")
                continue


def import_finetune_data(
    file_path: str,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 100,
    start_from: int = 0,
    max_records: int = None
):
    """
    Import finetune data from JSONL file into Qdrant.
    
    Args:
        file_path: Path to the combined_finetune_data.jsonl file
        collection_name: Name of the Qdrant collection to use
        batch_size: Number of records to process before showing progress
        start_from: Skip first N records (useful for resuming)
        max_records: Maximum number of records to import (None = all)
    """
    # Validate file exists
    if not Path(file_path).exists():
        logger.error(f"File not found: {file_path}")
        return
    
    # Ensure collection exists
    try:
        logger.info(f"Checking collection: {collection_name}")
        create_collection(collection_name)
        logger.info(f"Collection {collection_name} is ready")
    except Exception as e:
        if "already exists" in str(e):
            logger.info(f"Collection {collection_name} already exists")
        else:
            logger.error(f"Error creating collection: {e}")
            return
    
    # Count total lines for progress bar
    logger.info("Counting total records...")
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    
    logger.info(f"Total records in file: {total_lines}")
    
    # Determine how many to process
    records_to_process = total_lines - start_from
    if max_records:
        records_to_process = min(records_to_process, max_records)
    
    logger.info(f"Starting import from record {start_from}, will process {records_to_process} records")
    
    # Import data
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    # Use tqdm if available, otherwise use simple counter
    if HAS_TQDM:
        pbar = tqdm(total=records_to_process, desc="Importing")
    else:
        pbar = None
        logger.info("Progress will be logged every 100 records (install tqdm for progress bar)")
    
    try:
        for idx, data in enumerate(read_jsonl_file(file_path)):
            # Skip records before start_from
            if idx < start_from:
                continue
            
            # Stop if reached max_records
            if max_records and (idx - start_from) >= max_records:
                break
            
            try:
                # Extract fields
                question = data.get('question', '').strip()
                context = data.get('context', '').strip()
                record_id = data.get('id', f'record_{idx}')
                
                # Validate required fields
                if not question or not context:
                    logger.warning(f"Skipping record {idx} - missing question or context")
                    skipped_count += 1
                    if pbar:
                        pbar.update(1)
                    continue
                
                # Index document using existing function
                # title = question, content = context
                index_document_v2(
                    id=record_id,
                    title=question,
                    content=context,
                    collection_name=collection_name
                )
                
                success_count += 1
                
                # Update progress
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': success_count,
                        'errors': error_count,
                        'skipped': skipped_count
                    })
                elif success_count % 100 == 0:
                    logger.info(f"Progress: {success_count} records imported, {error_count} errors, {skipped_count} skipped")
                
                # Small delay to avoid overwhelming the system
                if success_count % batch_size == 0:
                    time.sleep(0.1)
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing record {idx}: {e}")
                if pbar:
                    pbar.update(1)
                continue
    finally:
        if pbar:
            pbar.close()
    
    # Summary
    logger.info("=" * 60)
    logger.info("Import Summary:")
    logger.info(f"  Total processed: {success_count + error_count + skipped_count}")
    logger.info(f"  Successfully imported: {success_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Skipped (missing data): {skipped_count}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Import finetune data from JSONL into Qdrant vector database'
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to the combined_finetune_data.jsonl file'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f'Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Batch size for progress updates (default: 100)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start from record N (useful for resuming, default: 0)'
    )
    parser.add_argument(
        '--max-records',
        type=int,
        default=None,
        help='Maximum number of records to import (default: all)'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting import process...")
    logger.info(f"File: {args.file}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Batch size: {args.batch_size}")
    
    import_finetune_data(
        file_path=args.file,
        collection_name=args.collection,
        batch_size=args.batch_size,
        start_from=args.start_from,
        max_records=args.max_records
    )
    
    logger.info("Import process completed!")


if __name__ == "__main__":
    main()
