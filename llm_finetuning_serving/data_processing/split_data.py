#!/usr/bin/env python3
"""
Script to split finetune dataset into train/test/valid splits
and upload to Digital Ocean bucket and save locally.
"""

import json
import os
import random
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import boto3
from botocore.config import Config
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(self, 
                 input_file: str,
                 output_dir: str = "splits",
                 train_ratio: float = 0.8,
                 valid_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42):
        """
        Initialize DataSplitter
        
        Args:
            input_file: Path to input JSONL file
            output_dir: Directory to save split files
            train_ratio: Ratio for training set
            valid_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # Validate ratios
        if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Train, valid, and test ratios must sum to 1.0")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        random.seed(random_seed)
        
        logger.info(f"DataSplitter initialized:")
        logger.info(f"  Input file: {input_file}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Ratios - Train: {train_ratio}, Valid: {valid_ratio}, Test: {test_ratio}")
    
    def load_data(self) -> List[Dict]:
        """Load data from JSONL file"""
        logger.info(f"Loading data from {self.input_file}")
        data = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading data"), 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/valid/test sets"""
        logger.info("Splitting data...")
        
        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        total_samples = len(shuffled_data)
        train_size = int(total_samples * self.train_ratio)
        valid_size = int(total_samples * self.valid_ratio)
        
        # Split data
        train_data = shuffled_data[:train_size]
        valid_data = shuffled_data[train_size:train_size + valid_size]
        test_data = shuffled_data[train_size + valid_size:]
        
        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(train_data)} samples ({len(train_data)/total_samples:.1%})")
        logger.info(f"  Valid: {len(valid_data)} samples ({len(valid_data)/total_samples:.1%})")
        logger.info(f"  Test: {len(test_data)} samples ({len(test_data)/total_samples:.1%})")
        
        return train_data, valid_data, test_data
    
    def save_split_files(self, train_data: List[Dict], valid_data: List[Dict], test_data: List[Dict]):
        """Save split data to JSONL files"""
        logger.info("Saving split files...")
        
        splits = {
            'train': train_data,
            'valid': valid_data,
            'test': test_data
        }
        
        file_paths = {}
        
        for split_name, split_data in splits.items():
            file_path = self.output_dir / f"{split_name}.jsonl"
            file_paths[split_name] = str(file_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in tqdm(split_data, desc=f"Saving {split_name}"):
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {split_name} split to {file_path} ({len(split_data)} samples)")
        
        return file_paths
    
    def run_split(self) -> Dict[str, str]:
        """Run the complete splitting process"""
        logger.info("Starting data splitting process...")
        
        # Load data
        data = self.load_data()
        
        # Split data
        train_data, valid_data, test_data = self.split_data(data)
        
        # Save split files
        file_paths = self.save_split_files(train_data, valid_data, test_data)
        
        logger.info("Data splitting completed successfully!")
        return file_paths


class DOUploader:
    def __init__(self, 
                 access_key: str = None,
                 secret_key: str = None,
                 endpoint_url: str = "https://sfo3.digitaloceanspaces.com",
                 bucket_name: str = "legal-datalake"):
        """
        Initialize Digital Ocean Spaces uploader
        
        Args:
            access_key: DO Spaces access key (if None, will try environment variables)
            secret_key: DO Spaces secret key (if None, will try environment variables)
            endpoint_url: DO Spaces endpoint URL
            bucket_name: Bucket name
        """
        self.bucket_name = bucket_name
        
        # Get credentials from environment if not provided
        self.access_key = access_key or os.getenv('DO_SPACES_KEY')
        self.secret_key = secret_key or os.getenv('DO_SPACES_SECRET')
        
        if not self.access_key or not self.secret_key:
            logger.warning("Digital Ocean credentials not found. Please set DO_SPACES_ACCESS_KEY and DO_SPACES_SECRET_KEY environment variables")
            self.client = None
            return
        
        # Initialize DO Spaces client
        try:
            session = boto3.session.Session()
            self.client = session.client('s3',
                                       config=Config(s3={'addressing_style': 'virtual'}),
                                       region_name='sfo3',
                                       endpoint_url=endpoint_url,
                                       aws_access_key_id=self.access_key,
                                       aws_secret_access_key=self.secret_key)
            logger.info(f"Digital Ocean Spaces client initialized for bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize DO Spaces client: {e}")
            self.client = None
    
    def upload_file(self, local_file_path: str, remote_key: str):
        """Upload a file to Digital Ocean Spaces"""
        if not self.client:
            logger.warning("DO Spaces client not available. Skipping upload.")
            return False
        
        try:
            file_size = os.path.getsize(local_file_path)
            logger.info(f"Uploading {local_file_path} to s3://{self.bucket_name}/{remote_key} ({file_size / 1024 / 1024:.1f} MB)")
            
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {Path(local_file_path).name}") as pbar:
                def upload_callback(bytes_transferred):
                    pbar.update(bytes_transferred)
                
                self.client.upload_file(
                    local_file_path, 
                    self.bucket_name, 
                    remote_key,
                    Callback=upload_callback
                )
            
            logger.info(f"Successfully uploaded {remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path}: {e}")
            return False
    
    def upload_splits(self, file_paths: Dict[str, str], base_remote_path: str = "process_data/finetune_data/splits"):
        """Upload all split files to DO Spaces"""
        if not self.client:
            logger.warning("DO Spaces client not available. Skipping all uploads.")
            return
        
        logger.info(f"Uploading splits to {base_remote_path}/")
        
        for split_name, file_path in file_paths.items():
            remote_key = f"{base_remote_path}/{split_name}.jsonl"
            self.upload_file(file_path, remote_key)


def main():
    parser = argparse.ArgumentParser(description="Split finetune dataset and upload to DO Spaces")
    parser.add_argument("--input-file", 
                       default="/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/data_pipeline/data/finetune_llm/finetune_llm_data.jsonl",
                       help="Path to input JSONL file")
    parser.add_argument("--output-dir", 
                       default="splits",
                       help="Directory to save split files")
    parser.add_argument("--train-ratio", 
                       type=float, 
                       default=0.8,
                       help="Training set ratio")
    parser.add_argument("--valid-ratio", 
                       type=float, 
                       default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--test-ratio", 
                       type=float, 
                       default=0.1,
                       help="Test set ratio")
    parser.add_argument("--random-seed", 
                       type=int, 
                       default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--skip-upload", 
                       action="store_true",
                       help="Skip uploading to Digital Ocean Spaces")
    parser.add_argument("--bucket-name", 
                       default="legal-datalake",
                       help="Digital Ocean Spaces bucket name")
    parser.add_argument("--remote-path", 
                       default="process_data/finetune_data/splits",
                       help="Remote path in bucket")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return
    
    # Initialize data splitter
    splitter = DataSplitter(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    # Run splitting
    file_paths = splitter.run_split()
    
    # Upload to Digital Ocean Spaces if not skipped
    if not args.skip_upload:
        uploader = DOUploader(bucket_name=args.bucket_name)
        uploader.upload_splits(file_paths, args.remote_path)
    else:
        logger.info("Skipping upload to Digital Ocean Spaces")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    logger.info(f"Local files saved to: {args.output_dir}/")
    for split_name, file_path in file_paths.items():
        file_size = os.path.getsize(file_path) / 1024 / 1024
        logger.info(f"  {split_name}.jsonl: {file_size:.1f} MB")
    
    if not args.skip_upload:
        logger.info(f"\nRemote files uploaded to: s3://{args.bucket_name}/{args.remote_path}/")
        for split_name in file_paths.keys():
            logger.info(f"  {split_name}.jsonl")


if __name__ == "__main__":
    main()