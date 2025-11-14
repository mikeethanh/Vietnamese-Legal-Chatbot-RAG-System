"""
Script để upload/download dữ liệu từ Digital Ocean Spaces
"""

import os
import boto3
import json
import logging
from pathlib import Path
from typing import Optional, List
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file will not be loaded.")
    print("Install with: pip install python-dotenv")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DOSpacesManager:
    def __init__(self, 
                 endpoint_url: str = None,
                 access_key: str = None, 
                 secret_key: str = None,
                 bucket_name: str = None):
        """
        Initialize Digital Ocean Spaces client
        
        Args:
            endpoint_url: DO Spaces endpoint (e.g., https://sgp1.digitaloceanspaces.com)
            access_key: DO Spaces access key
            secret_key: DO Spaces secret key  
            bucket_name: Bucket name (e.g., legal-datalake)
        """
        
        # Get credentials from environment if not provided
        self.endpoint_url = endpoint_url or os.getenv('DO_SPACES_ENDPOINT')
        self.access_key = access_key or os.getenv('DO_SPACES_KEY')
        self.secret_key = secret_key or os.getenv('DO_SPACES_SECRET')
        self.bucket_name = bucket_name or os.getenv('DO_SPACES_BUCKET', 'legal-datalake')
        
        if not all([self.endpoint_url, self.access_key, self.secret_key]):
            raise ValueError("Missing Digital Ocean Spaces credentials. Please set environment variables or pass parameters.")
        
        # Initialize S3 client (DO Spaces is S3-compatible)
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        logger.info(f"Initialized DO Spaces client for bucket: {self.bucket_name}")
    
    def upload_file(self, local_path: str, spaces_path: str, show_progress: bool = True) -> bool:
        """Upload file to DO Spaces"""
        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file not found: {local_path}")
                return False
            
            file_size = local_path.stat().st_size
            logger.info(f"Uploading {local_path} ({file_size / (1024*1024):.1f} MB) to {spaces_path}")
            
            if show_progress and file_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                # Upload with progress bar
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
                    def upload_callback(bytes_transferred):
                        pbar.update(bytes_transferred)
                    
                    self.s3_client.upload_file(
                        str(local_path), 
                        self.bucket_name, 
                        spaces_path,
                        Callback=upload_callback
                    )
            else:
                # Simple upload
                self.s3_client.upload_file(str(local_path), self.bucket_name, spaces_path)
            
            logger.info(f"✅ Upload completed: {spaces_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Upload failed: {e}")
            return False
    
    def download_file(self, spaces_path: str, local_path: str, show_progress: bool = True) -> bool:
        """Download file from DO Spaces"""
        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Get file size for progress bar
            try:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=spaces_path)
                file_size = response['ContentLength']
                logger.info(f"Downloading {spaces_path} ({file_size / (1024*1024):.1f} MB) to {local_path}")
            except ClientError:
                file_size = 0
                logger.info(f"Downloading {spaces_path} to {local_path}")
            
            if show_progress and file_size > 10 * 1024 * 1024:  # Show progress for files > 10MB
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    def download_callback(bytes_transferred):
                        pbar.update(bytes_transferred)
                    
                    self.s3_client.download_file(
                        self.bucket_name, 
                        spaces_path, 
                        str(local_path),
                        Callback=download_callback
                    )
            else:
                self.s3_client.download_file(self.bucket_name, spaces_path, str(local_path))
            
            logger.info(f"✅ Download completed: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def upload_directory(self, local_dir: str, spaces_prefix: str) -> bool:
        """Upload entire directory to DO Spaces"""
        try:
            local_dir = Path(local_dir)
            if not local_dir.exists():
                logger.error(f"Local directory not found: {local_dir}")
                return False
            
            success_count = 0
            total_count = 0
            
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    total_count += 1
                    relative_path = file_path.relative_to(local_dir)
                    spaces_path = f"{spaces_prefix.rstrip('/')}/{relative_path}"
                    
                    if self.upload_file(str(file_path), spaces_path, show_progress=False):
                        success_count += 1
            
            logger.info(f"✅ Directory upload completed: {success_count}/{total_count} files")
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"❌ Directory upload failed: {e}")
            return False
    
    def download_directory(self, spaces_prefix: str, local_dir: str) -> bool:
        """Download directory from DO Spaces"""
        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # List objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=spaces_prefix)
            
            success_count = 0
            total_count = 0
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_count += 1
                        spaces_path = obj['Key']
                        relative_path = spaces_path.replace(spaces_prefix.rstrip('/') + '/', '')
                        local_path = local_dir / relative_path
                        
                        if self.download_file(spaces_path, str(local_path), show_progress=False):
                            success_count += 1
            
            logger.info(f"✅ Directory download completed: {success_count}/{total_count} files")
            return success_count == total_count
            
        except Exception as e:
            logger.error(f"❌ Directory download failed: {e}")
            return False
    
    def list_objects(self, prefix: str = "") -> List[str]:
        """List objects in bucket with prefix"""
        try:
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        objects.append(obj['Key'])
            
            return objects
            
        except Exception as e:
            logger.error(f"❌ List objects failed: {e}")
            return []
    
    def delete_object(self, spaces_path: str) -> bool:
        """Delete object from DO Spaces"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=spaces_path)
            logger.info(f"✅ Deleted: {spaces_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            return False

def upload_training_data():
    """Upload processed training data to DO Spaces"""
    print("🔄 Uploading training data to Digital Ocean Spaces...")
    
    spaces_manager = DOSpacesManager()
    
    # Paths
    base_data_dir = Path("data_processing")
    spaces_base = "process_data/finetune_data"
    
    files_to_upload = [
        # Raw and processed data
        ("raw_data/finetune_llm_data.jsonl", f"{spaces_base}/raw/finetune_llm_data.jsonl"),
        ("processed_llama_data.jsonl", f"{spaces_base}/processed/processed_llama_data.jsonl"),
        
        # Analysis results
        ("data_analysis.json", f"{spaces_base}/analysis/data_analysis.json"),
        ("sample_processed_data.json", f"{spaces_base}/analysis/sample_processed_data.json"),
        
        # Train/val/test splits
        ("splits/train.jsonl", f"{spaces_base}/splits/train.jsonl"),
        ("splits/valid.jsonl", f"{spaces_base}/splits/valid.jsonl"),
        ("splits/test.jsonl", f"{spaces_base}/splits/test.jsonl"),
    ]
    
    success_count = 0
    for local_path, spaces_path in files_to_upload:
        full_local_path = base_data_dir / local_path
        if full_local_path.exists():
            if spaces_manager.upload_file(str(full_local_path), spaces_path):
                success_count += 1
        else:
            logger.warning(f"File not found: {full_local_path}")
    
    print(f"✅ Upload completed: {success_count}/{len(files_to_upload)} files")
    
    # List uploaded files
    print(f"\n📂 Files in {spaces_base}:")
    objects = spaces_manager.list_objects(spaces_base)
    for obj in objects:
        print(f"   - {obj}")

def download_training_data():
    """Download training data from DO Spaces"""
    print("⬇️ Downloading training data from Digital Ocean Spaces...")
    
    spaces_manager = DOSpacesManager()
    
    # Download splits for training
    spaces_base = "process_data/finetune_data"
    local_base = Path("data_processing")
    
    # Essential files (required for training)
    essential_files = [
        (f"{spaces_base}/splits/train.jsonl", "splits/train.jsonl"),
        (f"{spaces_base}/splits/test.jsonl", "splits/test.jsonl"),
    ]
    
    # Optional files (nice to have but not required)
    optional_files = [
        (f"{spaces_base}/splits/valid.jsonl", "splits/valid.jsonl"),
        (f"{spaces_base}/splits/val.jsonl", "splits/val.jsonl"),  # Fallback for old naming
    ]
    
    success_count = 0
    total_files = len(essential_files) + len(optional_files)
    
    # Download essential files
    for spaces_path, local_path in essential_files:
        full_local_path = local_base / local_path
        if spaces_manager.download_file(spaces_path, str(full_local_path)):
            success_count += 1
        else:
            logger.error(f"❌ Failed to download essential file: {spaces_path}")
    
    # Download optional files (don't fail if not found)
    for spaces_path, local_path in optional_files:
        full_local_path = local_base / local_path
        if spaces_manager.download_file(spaces_path, str(full_local_path)):
            success_count += 1
            break  # Only need one validation file
    
    print(f"✅ Download completed: {success_count}/{total_files} files")

def upload_model(model_dir: str, model_name: str = "vietnamese-legal-llama"):
    """Upload trained model to DO Spaces"""
    print(f"🔄 Uploading model {model_name} to Digital Ocean Spaces...")
    
    spaces_manager = DOSpacesManager()
    spaces_prefix = f"models/{model_name}"
    
    if spaces_manager.upload_directory(model_dir, spaces_prefix):
        print(f"✅ Model uploaded successfully to: {spaces_prefix}")
        
        # List uploaded model files
        print(f"\n📂 Model files:")
        objects = spaces_manager.list_objects(spaces_prefix)
        for obj in objects:
            print(f"   - {obj}")
    else:
        print(f"❌ Model upload failed")

def download_model(model_name: str = "vietnamese-legal-llama", local_dir: str = "./model"):
    """Download trained model from DO Spaces"""
    print(f"⬇️ Downloading model {model_name} from Digital Ocean Spaces...")
    
    spaces_manager = DOSpacesManager()
    spaces_prefix = f"models/{model_name}"
    
    if spaces_manager.download_directory(spaces_prefix, local_dir):
        print(f"✅ Model downloaded successfully to: {local_dir}")
    else:
        print(f"❌ Model download failed")

def main():
    """Main function for command line usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python do_spaces_manager.py upload-data     # Upload training data")
        print("  python do_spaces_manager.py download-data   # Download training data")
        print("  python do_spaces_manager.py upload-model <model_dir> [model_name]")
        print("  python do_spaces_manager.py download-model [model_name] [local_dir]")
        print("  python do_spaces_manager.py list [prefix]   # List objects")
        return
    
    command = sys.argv[1]
    
    try:
        if command == "upload-data":
            upload_training_data()
            
        elif command == "download-data":
            download_training_data()
            
        elif command == "upload-model":
            if len(sys.argv) < 3:
                print("Error: Model directory required")
                return
            model_dir = sys.argv[2]
            model_name = sys.argv[3] if len(sys.argv) > 3 else "vietnamese-legal-llama"
            upload_model(model_dir, model_name)
            
        elif command == "download-model":
            model_name = sys.argv[2] if len(sys.argv) > 2 else "vietnamese-legal-llama"
            local_dir = sys.argv[3] if len(sys.argv) > 3 else "./model"
            download_model(model_name, local_dir)
            
        elif command == "list":
            prefix = sys.argv[2] if len(sys.argv) > 2 else ""
            spaces_manager = DOSpacesManager()
            objects = spaces_manager.list_objects(prefix)
            print(f"📂 Objects in bucket (prefix: {prefix}):")
            for obj in objects:
                print(f"   - {obj}")
                
        else:
            print(f"Unknown command: {command}")
            
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")

if __name__ == "__main__":
    main()