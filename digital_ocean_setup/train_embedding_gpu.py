#!/usr/bin/env python3
"""
Simple and robust GPU training script for Vietnamese Legal documents
Optimized for stability and ease of use
"""

import os
import json
import logging
import time
import torch
import boto3
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available!")
        exit(1)
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🚀 GPU: {gpu_name}")
    logger.info(f"🚀 GPU Memory: {gpu_memory:.1f} GB")
    
    return device

def setup_spaces_client():
    """Setup Digital Ocean Spaces client"""
    access_key = os.getenv('SPACES_ACCESS_KEY')
    secret_key = os.getenv('SPACES_SECRET_KEY')
    endpoint = os.getenv('SPACES_ENDPOINT', 'https://sfo3.digitaloceanspaces.com')
    
    if not access_key or not secret_key:
        logger.error("❌ SPACES_ACCESS_KEY and SPACES_SECRET_KEY required!")
        exit(1)
    
    region = 'sgp1' if 'sgp1' in endpoint else 'sfo3'
    
    client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        region_name=region
    )
    
    logger.info(f"✅ Connected to Spaces: {endpoint}")
    return client

def download_data(spaces_client, bucket_name):
    """Download training data from Spaces"""
    data_path = '/tmp/data/merged_corpus.jsonl'
    os.makedirs('/tmp/data', exist_ok=True)
    
    try:
        logger.info("📥 Downloading training data...")
        spaces_client.download_file(
            bucket_name,
            'process_data/rag_corpus/merged_corpus.jsonl',
            data_path
        )
        logger.info(f"✅ Downloaded to {data_path}")
        return data_path
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        exit(1)

def load_texts(data_path, max_samples=None):
    """Load texts from JSONL file"""
    logger.info("📚 Loading texts...")
    
    texts = []
    text_key_found = None
    skipped_count = 0
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
                
                # Tự động detect key name từ sample đầu tiên
                if text_key_found is None:
                    if 'text' in data:
                        text_key_found = 'text'
                        logger.info("📄 Detected data format: using 'text' key")
                    elif 'content' in data:
                        text_key_found = 'content'
                        logger.info("📄 Detected data format: using 'content' key")
                    else:
                        logger.warning(f"⚠️ No 'text' or 'content' key found. Available keys: {list(data.keys())}")
                        continue
                
                # Lấy text content
                text_content = data.get(text_key_found, '')
                if text_content and len(text_content.strip()) > 20:
                    texts.append(text_content.strip())
                else:
                    skipped_count += 1
                    
            except json.JSONDecodeError as e:
                skipped_count += 1
                if i < 5:  # Log first few decode errors
                    logger.warning(f"⚠️ JSON decode error at line {i}: {e}")
                continue
    
    logger.info(f"✅ Loaded {len(texts)} texts (skipped {skipped_count} invalid entries)")
    if len(texts) == 0:
        logger.error("❌ No valid texts found! Check your data format.")
        exit(1)
        
    # Log sample text
    if texts:
        sample_text = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
        logger.info(f"📝 Sample text: {sample_text}")
    
    return texts

def create_training_examples(texts):
    """Create training examples from texts"""
    logger.info("🔧 Creating training examples...")
    
    examples = []
    
    # Positive pairs (sequential texts - similar)
    for i in range(0, len(texts) - 1, 2):
        if i + 1 < len(texts):
            examples.append(InputExample(
                texts=[texts[i], texts[i + 1]], 
                label=0.8
            ))
    
    # Negative pairs (random texts - dissimilar)
    num_negative = min(len(examples), 1000)
    for _ in range(num_negative):
        idx1, idx2 = random.sample(range(len(texts)), 2)
        examples.append(InputExample(
            texts=[texts[idx1], texts[idx2]], 
            label=0.2
        ))
    
    random.shuffle(examples)
    logger.info(f"✅ Created {len(examples)} training examples")
    
    return examples

def train_model(model_name, examples, device, epochs=3, batch_size=16):
    """Train the embedding model"""
    logger.info(f"🤖 Loading model: {model_name}")
    
    # Load model
    try:
        model = SentenceTransformer(model_name)
        model.to(device)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        exit(1)
    
    # Split data
    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=42
    )
    
    logger.info(f"📊 Training examples: {len(train_examples)}")
    logger.info(f"📊 Validation examples: {len(val_examples)}")
    
    # DataLoader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size
    )
    
    # Loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Evaluator (small set for speed)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples[:50], name='legal-eval'
    )
    
    # Training
    logger.info(f"🔥 Starting training...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    
    start_time = time.time()
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=len(train_dataloader) // 2,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path='/tmp/model',
        save_best_model=True,
        show_progress_bar=True
    )
    
    training_time = time.time() - start_time
    logger.info(f"⏱️ Training completed in {training_time:.1f} seconds")
    
    return model

def upload_model(spaces_client, bucket_name, model_name):
    """Upload trained model to Spaces"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_prefix = f"models/embedding_model_gpu_{timestamp}"
    
    logger.info(f"📤 Uploading model to {s3_prefix}...")
    
    model_dir = '/tmp/model'
    uploaded_files = []
    
    # Upload all files
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, model_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            
            try:
                spaces_client.upload_file(local_path, bucket_name, s3_key)
                uploaded_files.append(relative_path)
                logger.info(f"✅ Uploaded {relative_path}")
            except Exception as e:
                logger.error(f"❌ Upload failed {relative_path}: {e}")
    
    # Create and upload metadata
    metadata = {
        "model_name": f"embedding_model_gpu_{timestamp}",
        "base_model": model_name,
        "training_date": timestamp,
        "gpu_used": torch.cuda.get_device_name(),
        "model_path": s3_prefix,
        "uploaded_files": uploaded_files
    }
    
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    spaces_client.upload_file(
        metadata_path, bucket_name, f"{s3_prefix}/metadata.json"
    )
    
    logger.info(f"🎉 Model uploaded successfully!")
    logger.info(f"📍 Model path: {s3_prefix}")
    
    return s3_prefix

def main():
    """Main training function"""
    logger.info("🚀 Starting Vietnamese Legal Embedding Training")
    
    # Configuration from environment
    model_name = os.getenv('BASE_MODEL', 'BAAI/bge-m3')
    epochs = int(os.getenv('EPOCHS', '3'))
    batch_size = int(os.getenv('GPU_BATCH_SIZE', '16'))
    max_samples = int(os.getenv('MAX_SAMPLES', '10000')) if os.getenv('MAX_SAMPLES') else None
    bucket_name = os.getenv('SPACES_BUCKET', 'legal-datalake')
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Bucket: {bucket_name}")
    
    try:
        # Setup
        device = check_gpu()
        spaces_client = setup_spaces_client()
        
        # Create directories
        os.makedirs('/tmp/data', exist_ok=True)
        os.makedirs('/tmp/model', exist_ok=True)
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # Download data
        data_path = download_data(spaces_client, bucket_name)
        
        # Load and prepare data
        texts = load_texts(data_path, max_samples)
        examples = create_training_examples(texts)
        
        # Train model
        model = train_model(model_name, examples, device, epochs, batch_size)
        
        # Upload model
        model_path = upload_model(spaces_client, bucket_name, model_name)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📍 Model available at: {model_path}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise

if __name__ == "__main__":
    main()