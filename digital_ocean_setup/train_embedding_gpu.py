#!/usr/bin/env python3
"""
Simplified GPU training script for Vietnamese Legal documents on Digital Ocean
Optimized for NVIDIA V100/H100 GPU
"""

import os
import json
import logging
import time
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import boto3
from sklearn.model_selection import train_test_split
import random
from datetime import datetime

# Simple logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleTrainer:
    """Simplified trainer for Digital Ocean GPU"""
    
    def __init__(self, spaces_access_key, spaces_secret_key, spaces_bucket):
        self.spaces_bucket = spaces_bucket
        self.setup_gpu()
        self.setup_spaces_client(spaces_access_key, spaces_secret_key)
        
    def setup_gpu(self):
        """Setup GPU environment"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available!")
            
        self.device = torch.device('cuda:0')
        gpu_props = torch.cuda.get_device_properties(0)
        
        logger.info(f"🚀 GPU: {gpu_props.name}")
        logger.info(f"🚀 GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
    def setup_spaces_client(self, access_key, secret_key):
        """Setup Digital Ocean Spaces client"""
        endpoint_url = os.getenv('SPACES_ENDPOINT', 'https://sfo3.digitaloceanspaces.com')
        region =  'sfo3'
        
        self.spaces_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name=region
        )
        
    def download_data(self, data_dir):
        """Download training data from Spaces"""
        os.makedirs(data_dir, exist_ok=True)
        corpus_local_path = os.path.join(data_dir, 'merged_corpus.jsonl')
        
        try:
            logger.info("📥 Downloading training data...")
            self.spaces_client.download_file(
                self.spaces_bucket,
                'process_data/rag_corpus/merged_corpus.jsonl',
                corpus_local_path
            )
            logger.info(f"✅ Downloaded to {corpus_local_path}")
            return corpus_local_path
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise
            
    def create_training_dataset(self, data_path, max_samples=None):
        """Create training dataset from corpus"""
        logger.info("🔧 Creating training dataset...")
        
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    if 'content' in data and len(data['content'].strip()) > 20:
                        texts.append(data['content'].strip())
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"📚 Loaded {len(texts)} texts")
        
        # Create training examples
        examples = []
        
        # Sequential pairs with high similarity
        for i in range(0, len(texts) - 1, 2):
            if i + 1 < len(texts):
                examples.append(InputExample(
                    texts=[texts[i], texts[i + 1]], 
                    label=0.8
                ))
        
        # Random pairs with low similarity  
        num_negative = min(len(examples), 1000)  # Limit negative samples
        for _ in range(num_negative):
            idx1, idx2 = random.sample(range(len(texts)), 2)
            examples.append(InputExample(
                texts=[texts[idx1], texts[idx2]], 
                label=0.2
            ))
        
        random.shuffle(examples)
        logger.info(f"🎯 Created {len(examples)} training examples")
        
        return examples
        
    def load_model(self, base_model):
        """Load SentenceTransformer model"""
        logger.info(f"🤖 Loading model: {base_model}")
        
        try:
            model = SentenceTransformer(base_model)
            model.to(self.device)
            logger.info("✅ Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
    def train_model(self, base_model, data_path, output_dir, epochs=3, batch_size=32, learning_rate=2e-5, max_samples=None):
        """Train embedding model"""
        logger.info("🚀 Starting training...")
        
        # Load model
        model = self.load_model(base_model)
        
        # Create dataset
        train_examples = self.create_training_dataset(data_path, max_samples)
        
        # Split train/validation
        train_examples, val_examples = train_test_split(
            train_examples, 
            test_size=0.1, 
            random_state=42
        )
        
        # DataLoader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # Loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Evaluator (smaller validation set)
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples[:50],
            name='legal-eval'
        )
        
        # Training
        start_time = time.time()
        warmup_steps = int(len(train_dataloader) * 0.1)
        
        logger.info(f"🔥 Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=len(train_dataloader) // 2,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            output_path=output_dir,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=True  # Mixed precision
        )
        
        training_time = time.time() - start_time
        logger.info(f"⏱️ Training completed in {training_time:.1f} seconds")
        
        return model
        
    def upload_model(self, output_dir, base_model, epochs, batch_size):
        """Upload trained model to Spaces"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_prefix = f"models/embedding_model_gpu_{timestamp}"
        
        logger.info(f"📤 Uploading model to {s3_prefix}...")
        
        # Upload all files in output directory
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, output_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                try:
                    self.spaces_client.upload_file(local_path, self.spaces_bucket, s3_key)
                    logger.info(f"✅ Uploaded {relative_path}")
                except Exception as e:
                    logger.error(f"❌ Upload failed {relative_path}: {e}")
                    
        # Upload metadata
        metadata = {
            "model_name": f"embedding_model_gpu_{timestamp}",
            "base_model": base_model,
            "training_date": timestamp,
            "epochs": epochs,
            "batch_size": batch_size,
            "gpu_used": torch.cuda.get_device_name(),
            "model_path": s3_prefix
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.spaces_client.upload_file(
            metadata_path,
            self.spaces_bucket,
            f"{s3_prefix}/metadata.json"
        )
        
        logger.info(f"🎉 Model uploaded successfully to: {s3_prefix}")
        return s3_prefix

def main():
    """Main training function with simplified configuration"""
    
    # Get configuration from environment variables
    base_model = os.getenv('BASE_MODEL', 'BAAI/bge-m3')
    epochs = int(os.getenv('EPOCHS', '3'))
    batch_size = int(os.getenv('GPU_BATCH_SIZE', '32'))
    learning_rate = float(os.getenv('LEARNING_RATE', '2e-5'))
    max_samples = int(os.getenv('MAX_SAMPLES', '10000')) if os.getenv('MAX_SAMPLES') else None
    
    # Spaces configuration
    spaces_access_key = os.getenv('SPACES_ACCESS_KEY')
    spaces_secret_key = os.getenv('SPACES_SECRET_KEY')
    spaces_bucket = os.getenv('SPACES_BUCKET', 'legal-datalake')
    
    # Directories
    data_dir = '/tmp/data'
    output_dir = '/tmp/model'
    
    # Validate required environment variables
    if not spaces_access_key:
        logger.error("❌ SPACES_ACCESS_KEY is required!")
        return
    
    if not spaces_secret_key:
        logger.error("❌ SPACES_SECRET_KEY is required!")
        return
    
    # Log configuration
    logger.info("🔧 Training Configuration:")
    logger.info(f"   Base Model: {base_model}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch Size: {batch_size}")
    logger.info(f"   Learning Rate: {learning_rate}")
    logger.info(f"   Max Samples: {max_samples}")
    logger.info(f"   Spaces Bucket: {spaces_bucket}")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = SimpleTrainer(spaces_access_key, spaces_secret_key, spaces_bucket)
        
        # Download data
        logger.info("📥 Step 1: Downloading training data...")
        data_path = trainer.download_data(data_dir)
        
        # Train model
        logger.info("🚀 Step 2: Training model...")
        model = trainer.train_model(base_model, data_path, output_dir, epochs, batch_size, learning_rate, max_samples)
        
        # Upload model
        logger.info("📤 Step 3: Uploading trained model...")
        model_path = trainer.upload_model(output_dir, base_model, epochs, batch_size)
        
        # Final summary
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📍 Model saved at: {model_path}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        raise

if __name__ == "__main__":
    main()