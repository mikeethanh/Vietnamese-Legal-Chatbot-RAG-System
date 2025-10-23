#!/usr/bin/env python3
"""
GPU-optimized training script cho Vietnamese Legal documents trên Digital Ocean
Tối ưu hóa cho NVIDIA V100 GPU
"""

import os
import json
import argparse
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
import numpy as np
from typing import List, Tuple
import boto3
from sklearn.model_selection import train_test_split
import random
from datetime import datetime
import psutil

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUOptimizedTrainer:
    """GPU-optimized trainer với memory management"""
    
    def __init__(self, args):
        self.args = args
        self.setup_gpu()
        self.setup_spaces_client()
        
    def setup_gpu(self):
        """Setup GPU và kiểm tra resources"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA không khả dụng!")
            
        self.device = torch.device('cuda:0')
        gpu_props = torch.cuda.get_device_properties(0)
        
        logger.info(f"🚀 GPU: {gpu_props.name}")
        logger.info(f"🚀 GPU Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        logger.info(f"🚀 CUDA Version: {torch.version.cuda}")
        logger.info(f"🚀 PyTorch Version: {torch.__version__}")
        
        # Tối ưu hóa GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
    def setup_spaces_client(self):
        """Setup Digital Ocean Spaces client"""
        self.spaces_client = boto3.client(
            's3',
            aws_access_key_id=self.args.spaces_access_key,
            aws_secret_access_key=self.args.spaces_secret_key,
            endpoint_url=self.args.spaces_endpoint,
            region_name='sgp1'
        )
        
    def monitor_resources(self):
        """Monitor GPU và system resources"""
        # GPU monitoring
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"📊 GPU Memory: Used {gpu_memory:.1f}GB, Cached {gpu_memory_cached:.1f}GB")
            
        # System monitoring
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        logger.info(f"📊 CPU: {cpu_percent}%, RAM: {memory.percent}%")
        
    def download_data(self):
        """Download training data từ Spaces"""
        os.makedirs(self.args.data_dir, exist_ok=True)
        
        corpus_local_path = os.path.join(self.args.data_dir, 'merged_corpus.jsonl')
        
        try:
            logger.info("📥 Downloading training data...")
            self.spaces_client.download_file(
                self.args.spaces_bucket,
                'process_data/rag_corpus/merged_corpus.jsonl',
                corpus_local_path
            )
            logger.info(f"✅ Downloaded to {corpus_local_path}")
            return corpus_local_path
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise
            
    def create_optimized_dataset(self, data_path: str):
        """Tạo dataset được tối ưu cho GPU training"""
        logger.info("🔧 Creating optimized dataset...")
        
        texts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.args.max_samples and i >= self.args.max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    if 'content' in data and len(data['content'].strip()) > 20:
                        texts.append(data['content'].strip())
                except json.JSONDecodeError:
                    continue
                    
        logger.info(f"📚 Loaded {len(texts)} texts")
        
        # Tạo training examples với strategies khác nhau
        examples = []
        
        # Strategy 1: Sequential pairs (high similarity)
        for i in range(0, len(texts) - 1, 2):
            if i + 1 < len(texts):
                examples.append(InputExample(
                    texts=[texts[i], texts[i + 1]], 
                    label=0.85
                ))
        
        # Strategy 2: Random pairs (low similarity)
        num_negative = len(examples) // 2
        for _ in range(num_negative):
            idx1, idx2 = random.sample(range(len(texts)), 2)
            examples.append(InputExample(
                texts=[texts[idx1], texts[idx2]], 
                label=0.15
            ))
            
        # Strategy 3: Similar domain pairs (medium similarity)
        keywords = ['luật', 'bộ luật', 'nghị định', 'thông tư', 'quyết định']
        for keyword in keywords:
            keyword_texts = [t for t in texts if keyword.lower() in t.lower()]
            if len(keyword_texts) >= 2:
                for _ in range(min(50, len(keyword_texts) // 2)):
                    idx1, idx2 = random.sample(range(len(keyword_texts)), 2)
                    examples.append(InputExample(
                        texts=[keyword_texts[idx1], keyword_texts[idx2]], 
                        label=0.65
                    ))
        
        random.shuffle(examples)
        logger.info(f"🎯 Created {len(examples)} training examples")
        
        return examples
        
    def load_optimized_model(self):
        """Load model với optimization cho từng loại"""
        logger.info(f"🤖 Loading model: {self.args.base_model}")
        
        try:
            # Thử load trực tiếp như SentenceTransformer
            model = SentenceTransformer(self.args.base_model)
            logger.info("✅ Loaded as SentenceTransformer model")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load as SentenceTransformer: {e}")
            logger.info("� Loading as base transformer and wrapping...")
            
            # Load base transformer và wrap
            from sentence_transformers import models
            from transformers import AutoModel, AutoTokenizer
            
            # Load tokenizer và model
            tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
            transformer_model = AutoModel.from_pretrained(self.args.base_model)
            
            # Tạo SentenceTransformer wrapper với config max_seq_length
            max_seq_length = getattr(self.args, 'max_seq_length', 512)
            word_embedding_model = models.Transformer(
                model_name_or_path=self.args.base_model,
                max_seq_length=max_seq_length
            )
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode='mean'
            )
            
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            logger.info("✅ Created SentenceTransformer wrapper")
            
        model.to(self.device)
        
        # Model-specific optimizations
        if 'electra' in self.args.base_model.lower():
            logger.info("⚡ Applying ELECTRA optimizations")
            # ELECTRA works better with smaller learning rates
            model._first_module().auto_model.config.hidden_dropout_prob = 0.1
            
        elif 'phobert' in self.args.base_model.lower():
            logger.info("🇻🇳 Applying PhoBERT optimizations")
            # PhoBERT specific settings
            model._first_module().auto_model.config.attention_probs_dropout_prob = 0.1
            
        elif 'e5' in self.args.base_model.lower():
            logger.info("🌐 Applying E5 optimizations")
            # E5 models work better with instruction-style training
            
        return model
        
    def train_model(self, data_path: str):
        """GPU-optimized training"""
        logger.info("🚀 Starting GPU training...")
        
        # Load model với optimization
        model = self.load_optimized_model()
        
        # Enable mixed precision nếu GPU hỗ trợ
        if hasattr(torch.cuda, 'amp'):
            logger.info("⚡ Enabling mixed precision training")
            
        # Load dataset
        train_examples = self.create_optimized_dataset(data_path)
        
        # Split train/validation
        train_examples, val_examples = train_test_split(
            train_examples, 
            test_size=0.1, 
            random_state=42
        )
        
        # DataLoader với GPU optimization
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=self.args.batch_size,
            num_workers=4,  # Tăng số workers cho GPU
            pin_memory=True  # Pin memory cho transfer nhanh hơn
        )
        
        # Loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            val_examples[:100],  # Giảm validation set để tăng tốc
            name='legal-eval'
        )
        
        # Training với H100 optimizations
        start_time = time.time()
        
        # Use warmup steps from config
        warmup_steps = getattr(self.args, 'warmup_steps', int(len(train_dataloader) * 0.1))
        logger.info(f"🔥 Warmup steps: {warmup_steps}")
        
        # Use learning rate from config
        learning_rate = getattr(self.args, 'learning_rate', 2e-5)
        logger.info(f"📈 Learning rate: {learning_rate}")
        
        # Use FP16 setting from config
        use_amp = getattr(self.args, 'use_fp16', True)
        logger.info(f"⚡ Mixed precision: {use_amp}")
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.args.epochs,
            evaluation_steps=max(50, len(train_dataloader) // 10),  # Adaptive evaluation
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},  # Custom learning rate
            output_path=self.args.output_dir,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=use_amp,  # Use config FP16 setting
        )
        
        training_time = time.time() - start_time
        logger.info(f"⏱️ Training completed in {training_time:.1f} seconds")
        
        # Monitor final resources
        self.monitor_resources()
        
        return model
        
    def upload_model(self):
        """Upload trained model to Spaces"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_prefix = f"models/embedding_model_gpu_{timestamp}"
        
        logger.info(f"📤 Uploading model to {s3_prefix}...")
        
        # Upload all files in output directory
        for root, dirs, files in os.walk(self.args.output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, self.args.output_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                try:
                    self.spaces_client.upload_file(local_path, self.args.spaces_bucket, s3_key)
                    logger.info(f"✅ Uploaded {relative_path}")
                except Exception as e:
                    logger.error(f"❌ Upload failed {relative_path}: {e}")
                    
        # Upload metadata
        metadata = {
            "model_name": f"embedding_model_gpu_{timestamp}",
            "base_model": self.args.base_model,
            "training_date": timestamp,
            "epochs": self.args.epochs,
            "batch_size": self.args.batch_size,
            "gpu_used": torch.cuda.get_device_name(),
            "training_time_seconds": getattr(self, 'training_time', 0),
            "model_path": s3_prefix
        }
        
        metadata_path = os.path.join(self.args.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        self.spaces_client.upload_file(
            metadata_path,
            self.args.spaces_bucket,
            f"{s3_prefix}/metadata.json"
        )
        
        logger.info(f"🎉 Model uploaded successfully to: {s3_prefix}")
        return s3_prefix

def main():
    parser = argparse.ArgumentParser(description='GPU Training for Vietnamese Legal Embedding')
    
    # Model arguments
    parser.add_argument('--base-model', default=os.getenv('BASE_MODEL', 'VietAI/viet-electra-base'),
                       help='Base model for fine-tuning')
    parser.add_argument('--model-type', default='auto', 
                       choices=['auto', 'electra', 'phobert', 'mpnet', 'e5'],
                       help='Model type for specific optimizations')
    parser.add_argument('--data-dir', default='/tmp/data',
                       help='Directory to store downloaded data')
    parser.add_argument('--output-dir', default='/tmp/model',
                       help='Directory to save trained model')
    
    # Training arguments - Read from environment variables
    parser.add_argument('--epochs', type=int, default=int(os.getenv('EPOCHS', '5')),
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('GPU_BATCH_SIZE', '32')),
                       help='Batch size for training (higher for GPU)')
    parser.add_argument('--max-samples', type=int, 
                       default=int(os.getenv('MAX_SAMPLES', '0')) if os.getenv('MAX_SAMPLES') else None,
                       help='Maximum number of samples to use')
    
    # Digital Ocean Spaces arguments - Read from environment variables with fallback
    parser.add_argument('--spaces-access-key', 
                       default=os.getenv('SPACES_ACCESS_KEY'),
                       help='Digital Ocean Spaces access key')
    parser.add_argument('--spaces-secret-key', 
                       default=os.getenv('SPACES_SECRET_KEY'),
                       help='Digital Ocean Spaces secret key')
    parser.add_argument('--spaces-endpoint', 
                       default=os.getenv('SPACES_ENDPOINT', 'https://sgp1.digitaloceanspaces.com'),
                       help='Digital Ocean Spaces endpoint')
    parser.add_argument('--spaces-bucket', 
                       default=os.getenv('SPACES_BUCKET', 'legal-datalake'),
                       help='Digital Ocean Spaces bucket name')
    
    args = parser.parse_args()
    
    # Validate required environment variables
    if not args.spaces_access_key:
        logger.error("❌ SPACES_ACCESS_KEY is required! Set it in environment or pass --spaces-access-key")
        raise ValueError("SPACES_ACCESS_KEY is missing")
    
    if not args.spaces_secret_key:
        logger.error("❌ SPACES_SECRET_KEY is required! Set it in environment or pass --spaces-secret-key")
        raise ValueError("SPACES_SECRET_KEY is missing")
    
    # Apply H100 GPU optimizations from environment
    use_fp16 = os.getenv('USE_FP16', 'true').lower() == 'true'
    learning_rate = float(os.getenv('LEARNING_RATE', '1e-5'))
    warmup_steps = int(os.getenv('WARMUP_STEPS', '1000'))
    max_seq_length = int(os.getenv('MAX_SEQ_LENGTH', '512'))
    gradient_accumulation_steps = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4'))
    
    # Add H100 optimizations to args
    args.use_fp16 = use_fp16
    args.learning_rate = learning_rate
    args.warmup_steps = warmup_steps
    args.max_seq_length = max_seq_length
    args.gradient_accumulation_steps = gradient_accumulation_steps
    
    # Log configuration
    logger.info("🔧 Training Configuration:")
    logger.info(f"   Base Model: {args.base_model}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch Size: {args.batch_size}")
    logger.info(f"   Max Samples: {args.max_samples}")
    logger.info(f"   Learning Rate: {args.learning_rate}")
    logger.info(f"   Warmup Steps: {args.warmup_steps}")
    logger.info(f"   Max Seq Length: {args.max_seq_length}")
    logger.info(f"   Use FP16: {args.use_fp16}")
    logger.info(f"   Gradient Accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"   Spaces Bucket: {args.spaces_bucket}")
    logger.info(f"   Output Dir: {args.output_dir}")
    
    # Create output directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('/tmp/logs', exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = GPUOptimizedTrainer(args)
        
        # Monitor initial resources
        trainer.monitor_resources()
        
        # Download data
        logger.info("📥 Step 1: Downloading training data...")
        data_path = trainer.download_data()
        
        # Train model
        logger.info("🚀 Step 2: Training model...")
        model = trainer.train_model(data_path)
        
        # Upload model
        logger.info("📤 Step 3: Uploading trained model...")
        model_path = trainer.upload_model()
        
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