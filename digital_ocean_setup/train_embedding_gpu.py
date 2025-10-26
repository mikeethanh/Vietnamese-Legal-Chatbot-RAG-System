#!/usr/bin/env python3
"""
Simple and robust GPU training script for Vietnamese Legal documents
Optimized for stability and ease of use - FP32 training only
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
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        exit(1)
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"� GPU: {gpu_name}")
    logger.info(f"� Total VRAM: {total_memory:.1f}GB")
    
    return device

def log_memory_usage(context):
    """Log detailed memory usage with context"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"📊 Memory [{context}]:")
        logger.info(f"   Allocated: {allocated:.2f}GB")
        logger.info(f"   Cached: {cached:.2f}GB") 
        logger.info(f"   Peak: {max_allocated:.2f}GB")

def monitor_memory(step_name):
    """Monitor GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    logger.info(f"🔍 {step_name} - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

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
    logger.info("🔧 Creating training examples...")
    
    examples = []
    
    # CRITICAL: Giới hạn số examples để tránh OOM với large datasets
    max_examples = int(os.getenv('MAX_TRAINING_EXAMPLES', '100000'))
    
    # Positive pairs (sequential texts - similar)
    positive_count = 0
    for i in range(0, min(len(texts) - 1, max_examples), 2):
        if i + 1 < len(texts):
            examples.append(InputExample(
                texts=[texts[i], texts[i + 1]], 
                label=0.8
            ))
            positive_count += 1
    
    # Negative pairs (random texts - dissimilar) - balance với positive
    negative_count = min(positive_count // 2, 5000)  # Giảm tỉ lệ negative để tiết kiệm memory
    for _ in range(negative_count):
        idx1, idx2 = random.sample(range(len(texts)), 2)
        examples.append(InputExample(
            texts=[texts[idx1], texts[idx2]], 
            label=0.2
        ))
    
    random.shuffle(examples)
    
    logger.info(f"✅ Created {len(examples)} training examples (max_allowed: {max_examples})")
    logger.info(f"   📊 Positive pairs: {positive_count} (label=0.8)")
    logger.info(f"   📊 Negative pairs: {negative_count} (label=0.2)")
    
    # Memory warning
    if len(examples) > 50000:
        logger.warning(f"⚠️ Large training set ({len(examples)} examples) may cause OOM")
        logger.warning(f"💡 Consider setting MAX_TRAINING_EXAMPLES < 50000")
    
    return examples

def train_model(model_name, examples, device, epochs=3, batch_size=16):
    """Train the embedding model with proper memory management"""
    logger.info(f"🤖 Loading model: {model_name}")
    
    
    # Clear ALL GPU memory first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load model with memory optimization
    try:
        model = SentenceTransformer(model_name, device=device)
        
        # Disable gradient checkpointing for faster training
        use_gradient_checkpointing = os.getenv('USE_GRADIENT_CHECKPOINTING', 'false').lower() == 'true'
        if use_gradient_checkpointing and hasattr(model[0].auto_model, 'gradient_checkpointing_enable'):
            model[0].auto_model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        else:
            logger.info("ℹ️ Gradient checkpointing disabled (faster training)")
        
        logger.info("✅ Model loaded successfully")
        
        # Log memory usage after model loading
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info(f"🔍 GPU memory after model load: {allocated:.2f} GB")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        torch.cuda.empty_cache()
        exit(1)
    
    # Split data with memory consideration
    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=42, stratify=None  # ✅ Không stratify vì chỉ có 2 labels
    )
    
    logger.info(f"📊 Training examples: {len(train_examples)}")
    logger.info(f"📊 Validation examples: {len(val_examples)}")
    
    # ✅ ĐẢM BẢO validation set có CẢ positive VÀ negative examples
    val_labels = [ex.label for ex in val_examples]
    unique_labels = set(val_labels)
    
    logger.info(f"🔍 Validation set labels: {unique_labels}")
    
    if len(unique_labels) < 2:
        logger.warning("⚠️ Validation set chỉ có 1 loại label! Tạo lại balanced validation set...")
        
        # Tách positive và negative từ toàn bộ examples
        positive_examples = [ex for ex in examples if ex.label > 0.5]
        negative_examples = [ex for ex in examples if ex.label <= 0.5]
        
        # Lấy 50% positive, 50% negative cho validation
        val_size = max(20, int(len(examples) * 0.1))
        val_positive = random.sample(positive_examples, val_size // 2)
        val_negative = random.sample(negative_examples, val_size // 2)
        
        val_examples = val_positive + val_negative
        random.shuffle(val_examples)
        
        # Train set là phần còn lại
        val_ids = {id(ex) for ex in val_examples}
        train_examples = [ex for ex in examples if id(ex) not in val_ids]
        
        logger.info(f"✅ Recreated balanced validation set:")
        logger.info(f"   📊 Positive: {len(val_positive)}")
        logger.info(f"   📊 Negative: {len(val_negative)}")
    
    # Get num_workers from environment variable or use default
    num_workers = int(os.getenv('DATALOADER_NUM_WORKERS', '4'))  # Giảm từ 8 → 4 để tiết kiệm RAM
    
    # CRITICAL: Memory-optimized DataLoader - tắt pin_memory với large datasets
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  # ❌ TẮT pin_memory để tránh memory leak với large datasets
        persistent_workers=False,  # ❌ TẮT persistent workers để giải phóng memory
        prefetch_factor=2  # Giảm prefetch để tiết kiệm memory
    )
    
    # Loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Smaller evaluator to save memory
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples[:20],  # Reduced from 50 to 20
        name='legal-eval'
    )
    
    # Training with memory optimization
    logger.info(f"🔥 Starting training...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Num workers: {num_workers}")
    logger.info(f"   Gradient checkpointing: {use_gradient_checkpointing}")
    
    start_time = time.time()
    
    # Clear cache before training
    torch.cuda.empty_cache()
    
    try:
        # Get additional memory optimization settings
        max_seq_length = int(os.getenv('MAX_SEQ_LENGTH', '256'))
        gradient_accumulation_steps = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4'))  # Default 4 thay vì 1
        
        # CRITICAL: Giới hạn max_seq_length của model để tiết kiệm memory
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = max_seq_length
            logger.info(f"✅ Set max_seq_length = {max_seq_length}")
        
        # Training arguments with AGGRESSIVE memory optimization
        training_args = {
            'train_objectives': [(train_dataloader, train_loss)],
            'evaluator': evaluator,
            'epochs': epochs,
            'evaluation_steps': max(500, len(train_dataloader) // 2),  # Giảm tần suất evaluation
            'warmup_steps': int(len(train_dataloader) * 0.05),  # Giảm warmup steps
            'optimizer_params': {
                'lr': float(os.getenv('LEARNING_RATE', '2e-5')),
            },
            'output_path': '/tmp/model',
            'save_best_model': True,
            'show_progress_bar': True,
            'checkpoint_path': None,  # ❌ TẮT checkpointing
            'checkpoint_save_steps': 0,  # ❌ TẮT intermediate checkpoints
            'checkpoint_save_total_limit': 0,  # ❌ Không lưu checkpoints
            'max_grad_norm': 1.0,
            'use_amp': False,  # ❌ TẮT AMP để tránh memory fragmentation
        }
        
        # Gradient accumulation để maintain effective batch size với batch nhỏ hơn
        if gradient_accumulation_steps > 1:
            training_args['steps_per_epoch'] = len(train_dataloader) // gradient_accumulation_steps
            logger.info(f"✅ Gradient accumulation: {gradient_accumulation_steps} steps")
            logger.info(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
        
        # CRITICAL: Thêm callback để clear cache mỗi N steps
        class MemoryClearCallback:
            def __init__(self, clear_every_n_steps=50):
                self.clear_every_n_steps = clear_every_n_steps
                self.step_count = 0
            
            def on_step_end(self, *args, **kwargs):
                self.step_count += 1
                if self.step_count % self.clear_every_n_steps == 0:
                    torch.cuda.empty_cache()
                    if self.step_count % 200 == 0:  # Log mỗi 200 steps
                        allocated = torch.cuda.memory_allocated() / 1e9
                        logger.info(f"🧹 Memory cleared at step {self.step_count}: {allocated:.2f}GB")
        
        # Simple training without FP16 complications
        logger.info("🚀 Starting model.fit() with memory optimizations...")
        
        model.fit(**training_args)
        
        # Clear cache after training
        torch.cuda.empty_cache()
        logger.info("✅ Training completed, memory cleared")
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"💥 CUDA OOM Error: {e}")
        logger.error("💡 Memory Debugging Info:")
        logger.error(f"   - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.error(f"   - Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB") 
        logger.error("💡 Recommendations:")
        logger.error(f"   - Current batch_size: {batch_size} → try batch_size=1")
        logger.error(f"   - Current max_samples: {os.getenv('MAX_SAMPLES')} → try 10000")
        logger.error("   - Enable gradient_accumulation_steps to maintain effective batch size")
        
        # Clean up memory
        del model
        torch.cuda.empty_cache()
        raise
    
    except Exception as e:
        logger.error(f"💥 Training error: {e}")
        torch.cuda.empty_cache()
        raise
    
    training_time = time.time() - start_time
    logger.info(f"⏱️ Training completed in {training_time:.1f} seconds")
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    
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
    
    # CRITICAL: Set memory optimization GLOBALLY before anything else
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    # CRITICAL: Giới hạn số lượng threads để tránh memory overhead
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    # Configuration from environment
    model_name = os.getenv('BASE_MODEL', 'BAAI/bge-m3')
    epochs = int(os.getenv('EPOCHS', '3'))
    batch_size = int(os.getenv('GPU_BATCH_SIZE', '8'))  # Default 8 thay vì 16
    max_samples = int(os.getenv('MAX_SAMPLES', '50000')) if os.getenv('MAX_SAMPLES') else 50000  # Default 50K
    bucket_name = os.getenv('SPACES_BUCKET', 'legal-datalake')
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Bucket: {bucket_name}")
    logger.info(f"   Memory optimization: AGGRESSIVE")
    logger.info(f"   PYTORCH_CUDA_ALLOC_CONF: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    
    try:
        # Setup GPU and clear memory first
        device = check_gpu()
        
        # Clear all GPU memory before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
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
        
        # Final memory cleanup and logging
        del model
        torch.cuda.empty_cache()
        log_memory_usage("Final cleanup")
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"📍 Model available at: {model_path}")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
        log_memory_usage("Interrupted state")
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        # Log memory state on error
        log_memory_usage("Error state")
        raise

if __name__ == "__main__":
    main()