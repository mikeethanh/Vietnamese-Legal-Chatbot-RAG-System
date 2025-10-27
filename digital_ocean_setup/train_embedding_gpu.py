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
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import hashlib
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
    """
    Create training examples using SimCSE approach
    SimCSE: Same text với 2 lần forward pass (dropout tạo augmentation)
    """
    logger.info("🔧 Creating SimCSE training examples...")
    
    # CRITICAL: Giới hạn số examples để tránh OOM với large datasets
    max_examples = int(os.getenv('MAX_TRAINING_EXAMPLES', '50000'))
    
    # SimCSE: Mỗi text là 1 example, model sẽ tự tạo positive pair qua dropout
    # Format: InputExample(texts=[anchor, positive])
    # Với SimCSE, anchor và positive là cùng 1 text (dropout tạo khác biệt)
    examples = []
    
    limited_texts = texts[:max_examples]
    
    for text in limited_texts:
        # SimCSE: Positive pair = same text (dropout makes them different)
        examples.append(InputExample(
            texts=[text, text]  # Anchor và positive giống nhau
        ))
    
    logger.info(f"✅ Created {len(examples)} SimCSE training examples (max_allowed: {max_examples})")
    logger.info(f"   📊 Method: SimCSE (dropout-based augmentation)")
    logger.info(f"   📊 Each example uses same text twice for positive pair")
    
    # Memory warning
    if len(examples) > 50000:
        logger.warning(f"⚠️ Large training set ({len(examples)} examples) may cause OOM")
        logger.warning(f"💡 Consider setting MAX_TRAINING_EXAMPLES < 50000")
    
    return examples

def build_synthetic_evaluation(texts, num_queries=500, num_corpus=5000):
    """
    Build synthetic evaluation set for Information Retrieval
    Tạo queries và corpus từ texts để evaluate khả năng retrieval
    
    Returns:
        queries: Dict[query_id, query_text]
        corpus: Dict[doc_id, doc_text]
        relevant_docs: Dict[query_id, Set[doc_id]] - ground truth
    """
    logger.info("🔧 Building synthetic evaluation set...")
    
    # Đảm bảo không vượt quá số texts có sẵn
    num_queries = min(num_queries, len(texts) // 10)
    num_corpus = min(num_corpus, len(texts))
    
    # Shuffle texts để random selection
    eval_texts = texts.copy()
    random.shuffle(eval_texts)
    
    # Split: 10% cho queries, 90% cho corpus
    query_texts = eval_texts[:num_queries]
    corpus_texts = eval_texts[num_queries:num_queries + num_corpus]
    
    # Build queries dict
    queries = {}
    for i, text in enumerate(query_texts):
        query_id = f"q{i}"
        # Lấy 1 phần của text làm query (simulate real query)
        words = text.split()
        if len(words) > 10:
            query = ' '.join(words[:len(words)//2])  # Lấy nửa đầu làm query
        else:
            query = text
        queries[query_id] = query
    
    # Build corpus dict
    corpus = {}
    for i, text in enumerate(corpus_texts):
        doc_id = f"doc{i}"
        corpus[doc_id] = text
    
    # Build relevant_docs (ground truth)
    # Strategy: Tìm docs có overlap từ với query
    relevant_docs = {}
    
    for query_id, query_text in queries.items():
        query_words = set(query_text.lower().split())
        relevant_set = set()
        
        # Tìm top-K docs có nhiều overlap words nhất
        doc_scores = []
        for doc_id, doc_text in corpus.items():
            doc_words = set(doc_text.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                doc_scores.append((doc_id, overlap))
        
        # Lấy top-3 làm relevant docs
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        for doc_id, score in doc_scores[:3]:
            relevant_set.add(doc_id)
        
        if relevant_set:  # Chỉ add nếu có relevant docs
            relevant_docs[query_id] = relevant_set
    
    logger.info(f"✅ Synthetic evaluation set created:")
    logger.info(f"   📊 Queries: {len(queries)}")
    logger.info(f"   📊 Corpus: {len(corpus)}")
    logger.info(f"   📊 Queries with relevant docs: {len(relevant_docs)}")
    
    # Log sample
    if queries and corpus:
        sample_qid = list(queries.keys())[0]
        logger.info(f"   📝 Sample query: {queries[sample_qid][:100]}...")
        if sample_qid in relevant_docs:
            logger.info(f"   � Relevant docs: {len(relevant_docs[sample_qid])} docs")
    
    return queries, corpus, relevant_docs

def train_model(model_name, examples, device, epochs=3, batch_size=64):
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
        examples, test_size=0.1, random_state=42
    )
    
    logger.info(f"📊 Training examples: {len(train_examples)}")
    logger.info(f"📊 Validation examples: {len(val_examples)}")
    
    # Get num_workers from environment variable or use default
    num_workers = int(os.getenv('DATALOADER_NUM_WORKERS', '4'))  
    
    # CRITICAL: Memory-optimized DataLoader - tắt pin_memory với large datasets
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,  
        persistent_workers=False,  
        prefetch_factor=2  
    )
    
    # Loss function: MultipleNegativesRankingLoss cho SimCSE
    # Loss này tự động tạo in-batch negatives từ các examples khác trong batch
    train_loss = losses.MultipleNegativesRankingLoss(model)
    logger.info("✅ Using MultipleNegativesRankingLoss (SimCSE compatible)")
    
    # Build synthetic evaluation set
    logger.info("🔧 Building evaluation set from validation texts...")
    # Extract texts from validation examples
    val_texts = []
    for ex in val_examples[:5000]:  # Giới hạn để tiết kiệm memory
        val_texts.extend(ex.texts)
    val_texts = list(set(val_texts))  # Remove duplicates
    
    # Build IR evaluation
    queries, corpus, relevant_docs = build_synthetic_evaluation(
        val_texts, 
        num_queries=min(100, len(val_texts) // 50),  # Giảm xuống để tránh OOM
        num_corpus=min(1000, len(val_texts))
    )
    
    # Create InformationRetrievalEvaluator
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name='legal-ir-eval',
        show_progress_bar=False,  # Tắt progress bar để giảm overhead
        batch_size=32  # Batch size cho evaluation
    )
    logger.info("✅ InformationRetrievalEvaluator created")
    
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