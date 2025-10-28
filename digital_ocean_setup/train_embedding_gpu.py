#!/usr/bin/env python3
"""
Improved GPU training script for Vietnamese Legal documents
With enhanced diagnostics and better loss function
"""

import os
import json
import logging
import time
import sys
import torch
import boto3
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
from typing import List, Tuple

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

class ImprovedTripletEvaluator:
    """Enhanced evaluator with better metrics"""
    
    def __init__(self, anchors, positives, negatives, name="triplet_evaluation"):
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name
    
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """Evaluate with detailed metrics"""
        model.eval()
        
        # Encode all texts
        anchor_embeddings = model.encode(self.anchors, convert_to_tensor=True)
        positive_embeddings = model.encode(self.positives, convert_to_tensor=True)
        negative_embeddings = model.encode(self.negatives, convert_to_tensor=True)
        
        # Calculate similarities
        pos_similarities = torch.cosine_similarity(anchor_embeddings, positive_embeddings)
        neg_similarities = torch.cosine_similarity(anchor_embeddings, negative_embeddings)
        
        # Calculate metrics
        correct_predictions = (pos_similarities > neg_similarities).sum().item()
        total_predictions = len(pos_similarities)
        accuracy = correct_predictions / total_predictions
        
        mean_pos_sim = pos_similarities.mean().item()
        mean_neg_sim = neg_similarities.mean().item()
        similarity_gap = mean_pos_sim - mean_neg_sim
        
        # Log detailed metrics
        logger.info(f"🔍 Detailed Evaluation Results (Epoch {epoch:.2f}, Step {steps}):")
        logger.info(f"   📊 Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        logger.info(f"   📈 Mean Positive Similarity: {mean_pos_sim:.4f}")
        logger.info(f"   📉 Mean Negative Similarity: {mean_neg_sim:.4f}")
        logger.info(f"   📏 Similarity Gap: {similarity_gap:.4f}")
        logger.info(f"   🎯 Random Baseline: 50.0%")
        
        # Warning if performance is poor
        if accuracy < 0.5:
            logger.warning(f"⚠️  Accuracy {accuracy:.1%} < 50% (worse than random!)")
        if similarity_gap < 0.1:
            logger.warning(f"⚠️  Small similarity gap ({similarity_gap:.4f}) - model not learning distinction")
        
        model.train()
        return accuracy

def compute_baseline_performance(model, anchors, positives, negatives, sample_size=500):
    """Compute baseline performance before training"""
    logger.info("🔍 Computing baseline performance...")
    
    # Sample a subset for faster evaluation
    indices = random.sample(range(len(anchors)), min(sample_size, len(anchors)))
    sample_anchors = [anchors[i] for i in indices]
    sample_positives = [positives[i] for i in indices]
    sample_negatives = [negatives[i] for i in indices]
    
    evaluator = ImprovedTripletEvaluator(sample_anchors, sample_positives, sample_negatives)
    baseline_score = evaluator(model, epoch=0, steps=0)
    
    logger.info(f"📊 Baseline Performance: {baseline_score:.1%}")
    if baseline_score < 0.4:
        logger.warning("⚠️  Very low baseline! Consider:")
        logger.warning("   - Checking data quality")
        logger.warning("   - Using a domain-specific base model")
        logger.warning("   - Increasing learning rate")
    
    return baseline_score

def inspect_data_samples(examples, num_samples=5):
    """Inspect training data quality"""
    logger.info("🔍 Inspecting data samples...")
    
    for i, example in enumerate(examples[:num_samples]):
        query, positive, negative = example.texts
        logger.info(f"📄 Sample {i+1}:")
        logger.info(f"   Query: {query[:100]}...")
        logger.info(f"   Positive: {positive[:100]}...")
        logger.info(f"   Negative: {negative[:100]}...")
        logger.info("   ---")

def check_gpu():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        sys.exit(1)
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    logger.info(f"🔥 GPU: {gpu_name}")
    logger.info(f"💾 Total VRAM: {total_memory:.1f}GB")
    
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
    else:
        logger.info(f"📊 Memory [{context}]: CUDA not available")

def setup_spaces_client():
    """Setup Digital Ocean Spaces client"""
    access_key = os.getenv('SPACES_ACCESS_KEY')
    secret_key = os.getenv('SPACES_SECRET_KEY')
    endpoint = os.getenv('SPACES_ENDPOINT', 'https://sfo3.digitaloceanspaces.com')
    
    if not access_key or not secret_key:
        logger.error("❌ SPACES_ACCESS_KEY and SPACES_SECRET_KEY required!")
        sys.exit(1)
    
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
    data_path = '/tmp/data/law_vi.jsonl'
    os.makedirs('/tmp/data', exist_ok=True)
    
    try:
        logger.info("📥 Downloading triplet training data...")
        spaces_client.download_file(
            bucket_name,
            'process_data/embed/law_vi.jsonl',
            data_path
        )
        logger.info(f"✅ Downloaded to {data_path}")
        return data_path
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)

def create_mixed_negatives(examples, hard_negative_ratio=0.7):
    """Create mixed negatives: some hard, some random for better training"""
    logger.info(f"🔄 Creating mixed negatives (hard: {hard_negative_ratio:.0%}, random: {1-hard_negative_ratio:.0%})")
    
    modified_examples = []
    num_random = int(len(examples) * (1 - hard_negative_ratio))
    
    # Get all positive texts for random sampling
    all_positives = [ex.texts[1] for ex in examples]
    
    for i, example in enumerate(examples):
        query, positive, hard_neg = example.texts
        
        # Use random negative for some examples
        if i < num_random:
            # Ensure we don't pick the same positive as negative
            available_negatives = [pos for pos in all_positives if pos != positive]
            random_neg = random.choice(available_negatives)
            modified_examples.append(InputExample(texts=[query, positive, random_neg]))
        else:
            # Keep original hard negative
            modified_examples.append(example)
    
    # Shuffle to mix hard and random negatives
    random.shuffle(modified_examples)
    
    logger.info(f"✅ Created {len(modified_examples)} examples with mixed negatives")
    return modified_examples

def load_triplet_data(data_path, max_samples=None):
    """Load triplet data from JSONL file and convert to triplet examples"""
    logger.info(f"📖 Loading triplet data from {data_path}")
    
    examples = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(examples) >= max_samples:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    
                    # Validate required fields
                    if not all(key in data for key in ['query', 'positive', 'hard_neg']):
                        logger.warning(f"Line {line_num}: Missing required fields")
                        continue
                    
                    query = data['query'].strip()
                    positive = data['positive'].strip()
                    hard_neg = data['hard_neg'].strip()
                    
                    # Skip empty texts
                    if not query or not positive or not hard_neg:
                        continue
                    
                    # Create triplet example
                    example = InputExample(texts=[query, positive, hard_neg])
                    examples.append(example)
                    
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
                    
                # Log progress
                if line_num % 10000 == 0:
                    logger.info(f"Processed {line_num} lines, created {len(examples)} examples")
        
        logger.info(f"✅ Loaded {len(examples)} triplet examples")
        return examples
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        sys.exit(1)

def train_model(model_name, examples, device, epochs=3, batch_size=64):
    """Train the embedding model with improved loss and evaluation"""
    logger.info(f"🤖 Loading model: {model_name}")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        model = SentenceTransformer(model_name, device=device)
        logger.info("✅ Model loaded successfully")
        log_memory_usage("After model load")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        torch.cuda.empty_cache()
        sys.exit(1)
    
    # Inspect data samples
    inspect_data_samples(examples)
    
    # Create mixed negatives for better training
    use_mixed_negatives = os.getenv('USE_MIXED_NEGATIVES', 'true').lower() == 'true'
    if use_mixed_negatives:
        examples = create_mixed_negatives(examples, hard_negative_ratio=0.7)
    
    # Split data
    train_examples, val_examples = train_test_split(
        examples, test_size=0.1, random_state=42
    )
    
    logger.info(f"📊 Training examples: {len(train_examples)}")
    logger.info(f"📊 Validation examples: {len(val_examples)}")
    
    # Prepare evaluation data
    eval_size = min(3000, len(val_examples))  # Larger eval set for stable metrics
    eval_examples = val_examples[:eval_size]
    
    anchors = [ex.texts[0] for ex in eval_examples]
    positives = [ex.texts[1] for ex in eval_examples]
    negatives = [ex.texts[2] for ex in eval_examples]
    
    # Compute baseline performance
    baseline_score = compute_baseline_performance(model, anchors, positives, negatives)
    
    # Create improved evaluator
    evaluator = ImprovedTripletEvaluator(anchors, positives, negatives, "triplet_evaluation")
    
    # Choose loss function
    loss_type = os.getenv('LOSS_TYPE', 'MultipleNegativesRanking')  # Changed default
    
    if loss_type == 'MultipleNegativesRanking':
        # Convert triplets to pairs for MultipleNegativesRankingLoss
        train_pairs = []
        for ex in train_examples:
            # Create positive pair
            train_pairs.append(InputExample(texts=[ex.texts[0], ex.texts[1]]))
        
        train_dataloader = DataLoader(train_pairs, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        logger.info("🔥 Using MultipleNegativesRankingLoss")
    else:
        # Use TripletLoss
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=model)
        logger.info("🔥 Using TripletLoss")
    
    # Training settings
    learning_rate = float(os.getenv('LEARNING_RATE', '2e-5'))
    max_seq_length = int(os.getenv('MAX_SEQ_LENGTH', '256'))
    
    if hasattr(model, 'max_seq_length'):
        model.max_seq_length = max_seq_length
        logger.info(f"✅ Set max_seq_length = {max_seq_length}")
    
    warmup_steps = int(len(train_dataloader) * 0.1)
    evaluation_steps = max(100, len(train_dataloader) // 5)  # More frequent evaluation
    
    logger.info(f"📊 Training settings:")
    logger.info(f"   Loss: {loss_type}")
    logger.info(f"   Learning rate: {learning_rate}")
    logger.info(f"   Warmup steps: {warmup_steps}")
    logger.info(f"   Evaluation steps: {evaluation_steps}")
    logger.info(f"   Eval set size: {eval_size}")
    logger.info(f"   Baseline accuracy: {baseline_score:.1%}")
    
    start_time = time.time()
    torch.cuda.empty_cache()
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=evaluation_steps,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            output_path='/tmp/model',
            save_best_model=True,
            show_progress_bar=True
        )
        
        torch.cuda.empty_cache()
        logger.info("✅ Training completed")
        
    except Exception as e:
        logger.error(f"💥 Training error: {e}")
        if "out of memory" in str(e).lower():
            logger.error("💡 Try reducing batch_size or max_seq_length")
        torch.cuda.empty_cache()
        raise
    
    training_time = time.time() - start_time
    logger.info(f"⏱️ Training completed in {training_time:.1f} seconds")
    
    return model

def upload_model(spaces_client, bucket_name, model_name):
    """Upload trained model to Spaces"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_prefix = f"models/embedding_model_improved_{timestamp}"
    
    logger.info(f"📤 Uploading model to {s3_prefix}...")
    
    model_dir = '/tmp/model'
    
    if not os.path.exists(model_dir):
        logger.error(f"❌ Model directory not found: {model_dir}")
        return None
    
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
    
    # Create metadata
    metadata = {
        "model_name": f"embedding_model_improved_{timestamp}",
        "base_model": model_name,
        "training_date": timestamp,
        "improvements": [
            "Mixed negatives (hard + random)",
            "MultipleNegativesRankingLoss option",
            "Enhanced evaluation metrics",
            "Baseline performance computation",
            "Better data inspection"
        ],
        "model_path": s3_prefix,
        "uploaded_files": uploaded_files
    }
    
    metadata_path = os.path.join(model_dir, 'metadata.json')
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        spaces_client.upload_file(
            metadata_path, bucket_name, f"{s3_prefix}/metadata.json"
        )
        logger.info("✅ Metadata uploaded")
    except Exception as e:
        logger.warning(f"⚠️ Failed to upload metadata: {e}")
    
    logger.info(f"🎉 Model uploaded successfully!")
    return s3_prefix

def main():
    """Main training function with improvements"""
    logger.info("🚀 Starting IMPROVED Vietnamese Legal Embedding Training")
    
    # Memory optimization
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
    
    # Configuration
    model_name = os.getenv('BASE_MODEL', 'BAAI/bge-m3')
    epochs = int(os.getenv('EPOCHS', '3'))
    batch_size = int(os.getenv('GPU_BATCH_SIZE', '32'))  # Reduced default batch size
    max_samples = int(os.getenv('MAX_SAMPLES', '30000'))
    bucket_name = os.getenv('SPACES_BUCKET', 'legal-datalake')
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Mixed negatives: {os.getenv('USE_MIXED_NEGATIVES', 'true')}")
    logger.info(f"   Loss type: {os.getenv('LOSS_TYPE', 'MultipleNegativesRanking')}")
    
    try:
        device = check_gpu()
        torch.cuda.empty_cache()
        
        spaces_client = setup_spaces_client()
        
        # Create directories
        os.makedirs('/tmp/data', exist_ok=True)
        os.makedirs('/tmp/model', exist_ok=True)
        os.makedirs('/tmp/logs', exist_ok=True)
        
        # Download and load data
        data_path = download_data(spaces_client, bucket_name)
        examples = load_triplet_data(data_path, max_samples)
        
        # Train model
        model = train_model(model_name, examples, device, epochs, batch_size)
        
        # Upload model
        model_path = upload_model(spaces_client, bucket_name, model_name)
        
        if model_path:
            logger.info("🎉 Training completed successfully!")
            logger.info(f"📍 Model: {model_path}")
        else:
            logger.error("❌ Model upload failed!")
            sys.exit(1)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        log_memory_usage("Final cleanup")
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        log_memory_usage("Error state")
        raise

if __name__ == "__main__":
    main()