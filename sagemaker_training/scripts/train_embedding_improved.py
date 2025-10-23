#!/usr/bin/env python3
"""
Improved Script training model embedding cho Vietnamese Legal documents trên SageMaker
"""

import os
import json
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import pandas as pd
import numpy as np
from typing import List, Tuple
import boto3
from sklearn.model_selection import train_test_split
import random

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedLegalTextDataset(Dataset):
    """Improved Dataset cho legal text embedding với better training strategy"""
    
    def __init__(self, data_path: str, max_seq_length: int = 512, max_samples: int = None):
        self.max_seq_length = max_seq_length
        self.texts = self._load_texts(data_path, max_samples)
        logger.info(f"Loaded {len(self.texts)} texts for training")
        
    def _load_texts(self, data_path: str, max_samples: int = None) -> List[str]:
        """Load texts từ JSONL file"""
        texts = []
        
        # Đọc dữ liệu từ S3 hoặc local
        if data_path.startswith('s3://'):
            s3 = boto3.client('s3')
            bucket, key = data_path.replace('s3://', '').split('/', 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            lines = obj['Body'].read().decode('utf-8').strip().split('\n')
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        for i, line in enumerate(lines):
            if max_samples and i >= max_samples:
                break
                
            if line.strip():
                data = json.loads(line.strip())
                text = data.get('text', '').strip()
                
                # Filter text quality
                if text and len(text) > 20 and len(text) < 2000:  # Reasonable length
                    texts.append(text)
        
        logger.info(f"Filtered to {len(texts)} high-quality texts")
        return texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

def create_evaluation_examples(texts: List[str], num_examples: int = 1000) -> List[InputExample]:
    """Tạo evaluation examples để đánh giá model performance"""
    eval_examples = []
    
    # Tạo positive pairs từ text có từ khóa legal chung
    legal_keywords = ['luật', 'pháp', 'quy định', 'điều khoản', 'nghị định', 'thông tư', 'quyết định']
    
    # Group texts by keywords
    keyword_groups = {}
    for text in texts[:num_examples * 2]:  # Limit để tránh quá chậm
        for keyword in legal_keywords:
            if keyword in text.lower():
                if keyword not in keyword_groups:
                    keyword_groups[keyword] = []
                keyword_groups[keyword].append(text)
    
    # Tạo positive pairs trong cùng group
    for keyword, group_texts in keyword_groups.items():
        if len(group_texts) >= 2:
            for i in range(min(50, len(group_texts) - 1)):  # Limit số pairs
                for j in range(i + 1, min(i + 3, len(group_texts))):
                    eval_examples.append(InputExample(
                        texts=[group_texts[i], group_texts[j]], 
                        label=0.8
                    ))
    
    # Tạo negative pairs từ different groups
    group_keys = list(keyword_groups.keys())
    for i in range(min(200, len(eval_examples))):  # Balance positive/negative
        if len(group_keys) >= 2:
            key1, key2 = random.sample(group_keys, 2)
            if keyword_groups[key1] and keyword_groups[key2]:
                text1 = random.choice(keyword_groups[key1])
                text2 = random.choice(keyword_groups[key2])
                eval_examples.append(InputExample(
                    texts=[text1, text2], 
                    label=0.2
                ))
    
    logger.info(f"Created {len(eval_examples)} evaluation examples")
    return eval_examples

def train_model(args):
    """Improved training function"""
    
    # Khởi tạo model base
    logger.info(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    
    # Set max sequence length
    model.max_seq_length = args.max_seq_length
    
    # Load dữ liệu training
    logger.info(f"Loading training data from: {args.data_path}")
    dataset = ImprovedLegalTextDataset(
        args.data_path, 
        args.max_seq_length,
        max_samples=args.max_samples
    )
    
    # Split train/validation
    train_texts, val_texts = train_test_split(
        dataset.texts, 
        test_size=0.1,  # 10% cho validation
        random_state=42
    )
    
    logger.info(f"Train texts: {len(train_texts)}")
    logger.info(f"Validation texts: {len(val_texts)}")
    
    # Tạo evaluation examples
    eval_examples = create_evaluation_examples(val_texts, num_examples=500)
    
    # Sử dụng MultipleNegativesRankingLoss thay vì CosineSimilarityLoss
    # Approach này tốt hơn cho domain-specific fine-tuning
    train_dataloader = DataLoader(
        train_texts, 
        shuffle=True, 
        batch_size=args.batch_size
    )
    
    # MultipleNegativesRankingLoss: mỗi text trong batch là positive với chính nó
    # và negative với tất cả texts khác trong batch
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Evaluator cho validation
    evaluator = None
    if eval_examples:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples, 
            name='vietnamese-legal-eval'
        )
    
    # Training với improved configuration
    logger.info("Starting improved training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=args.evaluation_steps,
        warmup_steps=args.warmup_steps,
        output_path=args.model_dir,
        save_best_model=True,
        optimizer_params={
            'lr': args.learning_rate,
        },
        scheduler='WarmupLinear',
        use_amp=True,  # Automatic Mixed Precision để training nhanh hơn
        show_progress_bar=True
    )
    
    # Save final model
    logger.info(f"Saving model to: {args.model_dir}")
    model.save(args.model_dir)
    
    # Upload to S3 if specified
    if args.s3_output_path:
        upload_model_to_s3(args.model_dir, args.s3_output_path)
    
    # Test model với sample legal queries
    test_model_performance(model)
    
    logger.info("Training completed!")

def test_model_performance(model):
    """Test model với legal queries"""
    test_queries = [
        "luật pháp việt nam",
        "quy định về hợp đồng lao động", 
        "nghị định về đầu tư",
        "thủ tục hành chính",
        "quyền và nghĩa vụ của công dân"
    ]
    
    logger.info("Testing model performance:")
    for query in test_queries:
        try:
            embedding = model.encode(query)
            logger.info(f"   '{query}' -> embedding shape: {embedding.shape}")
        except Exception as e:
            logger.error(f"   Error encoding '{query}': {e}")

def upload_model_to_s3(local_path: str, s3_path: str):
    """Upload trained model to S3"""
    s3 = boto3.client('s3')
    bucket, key_prefix = s3_path.replace('s3://', '').split('/', 1)
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = f"{key_prefix}/{relative_path}"
            
            logger.info(f"Uploading {local_file} to s3://{bucket}/{s3_key}")
            s3.upload_file(local_file, bucket, s3_key)

def main():
    parser = argparse.ArgumentParser(description='Train Vietnamese Legal Embedding Model - Improved Version')
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-path', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', 's3://legal-datalake/processed/rag_corpus/merged_corpus.jsonl'))
    parser.add_argument('--s3-output-path', type=str, default='s3://legal-datalake/models/embedding/')
    
    # Model parameters - IMPROVED
    parser.add_argument('--base-model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    parser.add_argument('--max-seq-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=48)  
    parser.add_argument('--epochs', type=int, default=6)       
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--evaluation-steps', type=int, default=500)
    parser.add_argument('--max-samples', type=int, default=None, help='Limit số samples để test')
    
    args = parser.parse_args()
    
    logger.info(f"Improved training arguments: {args}")
    
    # Kiểm tra GPU
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("Training on CPU")
    
    train_model(args)

if __name__ == '__main__':
    main()