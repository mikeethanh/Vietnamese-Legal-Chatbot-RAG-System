#!/usr/bin/env python3
"""
Inference script cho SageMaker endpoint - model embedding
"""

import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load model từ model directory"""
    try:
        model = SentenceTransformer(model_dir)
        logger.info(f"Model loaded successfully from {model_dir}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def input_fn(request_body, request_content_type='application/json'):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Expect format: {"texts": ["text1", "text2", ...]}
        if 'texts' in input_data:
            return input_data['texts']
        elif 'text' in input_data:
            return [input_data['text']]
        else:
            raise ValueError("Input must contain 'texts' or 'text' field")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Generate embeddings"""
    try:
        # Generate embeddings
        embeddings = model.encode(
            input_data,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        return {
            'embeddings': embeddings_list,
            'count': len(embeddings_list),
            'dimension': len(embeddings_list[0]) if embeddings_list else 0
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def output_fn(prediction, content_type='application/json'):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")