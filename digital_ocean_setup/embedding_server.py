#!/usr/bin/env python3
"""
API Server để serving embedding model trên Digital Ocean
"""

import os
import json
import logging
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import boto3
from datetime import datetime
import zipfile
import tempfile

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EmbeddingServer:
    """Embedding model serving server"""
    
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_info = {}
        logger.info(f"Using device: {self.device}")
        
        # Initialize Spaces client
        self.spaces_client = self._init_spaces_client()
        
    def _init_spaces_client(self):
        """Initialize Digital Ocean Spaces client"""
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('SPACES_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('SPACES_SECRET_KEY'),
            endpoint_url=os.getenv('SPACES_ENDPOINT', 'https://sgp1.digitaloceanspaces.com'),
            region_name='sgp1'
        )
        
    def download_model(self, model_s3_path: str, local_dir: str = '/tmp/model'):
        """Download model từ Spaces"""
        try:
            bucket_name = os.getenv('SPACES_BUCKET', 'legal-datalake')
            
            # List all objects trong model path
            response = self.spaces_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=model_s3_path
            )
            
            if 'Contents' not in response:
                raise ValueError(f"No model found at {model_s3_path}")
                
            os.makedirs(local_dir, exist_ok=True)
            
            # Download all files
            for obj in response['Contents']:
                s3_key = obj['Key']
                local_path = os.path.join(local_dir, os.path.basename(s3_key))
                
                self.spaces_client.download_file(bucket_name, s3_key, local_path)
                logger.info(f"Downloaded {s3_key} to {local_path}")
                
            return local_dir
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
            
    def load_model(self, model_path: str):
        """Load model từ local path"""
        try:
            self.model = SentenceTransformer(model_path)
            self.model.to(self.device)
            
            # Load metadata nếu có
            metadata_path = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_info = json.load(f)
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts thành embeddings"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise

# Global server instance
embedding_server = EmbeddingServer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': embedding_server.model is not None,
        'device': str(embedding_server.device),
        'timestamp': datetime.now().isoformat()
    }
    
    if embedding_server.model_info:
        status['model_info'] = embedding_server.model_info
        
    return jsonify(status)

@app.route('/embed', methods=['POST'])
def embed_texts():
    """Endpoint để encode texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Missing texts field'}), 400
            
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
            
        if len(texts) == 0:
            return jsonify({'error': 'texts list cannot be empty'}), 400
            
        # Encode texts
        embeddings = embedding_server.encode_texts(texts)
        
        response = {
            'embeddings': embeddings.tolist(),
            'num_texts': len(texts),
            'embedding_dim': embeddings.shape[1]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in embed_texts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/similarity', methods=['POST'])
def compute_similarity():
    """Endpoint để tính similarity giữa texts"""
    try:
        data = request.get_json()
        
        if not data or 'texts1' not in data or 'texts2' not in data:
            return jsonify({'error': 'Missing texts1 or texts2 field'}), 400
            
        texts1 = data['texts1']
        texts2 = data['texts2']
        
        if not isinstance(texts1, list) or not isinstance(texts2, list):
            return jsonify({'error': 'texts1 and texts2 must be lists'}), 400
            
        # Encode both sets
        embeddings1 = embedding_server.encode_texts(texts1)
        embeddings2 = embedding_server.encode_texts(texts2)
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings1, embeddings2)
        
        response = {
            'similarities': similarities.tolist(),
            'shape': similarities.shape
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in compute_similarity: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint để load model từ Spaces"""
    try:
        data = request.get_json()
        
        if not data or 'model_path' not in data:
            return jsonify({'error': 'Missing model_path field'}), 400
            
        model_s3_path = data['model_path']
        
        # Download và load model
        local_model_path = embedding_server.download_model(model_s3_path)
        embedding_server.load_model(local_model_path)
        
        response = {
            'status': 'success',
            'message': f'Model loaded from {model_s3_path}',
            'model_info': embedding_server.model_info
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup nếu có
    model_path = os.getenv('MODEL_PATH')
    if model_path:
        try:
            if model_path.startswith('models/'):
                # Load from Spaces
                local_path = embedding_server.download_model(model_path)
                embedding_server.load_model(local_path)
            else:
                # Load from local path
                embedding_server.load_model(model_path)
        except Exception as e:
            logger.warning(f"Could not load model at startup: {e}")
    
    # Start server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)