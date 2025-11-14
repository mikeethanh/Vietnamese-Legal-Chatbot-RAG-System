#!/usr/bin/env python3
"""
Flask API đơn giản để serving embedding model trên CPU
Endpoints: /health, /embed, /similarity
"""

import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global model variable
model = None
model_loaded = False


def load_model():
    """Load embedding model"""
    global model, model_loaded

    model_path = os.getenv("MODEL_PATH", "./models")

    logger.info(f"📥 Loading model from: {model_path}")

    try:
        # Force CPU usage
        device = "cpu"
        logger.info(f"💻 Using device: {device}")

        # Load model
        model = SentenceTransformer(model_path, device=device)

        # Test model with dummy text
        test_embedding = model.encode(["test"], show_progress_bar=False)
        embedding_dim = test_embedding.shape[1]

        model_loaded = True
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Embedding dimension: {embedding_dim}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model_loaded = False
        return False


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "device": "cpu",
        "timestamp": time.time(),
    }

    if model_loaded and model is not None:
        status["embedding_dim"] = model.get_sentence_embedding_dimension()

    return jsonify(status), 200 if model_loaded else 503


@app.route("/embed", methods=["POST"])
def embed():
    """
    Embedding endpoint
    Input: {"texts": ["text1", "text2", ...]}
    Output: {"embeddings": [[...], [...]], "embedding_dim": 1024}
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Parse request
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400

        texts = data["texts"]

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "'texts' must be a non-empty list"}), 400

        # Limit batch size
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))
        if len(texts) > max_batch_size:
            return (
                jsonify({"error": f"Batch size exceeds limit. Max: {max_batch_size}"}),
                400,
            )

        # Generate embeddings
        start_time = time.time()
        embeddings = model.encode(
            texts, batch_size=16, show_progress_bar=False, convert_to_numpy=True
        )
        inference_time = time.time() - start_time

        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()

        response = {
            "embeddings": embeddings_list,
            "embedding_dim": embeddings.shape[1],
            "num_texts": len(texts),
            "inference_time": round(inference_time, 3),
        }

        logger.info(f"✅ Embedded {len(texts)} texts in {inference_time:.3f}s")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Embedding error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/similarity", methods=["POST"])
def similarity():
    """
    Similarity endpoint
    Input: {"texts1": ["text1", ...], "texts2": ["text2", ...]}
    Output: {"similarities": [[0.9, 0.8], ...], "shape": [2, 2]}
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Parse request
        data = request.get_json()

        if not data or "texts1" not in data or "texts2" not in data:
            return jsonify({"error": "Missing 'texts1' or 'texts2' field"}), 400

        texts1 = data["texts1"]
        texts2 = data["texts2"]

        if not isinstance(texts1, list) or not isinstance(texts2, list):
            return jsonify({"error": "'texts1' and 'texts2' must be lists"}), 400

        if len(texts1) == 0 or len(texts2) == 0:
            return jsonify({"error": "Input lists cannot be empty"}), 400

        # Limit batch size
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))
        if len(texts1) > max_batch_size or len(texts2) > max_batch_size:
            return (
                jsonify({"error": f"Batch size exceeds limit. Max: {max_batch_size}"}),
                400,
            )

        # Generate embeddings
        start_time = time.time()

        embeddings1 = model.encode(
            texts1, batch_size=16, show_progress_bar=False, convert_to_numpy=True
        )

        embeddings2 = model.encode(
            texts2, batch_size=16, show_progress_bar=False, convert_to_numpy=True
        )

        # Calculate cosine similarity
        # Normalize vectors
        embeddings1_norm = embeddings1 / np.linalg.norm(
            embeddings1, axis=1, keepdims=True
        )
        embeddings2_norm = embeddings2 / np.linalg.norm(
            embeddings2, axis=1, keepdims=True
        )

        # Compute similarity matrix
        similarities = np.matmul(embeddings1_norm, embeddings2_norm.T)

        inference_time = time.time() - start_time

        response = {
            "similarities": similarities.tolist(),
            "shape": list(similarities.shape),
            "inference_time": round(inference_time, 3),
        }

        logger.info(
            f"✅ Computed similarity for {len(texts1)}x{len(texts2)} in {inference_time:.3f}s"
        )

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"❌ Similarity error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API info"""
    return (
        jsonify(
            {
                "service": "Vietnamese Legal Embedding API",
                "version": "1.0.0",
                "endpoints": {
                    "/health": "Health check",
                    "/embed": "Generate embeddings (POST)",
                    "/similarity": "Compute similarity (POST)",
                },
                "status": "ready" if model_loaded else "loading",
            }
        ),
        200,
    )


def main():
    """Main function to start the server"""
    logger.info("🚀 Starting Vietnamese Legal Embedding Serving API")

    # Load model first
    if not load_model():
        logger.error("❌ Failed to load model. Exiting.")
        return

    # Get config from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"🌐 Starting server on {host}:{port}")
    logger.info(f"🔧 Debug mode: {debug}")

    # Start Flask server
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True,  # Enable multi-threading for better performance
    )


if __name__ == "__main__":
    main()