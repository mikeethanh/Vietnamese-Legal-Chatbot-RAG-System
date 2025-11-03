#!/usr/bin/env python3

import json
import logging
import os
import random
import sys
import time
from datetime import datetime

import boto3
import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("/tmp/logs/training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def check_gpu():
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        sys.exit(1)

    device = torch.device("cuda")
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
    access_key = os.getenv("SPACES_ACCESS_KEY")
    secret_key = os.getenv("SPACES_SECRET_KEY")
    endpoint = os.getenv("SPACES_ENDPOINT", "https://sfo3.digitaloceanspaces.com")

    if not access_key or not secret_key:
        logger.error("❌ SPACES_ACCESS_KEY and SPACES_SECRET_KEY required!")
        sys.exit(1)

    region = "sfo3"

    client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        region_name=region,
    )

    logger.info(f"✅ Connected to Spaces: {endpoint}")
    return client


def download_data(spaces_client, bucket_name):
    """Download training data from Spaces"""
    data_path = "/tmp/data/law_vi.jsonl"
    os.makedirs("/tmp/data", exist_ok=True)

    try:
        logger.info("📥 Downloading triplet training data...")
        spaces_client.download_file(
            bucket_name, "process_data/embed/law_vi.jsonl", data_path
        )
        logger.info(f"✅ Downloaded to {data_path}")
        return data_path
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)


def load_triplet_data(data_path, max_samples=None):
    """Load triplet data from JSONL file and convert to triplet examples for TripletLoss"""
    logger.info(f"📖 Loading triplet data from {data_path}")

    examples = []

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and len(examples) >= max_samples:
                    break

                try:
                    data = json.loads(line.strip())

                    # Validate required fields
                    if not all(
                        key in data for key in ["query", "positive", "hard_neg"]
                    ):
                        logger.warning(
                            f"Line {line_num}: Missing required fields (query, positive, hard_neg)"
                        )
                        continue

                    query = data["query"].strip()
                    positive = data["positive"].strip()
                    hard_neg = data["hard_neg"].strip()

                    # Skip empty texts
                    if not query or not positive or not hard_neg:
                        logger.warning(f"Line {line_num}: Empty text found, skipping")
                        continue

                    # Create triplet example for TripletLoss
                    example = InputExample(texts=[query, positive, hard_neg])
                    examples.append(example)

                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error - {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Line {line_num}: Unexpected error - {e}")
                    continue

                # Log progress every 10000 lines
                if line_num % 10000 == 0:
                    logger.info(
                        f"Processed {line_num} lines, created {len(examples)} examples"
                    )
        logger.info(f"✅ Loaded {len(examples)} triplet examples")
        return examples

    except FileNotFoundError:
        logger.error(f"❌ File not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        sys.exit(1)


def evaluate_baseline(model, val_examples, device):
    """Evaluate baseline model performance before training"""
    logger.info("📊 Evaluating baseline model...")

    # Sample evaluation set (use smaller sample to save time)
    eval_examples = random.sample(val_examples, min(5000, len(val_examples)))

    # Prepare data for TripletEvaluator
    anchors = []
    positives = []
    negatives = []

    for example in eval_examples:
        anchors.append(example.texts[0])  # query
        positives.append(example.texts[1])  # positive
        negatives.append(example.texts[2])  # hard_neg

    # Create evaluator
    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="baseline_evaluation",
    )

    # Evaluate
    baseline_result = evaluator(model, output_path="/tmp/baseline_eval")

    # Extract accuracy from result (TripletEvaluator returns a dict or float)
    if isinstance(baseline_result, dict):
        baseline_score = baseline_result.get(
            "accuracy", baseline_result.get("cosine_accuracy", 0.0)
        )
    else:
        baseline_score = baseline_result

    logger.info(f"✅ Baseline Accuracy: {baseline_score:.4f}")
    logger.info(
        f"   (Percentage of cases where anchor is closer to positive than negative)"
    )

    return baseline_score


def train_model(model_name, examples, device, epochs=5, batch_size=64):
    """Train the embedding model with triplet loss and evaluation"""
    logger.info(f"🤖 Loading model: {model_name}")

    # Clear ALL GPU memory first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load model with memory optimization
    try:
        model = SentenceTransformer(model_name, device=device)

        # Disable gradient checkpointing for faster training
        use_gradient_checkpointing = (
            os.getenv("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"
        )
        if (
            use_gradient_checkpointing
            and hasattr(model[0], "auto_model")
            and hasattr(model[0].auto_model, "gradient_checkpointing_enable")
        ):
            model[0].auto_model.gradient_checkpointing_enable()
            logger.info("✅ Gradient checkpointing enabled")
        else:
            logger.info("ℹ️ Gradient checkpointing disabled (faster training)")

        logger.info("✅ Model loaded successfully")

        # Log memory usage after model loading
        log_memory_usage("After model load")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        torch.cuda.empty_cache()
        sys.exit(1)

    # Split data for training and evaluation (8:2 ratio as requested)
    train_examples, val_examples = train_test_split(
        examples, test_size=0.2, random_state=42
    )

    logger.info(f"📊 Training examples: {len(train_examples)}")
    logger.info(f"📊 Validation examples: {len(val_examples)}")

    # Evaluate baseline model before training
    logger.info("=" * 80)
    logger.info("🔍 BASELINE EVALUATION (Before Training)")
    logger.info("=" * 80)
    baseline_score = evaluate_baseline(model, val_examples, device)
    logger.info("=" * 80)

    # Get num_workers from environment variable or use default
    num_workers = int(os.getenv("DATALOADER_NUM_WORKERS", "4"))

    # Create DataLoader for training
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )

    # Define TripletLoss
    train_loss = losses.TripletLoss(model=model)

    # Prepare evaluation data for TripletEvaluator
    anchors = []
    positives = []
    negatives = []

    # Take first 5000 validation examples for evaluation to avoid memory issues
    eval_examples = random.sample(val_examples, min(5000, len(val_examples)))

    for example in eval_examples:
        anchors.append(example.texts[0])  # query
        positives.append(example.texts[1])  # positive
        negatives.append(example.texts[2])  # hard_neg

    # Create TripletEvaluator
    evaluator = TripletEvaluator(
        anchors=anchors,
        positives=positives,
        negatives=negatives,
        name="triplet_evaluation",
    )

    # Training with memory optimization
    logger.info(f"🔥 Starting training...")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Num workers: {num_workers}")
    logger.info(f"   Training samples: {len(train_examples)}")
    logger.info(f"   Evaluation samples: {len(eval_examples)}")
    logger.info(f"   Loss function: TripletLoss")
    logger.info(f"   Evaluator: TripletEvaluator")

    start_time = time.time()

    # Clear cache before training
    torch.cuda.empty_cache()

    try:
        # Get additional memory optimization settings
        max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "256"))

        # Set max sequence length for memory optimization
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = max_seq_length
            logger.info(f"✅ Set max_seq_length = {max_seq_length}")

        # Improved learning rate
        learning_rate = float(os.getenv("LEARNING_RATE", "1e-5"))

        logger.info(f"📊 Training settings:")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Warmup steps: {int(len(train_dataloader) * 0.1)}")
        logger.info(f"   Evaluation steps: {max(200, len(train_dataloader) // 4)}")

        # Start training
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=max(200, len(train_dataloader) // 4),
            warmup_steps=int(len(train_dataloader) * 0.1),
            optimizer_params={"lr": learning_rate},
            output_path="/tmp/model",
            save_best_model=True,
            show_progress_bar=True,
        )

        # Clear cache after training
        torch.cuda.empty_cache()
        logger.info("✅ Training completed, memory cleared")

        # Evaluate final model performance
        logger.info("=" * 80)
        logger.info("🔍 FINAL EVALUATION (After Training)")
        logger.info("=" * 80)
        final_score = evaluate_baseline(model, val_examples, device)
        logger.info("=" * 80)

        # Compare baseline vs final
        improvement = final_score - baseline_score
        improvement_pct = (
            (improvement / baseline_score) * 100 if baseline_score > 0 else 0
        )

        logger.info("=" * 80)
        logger.info("📈 PERFORMANCE COMPARISON")
        logger.info("=" * 80)
        logger.info(f"   Baseline Score: {baseline_score:.4f}")
        logger.info(f"   Final Score:    {final_score:.4f}")
        logger.info(f"   Improvement:    {improvement:+.4f} ({improvement_pct:+.2f}%)")
        logger.info("=" * 80)

        # Save comparison to file
        comparison_path = "/tmp/model/performance_comparison.txt"
        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write("PERFORMANCE COMPARISON\n")
            f.write("=" * 80 + "\n")
            f.write(f"Baseline Score: {baseline_score:.4f}\n")
            f.write(f"Final Score:    {final_score:.4f}\n")
            f.write(f"Improvement:    {improvement:+.4f} ({improvement_pct:+.2f}%)\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        logger.info(f"💾 Performance comparison saved to {comparison_path}")

    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"💥 CUDA OOM Error: {e}")
        logger.error("💡 Memory Debugging Info:")
        logger.error(f"   - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.error(f"   - Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        logger.error("💡 Recommendations:")
        logger.error(f"   - Current batch_size: {batch_size} → try batch_size=16")
        logger.error(f"   - Reduce max_seq_length or eval corpus size")
        logger.error(f"   - Try using gradient checkpointing")

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

    return model, baseline_score, final_score


def upload_model(spaces_client, bucket_name, model_name):
    """Upload trained model to Spaces"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_prefix = f"models/embedding_model_gpu_{timestamp}"

    logger.info(f"📤 Uploading model to {s3_prefix}...")

    model_dir = "/tmp/model"

    # Check if model directory exists
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

    # Create and upload metadata
    metadata = {
        "model_name": f"embedding_model_gpu_{timestamp}",
        "base_model": model_name,
        "training_date": timestamp,
        "gpu_used": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "Unknown"
        ),
        "model_path": s3_prefix,
        "uploaded_files": uploaded_files,
        "training_data": "law_vi.jsonl (triplet format)",
    }

    metadata_path = os.path.join(model_dir, "metadata.json")
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        spaces_client.upload_file(
            metadata_path, bucket_name, f"{s3_prefix}/metadata.json"
        )
        logger.info("✅ Metadata uploaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to upload metadata: {e}")

    logger.info(f"🎉 Model uploaded successfully!")
    logger.info(f"📍 Model path: {s3_prefix}")

    return s3_prefix


def main():
    """Main training function"""
    logger.info("🚀 Starting Vietnamese Legal Embedding Training with Triplet Data")

    # Set memory optimization GLOBALLY before anything else
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

    # Configuration from environment
    model_name = os.getenv("BASE_MODEL", "BAAI/bge-m3")
    epochs = int(os.getenv("EPOCHS", "3"))
    batch_size = int(os.getenv("GPU_BATCH_SIZE", "32"))
    max_samples = (
        int(os.getenv("MAX_SAMPLES", "30000")) if os.getenv("MAX_SAMPLES") else 30000
    )
    bucket_name = os.getenv("SPACES_BUCKET", "legal-datalake")

    logger.info(f"📋 Configuration:")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Bucket: {bucket_name}")
    logger.info(f"   Data format: Triplet (query, positive, hard_neg)")
    logger.info(f"   Loss function: TripletLoss")
    logger.info(f"   Train/Eval split: 9:1")
    logger.info(f"   Memory optimization: ENABLED")
    logger.info(f"   Data source: Digital Ocean Spaces")

    try:
        # Setup GPU and clear memory first
        device = check_gpu()

        # Clear all GPU memory before starting
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        spaces_client = setup_spaces_client()

        # Create directories
        os.makedirs("/tmp/data", exist_ok=True)
        os.makedirs("/tmp/model", exist_ok=True)
        os.makedirs("/tmp/logs", exist_ok=True)

        # Download data from Spaces
        data_path = download_data(spaces_client, bucket_name)

        # Load triplet data and prepare examples
        examples = load_triplet_data(data_path, max_samples)

        # Train model
        model, baseline_score, final_score = train_model(
            model_name, examples, device, epochs, batch_size
        )

        # Upload model
        model_path = upload_model(spaces_client, bucket_name, model_name)

        if model_path is None:
            logger.error("❌ Model upload failed!")
            sys.exit(1)

        # Final memory cleanup and logging
        del model
        torch.cuda.empty_cache()
        log_memory_usage("Final cleanup")

        # Print final summary
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📍 Model location: {model_path}")
        logger.info(f"📊 Baseline Score: {baseline_score:.4f}")
        logger.info(f"� Final Score:    {final_score:.4f}")
        improvement = final_score - baseline_score
        improvement_pct = (
            (improvement / baseline_score) * 100 if baseline_score > 0 else 0
        )
        logger.info(f"📈 Improvement:    {improvement:+.4f} ({improvement_pct:+.2f}%)")
        logger.info("=" * 80)

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
