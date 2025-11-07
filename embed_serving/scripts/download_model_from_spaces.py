#!/usr/bin/env python3
"""
Script để download model cho serving:
- Download baseline model từ Hugging Face (RECOMMENDED)
"""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_baseline_model(model_name, local_dir="./models"):
    """
    Download baseline model từ Hugging Face

    Args:
        model_name: Tên model trên Hugging Face (ví dụ: 'BAAI/bge-m3')
        local_dir: Thư mục lưu model
    """
    logger.info("=" * 70)
    logger.info("🎯 DOWNLOADING BASELINE MODEL (RECOMMENDED)")
    logger.info("=" * 70)
    logger.info(f"📦 Model: {model_name}")
    logger.info(f"📁 Local directory: {local_dir}")

    try:
        # Tạo thư mục nếu chưa tồn tại
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"⬇️  Downloading from Hugging Face...")

        # Download model từ Hugging Face
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        logger.info("✅ Baseline model downloaded successfully!")
        logger.info(f"📍 Model path: {local_dir}")
        logger.info("")
        logger.info("💡 Model này chưa được fine-tune, phù hợp cho serving")
        logger.info("   vì baseline model có performance tốt hơn fine-tuned model.")

        return True

    except Exception as e:
        logger.error(f"❌ Lỗi khi download baseline model: {e}")
        return False


def main():
    """
    Main function để download model cho serving
    """
    # Configuration
    MODEL_NAME = "BAAI/bge-m3"  # Baseline model được recommend
    LOCAL_DIR = "./models/bge-m3"

    logger.info("🚀 Starting model download process...")
    logger.info("")

    # Download baseline model
    success = download_baseline_model(MODEL_NAME, LOCAL_DIR)

    if success:
        logger.info("")
        logger.info("=" * 70)
        logger.info("✨ DOWNLOAD COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"📂 Model location: {LOCAL_DIR}")
        logger.info("")
        logger.info("📋 Next steps:")
        logger.info("   1. Sử dụng model này cho serving với serve_model.py")
        logger.info("   2. Model đã sẵn sàng để phục vụ requests")
        logger.info("")
    else:
        logger.error("")
        logger.error("=" * 70)
        logger.error("❌ DOWNLOAD FAILED")
        logger.error("=" * 70)
        logger.error("Vui lòng kiểm tra lại network và thử lại.")
        exit(1)


if __name__ == "__main__":
    main()
