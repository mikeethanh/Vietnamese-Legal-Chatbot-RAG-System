#!/usr/bin/env python3
"""
Script đơn giản để download model đã train từ DigitalOcean Spaces về local
"""

import os
import sys
import boto3
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_spaces_client():
    """Setup Digital Ocean Spaces client"""
    access_key = os.getenv('SPACES_ACCESS_KEY','DO00DY94RFHXGEX7BK2B')
    secret_key = os.getenv('SPACES_SECRET_KEY','qxssv4A4kBvDu5GHIns+CzBRVPN8CDNfH8o5AXyHf7s')
    endpoint = os.getenv('SPACES_ENDPOINT', 'https://sfo3.digitaloceanspaces.com')
    
    if not access_key or not secret_key:
        logger.error("❌ Cần set SPACES_ACCESS_KEY và SPACES_SECRET_KEY!")
        sys.exit(1)
    
    # Extract region from endpoint
    region = 'sgp1' if 'sgp1' in endpoint else 'sfo3'
    
    client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        region_name=region
    )
    
    logger.info(f"✅ Đã kết nối với Spaces: {endpoint}")
    return client

def list_available_models(spaces_client, bucket_name):
    """List all available models in Spaces"""
    logger.info(f"📋 Liệt kê models có sẵn trong bucket '{bucket_name}'...")
    
    try:
        response = spaces_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='models/',
            Delimiter='/'
        )
        
        if 'CommonPrefixes' not in response:
            logger.warning("⚠️ Không tìm thấy model nào trong Spaces!")
            return []
        
        models = []
        for prefix in response['CommonPrefixes']:
            model_path = prefix['Prefix'].rstrip('/')
            models.append(model_path)
        
        logger.info(f"✅ Tìm thấy {len(models)} model(s):")
        for i, model in enumerate(models, 1):
            logger.info(f"   {i}. {model}")
        
        return models
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi list models: {e}")
        return []

def download_model(spaces_client, bucket_name, model_prefix, local_dir='./models'):
    """Download model từ Spaces về local"""
    logger.info(f"📥 Đang download model từ '{model_prefix}'...")
    
    # Tạo thư mục local
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # List all files trong model prefix
        response = spaces_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=model_prefix
        )
        
        if 'Contents' not in response:
            logger.error(f"❌ Không tìm thấy file nào trong '{model_prefix}'")
            return False
        
        files = response['Contents']
        logger.info(f"📦 Tìm thấy {len(files)} file(s) để download")
        
        # Download từng file
        downloaded_count = 0
        for obj in files:
            s3_key = obj['Key']
            
            # Skip nếu là folder
            if s3_key.endswith('/'):
                continue
            
            # Tạo relative path
            relative_path = s3_key.replace(model_prefix + '/', '')
            local_file_path = local_path / relative_path
            
            # Tạo parent directories
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            logger.info(f"   Downloading: {relative_path}")
            spaces_client.download_file(
                bucket_name,
                s3_key,
                str(local_file_path)
            )
            downloaded_count += 1
        
        logger.info(f"✅ Đã download {downloaded_count} file(s) vào '{local_path}'")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi download model: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Bắt đầu download model từ DigitalOcean Spaces")
    
    # Lấy config từ environment
    bucket_name = os.getenv('SPACES_BUCKET', 'legal-datalake')
    model_path = os.getenv('MODEL_PATH', '')
    local_dir = os.getenv('LOCAL_MODEL_DIR', './models')
    
    logger.info(f"📋 Configuration:")
    logger.info(f"   Bucket: {bucket_name}")
    logger.info(f"   Local dir: {local_dir}")
    
    # Setup client
    spaces_client = setup_spaces_client()
    
    # List available models
    available_models = list_available_models(spaces_client, bucket_name)
    
    if not available_models:
        logger.error("❌ Không có model nào để download!")
        sys.exit(1)
    
    # Nếu không chỉ định MODEL_PATH, sử dụng model mới nhất
    if not model_path:
        model_path = available_models[-1]  # Lấy model cuối (mới nhất)
        logger.info(f"💡 Tự động chọn model mới nhất: {model_path}")
    else:
        # Kiểm tra xem model_path có trong danh sách không
        if model_path not in available_models:
            logger.warning(f"⚠️ Model '{model_path}' không tồn tại!")
            logger.info("💡 Sử dụng model mới nhất thay thế")
            model_path = available_models[-1]
    
    # Download model
    success = download_model(spaces_client, bucket_name, model_path, local_dir)
    
    if success:
        logger.info("🎉 Download model thành công!")
        logger.info(f"📍 Model path: {local_dir}")
    else:
        logger.error("❌ Download model thất bại!")
        sys.exit(1)

if __name__ == "__main__":
    main()
