#!/usr/bin/env python3
"""
Script để upload dữ liệu từ local lên S3 bucket
"""

import boto3
import argparse
import os
import json
from typing import Optional
import logging
from tqdm import tqdm

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_to_s3(local_file, bucket, s3_key):
    """Upload file lên S3"""
    s3_client = boto3.client('s3')
    
    try:
        print(f"Uploading {local_file} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(local_file, bucket, s3_key)
        print(f"✅ Upload successful!")
        return True
        
    except FileNotFoundError:
        print(f"❌ File {local_file} not found")
        return False
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        return False
    except ClientError as e:
        print(f"❌ Upload failed: {e}")
        return False

def verify_upload(bucket, s3_key):
    """Verify file đã upload thành công"""
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        size_mb = response['ContentLength'] / (1024 * 1024)
        print(f"✅ File verified: {size_mb:.2f} MB")
        return True
    except ClientError:
        print(f"❌ File not found on S3")
        return False

def main():
    parser = argparse.ArgumentParser(description='Upload corpus data to S3')
    parser.add_argument('--local-file', required=True, help='Local file path')
    parser.add_argument('--bucket', default='legal-datalake', help='S3 bucket name')
    parser.add_argument('--s3-key', default='processed/rag_corpus/merged_corpus.jsonl', help='S3 key')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.local_file):
        print(f"❌ Local file not found: {args.local_file}")
        return
    
    # Upload file
    success = upload_to_s3(args.local_file, args.bucket, args.s3_key)
    
    if success:
        # Verify upload
        verify_upload(args.bucket, args.s3_key)
        print(f"\n🎉 File available at: s3://{args.bucket}/{args.s3_key}")
    else:
        print("❌ Upload failed")

if __name__ == '__main__':
    main()