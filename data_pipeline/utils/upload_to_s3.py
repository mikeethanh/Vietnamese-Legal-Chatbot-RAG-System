import boto3
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
LOCAL_DIRS = os.getenv("LOCAL_DIRS").split(",")

s3 = boto3.client("s3")

def upload_folder(local_dir, bucket, s3_prefix="raw"):
    folder_name = os.path.basename(local_dir.rstrip("/"))
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            
            s3_key = os.path.join(s3_prefix, folder_name, relative_path).replace("\\", "/")
            
            print(f"⬆️ Uploading {local_path} → s3://{bucket}/{s3_key}")
            s3.upload_file(local_path, bucket, s3_key)
    
    print(f"✅ Finished uploading folder: {local_dir}")

if __name__ == "__main__":
    for folder in LOCAL_DIRS:
        upload_folder(folder, BUCKET_NAME)
