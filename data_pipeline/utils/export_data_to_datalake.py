# utils/upload_to_minio.py
from minio import Minio
from helpers import load_cfg
import os
from pathlib import Path
import mimetypes

CFG_FILE = "./utils/config.yaml"

def ensure_bucket(client, bucket_name):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name}")
    else:
        print(f"Bucket {bucket_name} exists")

def upload_dir(client, bucket_name, local_dir, dest_prefix):
    """Upload all files recursively from local_dir to bucket_name/dest_prefix/...
    Keeps directory structure relative to local_dir.
    """
    local_dir = Path(local_dir)
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            fp = Path(root) / f
            # object name = dest_prefix + relative path from local_dir
            rel = fp.relative_to(local_dir)
            object_name = str(Path(dest_prefix) / rel).replace("\\", "/")
            content_type, _ = mimetypes.guess_type(str(fp))
            print(f"Uploading {fp} -> s3://{bucket_name}/{object_name}")
            client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(fp),
                content_type=content_type or "application/octet-stream",
            )

def main():
    cfg = load_cfg(CFG_FILE)
    dl = cfg["datalake"]
    local = cfg.get("local", {})
    client = Minio(endpoint=dl["endpoint"], access_key=dl["access_key"],
                   secret_key=dl["secret_key"], secure=dl.get("secure", False))
    ensure_bucket(client, dl["bucket_name"])

    # upload corpus (raw)
    base = Path(local.get("base_raw_path", "./data"))
    # expected local folders
    corpus_local = base / "rag_corpus"
    finetune_local = base / "finetune_data"

    # Upload corpus -> raw prefix
    if corpus_local.exists():
        upload_dir(client, dl["bucket_name"], corpus_local, dl["prefixes"]["raw"]["corpus"])
    else:
        print(f"{corpus_local} not found, skip corpus upload.")

    # Upload finetune -> raw prefix
    if finetune_local.exists():
        upload_dir(client, dl["bucket_name"], finetune_local, dl["prefixes"]["raw"]["finetune"])
    else:
        print(f"{finetune_local} not found, skip finetune upload.")

if __name__ == "__main__":
    main()
