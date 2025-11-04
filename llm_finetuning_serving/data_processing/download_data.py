"""
Script để tải dữ liệu từ HuggingFace Spaces
"""

import os
import json
import requests
from pathlib import Path
from typing import Optional
import time

def download_from_hf_spaces(
    space_name: str,
    file_path: str,
    output_path: str,
    hf_token: Optional[str] = None
) -> bool:
    """
    Download file from HuggingFace Spaces
    
    Args:
        space_name: HF space name (e.g., "username/space-name")
        file_path: Path to file in the space
        output_path: Local path to save file
        hf_token: HuggingFace token (optional)
    
    Returns:
        bool: Success status
    """
    
    # Construct download URL
    base_url = f"https://huggingface.co/spaces/{space_name}/resolve/main/{file_path}"
    
    # Headers
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    try:
        print(f"Downloading from: {base_url}")
        print(f"Saving to: {output_path}")
        
        # Make request
        response = requests.get(base_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download with progress
        file_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if file_size > 0:
                        progress = (downloaded / file_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)
        
        print(f"\n✅ Download completed: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main download function"""
    
    # Configuration
    SPACE_NAME = "legal-datalake/process_data"  # Replace with your actual space
    FILE_PATH = "finetune_data/finetune_llm_data.jsonl"
    OUTPUT_PATH = "./data_processing/raw_data/finetune_llm_data.jsonl"
    
    # Get HF token from environment
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    if not HF_TOKEN:
        print("⚠️  Warning: HF_TOKEN not found in environment variables")
        print("   Some files may not be accessible without authentication")
    
    # Download file
    success = download_from_hf_spaces(
        space_name=SPACE_NAME,
        file_path=FILE_PATH,
        output_path=OUTPUT_PATH,
        hf_token=HF_TOKEN
    )
    
    if success:
        # Verify download
        if os.path.exists(OUTPUT_PATH):
            file_size = os.path.getsize(OUTPUT_PATH)
            print(f"📊 File size: {file_size / (1024*1024):.1f} MB")
            
            # Quick validation - count lines
            try:
                with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"📝 Number of lines: {line_count:,}")
                
                # Check first few lines
                with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        try:
                            data = json.loads(line)
                            print(f"✅ Line {i+1} JSON valid")
                        except json.JSONDecodeError:
                            print(f"❌ Line {i+1} JSON invalid")
                            
            except Exception as e:
                print(f"⚠️  Validation error: {e}")
        
        print(f"\n🎉 Data download completed successfully!")
        print(f"   Next steps:")
        print(f"   1. Run: python data_processing/analyze_data.py")
        print(f"   2. Run: python data_processing/process_llama_data.py")
        
    else:
        print(f"\n❌ Download failed!")
        print(f"   Alternative: Copy file manually from your local data_pipeline directory")
        
        # Fallback - copy from local if exists
        local_path = "../data_pipeline/data/finetune_llm/finetune_llm_data.jsonl"
        if os.path.exists(local_path):
            print(f"   Found local file: {local_path}")
            
            import shutil
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            shutil.copy2(local_path, OUTPUT_PATH)
            print(f"   ✅ Copied local file to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()