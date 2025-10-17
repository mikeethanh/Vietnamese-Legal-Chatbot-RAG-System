#!/usr/bin/env python3
"""
Script to process finetune data for LLaMA 13B training
Converts CSV files to standardized JSONL format
"""
import pandas as pd
import json
import os
from pathlib import Path
import argparse

def clean_text(text):
    """Clean and normalize text content"""
    if pd.isna(text):
        return ""
    
    # Convert to string and clean
    text = str(text).strip()
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Handle list-like strings from context fields
    if text.startswith("['") and text.endswith("']"):
        try:
            # Try to parse as Python list
            import ast
            text_list = ast.literal_eval(text)
            if isinstance(text_list, list):
                text = ' '.join(text_list)
        except:
            # If parsing fails, remove the brackets manually
            text = text[2:-2]  # Remove [' and ']
    
    return text.strip()

def process_large_vi_legal_queries(file_path):
    """Process large_vi_legal_queries.csv file"""
    print(f"📄 Processing {file_path}...")
    
    # Read CSV with proper encoding
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"   📊 Total rows: {len(df):,}")
    
    # Check required columns
    if 'context' not in df.columns or 'query' not in df.columns:
        print(f"   ❌ Missing required columns. Available: {df.columns.tolist()}")
        return []
    
    # Extract and clean data
    processed_data = []
    for idx, row in df.iterrows():
        question = clean_text(row['query'])
        context = clean_text(row['context'])
        
        # Skip empty entries
        if not question or not context:
            continue
            
        processed_data.append({
            'question': question,
            'context': context,
            'source': 'large_vi_legal_queries',
            'id': f"legal_query_{idx}"
        })
        
        # Progress indicator
        if (idx + 1) % 50000 == 0:
            print(f"   📊 Processed {idx + 1:,} rows...")
    
    print(f"   ✅ Extracted {len(processed_data):,} valid entries")
    return processed_data

def process_train_valid_file(file_path, source_name):
    """Process train.csv or valid.csv file"""
    print(f"📄 Processing {file_path}...")
    
    # Read CSV with proper encoding
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"   📊 Total rows: {len(df):,}")
    
    # Check required columns
    if 'question' not in df.columns or 'context' not in df.columns:
        print(f"   ❌ Missing required columns. Available: {df.columns.tolist()}")
        return []
    
    # Extract and clean data
    processed_data = []
    for idx, row in df.iterrows():
        question = clean_text(row['question'])
        context = clean_text(row['context'])
        
        # Skip empty entries
        if not question or not context:
            continue
            
        processed_data.append({
            'question': question,
            'context': context,
            'source': source_name,
            'id': f"{source_name}_{idx}"
        })
        
        # Progress indicator
        if (idx + 1) % 20000 == 0:
            print(f"   📊 Processed {idx + 1:,} rows...")
    
    print(f"   ✅ Extracted {len(processed_data):,} valid entries")
    return processed_data

def save_to_jsonl(data, output_file):
    """Save processed data to JSONL format"""
    print(f"💾 Saving to {output_file}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Saved {len(data):,} entries to {output_file}")
    
    # Show file size
    file_size = os.path.getsize(output_file)
    print(f"📏 File size: {file_size / (1024*1024):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Process finetune data for LLaMA training")
    parser.add_argument("--input-dir", default="data/raw/finetune_data", 
                       help="Input directory containing CSV files")
    parser.add_argument("--output-dir", default="data/process_data/finetune_data", 
                       help="Output directory for JSONL files")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("🏛️  VIETNAMESE LEGAL FINETUNE DATA PROCESSING")
    print("=" * 60)
    print(f"📂 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 60)
    
    all_data = []
    
    # Process each file
    files_to_process = [
        ("large_vi_legal_queries.csv", "large_vi_legal_queries", process_large_vi_legal_queries),
        ("train.csv", "train", process_train_valid_file),
        ("valid.csv", "valid", process_train_valid_file)
    ]
    
    for filename, source_name, process_func in files_to_process:
        file_path = input_dir / filename
        
        if file_path.exists():
            if process_func == process_large_vi_legal_queries:
                data = process_func(str(file_path))
            else:
                data = process_func(str(file_path), source_name)
            all_data.extend(data)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    if not all_data:
        print("❌ No data processed. Exiting.")
        return
    
    print(f"\n📊 TOTAL PROCESSED DATA: {len(all_data):,} entries")
    
    # Save combined data
    combined_output = output_dir / "combined_finetune_data.jsonl"
    save_to_jsonl(all_data, str(combined_output))
    
    # Save individual files
    for source_name in ['large_vi_legal_queries', 'train', 'valid']:
        source_data = [item for item in all_data if item['source'] == source_name]
        if source_data:
            source_output = output_dir / f"{source_name}.jsonl"
            save_to_jsonl(source_data, str(source_output))
    
    # Statistics
    print(f"\n📈 DATA STATISTICS:")
    for source in ['large_vi_legal_queries', 'train', 'valid']:
        count = sum(1 for item in all_data if item['source'] == source)
        print(f"   - {source}: {count:,} entries")
    
    print(f"\n🎉 Processing completed successfully!")
    print(f"📁 Output files available in: {output_dir}")

if __name__ == "__main__":
    main()