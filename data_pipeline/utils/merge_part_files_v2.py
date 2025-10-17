#!/usr/bin/env python3
"""
Script to merge Spark part files into a single JSONL file
"""
import os
import gzip
import glob
import argparse
import json
import re
from pathlib import Path

def merge_part_files(input_dir, output_file):
    """
    Merge all part files from a Spark output directory into a single JSONL file
    
    Args:
        input_dir: Directory containing part.* files
        output_file: Output JSONL file path
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all part files
    part_files = sorted(glob.glob(os.path.join(input_dir, "part.*")))
    
    if not part_files:
        print(f"❌ No part files found in {input_dir}")
        return
    
    print(f"📂 Found {len(part_files)} part files")
    print(f"📝 Output file: {output_file}")
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, part_file in enumerate(part_files, 1):
            print(f"📄 Processing {os.path.basename(part_file)} ({i}/{len(part_files)})")
            
            file_lines = process_part_file(part_file, outfile)
            total_lines += file_lines
            print(f"   ✅ {file_lines:,} lines processed")
    
    print(f"\n🎉 Successfully merged {len(part_files)} files")
    print(f"📊 Total lines: {total_lines:,}")
    print(f"💾 Output file: {output_file}")
    
    # Show file size
    file_size = os.path.getsize(output_file)
    print(f"📏 File size: {file_size / (1024*1024):.2f} MB")

def process_part_file(part_file, outfile):
    """Process a single part file and extract valid JSON lines"""
    file_lines = 0
    
    # Method 1: Try reading as binary and extracting JSON
    try:
        with open(part_file, 'rb') as f:
            content = f.read()
        
        # Try to decode with utf-8, replacing errors
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = content.decode('utf-8', errors='replace')
        
        # Extract JSON objects using regex
        # Look for complete JSON objects with id and text fields
        json_pattern = r'\{"id":"[^"]+","text":"[^"]*"\}'
        
        # Find all JSON-like patterns
        matches = re.finditer(json_pattern, text_content)
        
        for match in matches:
            json_str = match.group()
            try:
                # Validate it's proper JSON
                json_obj = json.loads(json_str)
                if 'id' in json_obj and 'text' in json_obj:
                    # Write to output file
                    outfile.write(json_str + '\n')
                    file_lines += 1
            except json.JSONDecodeError:
                continue
                
        return file_lines
        
    except Exception as e:
        print(f"   ❌ Error processing {part_file}: {e}")
        return 0

def extract_json_from_binary(content):
    """Extract JSON objects from binary content with various encodings"""
    json_objects = []
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            # Look for JSON patterns
            pattern = r'\{"id":"[^"]*","text":"[^"]*"\}'
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    obj = json.loads(match)
                    if 'id' in obj and 'text' in obj and obj['text'].strip():
                        json_objects.append(match)
                except json.JSONDecodeError:
                    continue
                    
            if json_objects:
                break
                
        except UnicodeDecodeError:
            continue
    
    return json_objects

def main():
    parser = argparse.ArgumentParser(description="Merge Spark part files into JSONL")
    parser.add_argument("input_dir", help="Directory containing part.* files")
    parser.add_argument("output_file", help="Output JSONL file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        return
    
    merge_part_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()