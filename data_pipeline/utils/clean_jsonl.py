#!/usr/bin/env python3
"""
Script to clean and fix invalid JSON lines in JSONL file
"""
import json
import sys
import argparse
from pathlib import Path

def clean_jsonl_file(input_file, output_file=None):
    """
    Clean JSONL file by removing invalid JSON lines
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path (if None, overwrites input)
    """
    if output_file is None:
        output_file = input_file + ".cleaned"
    
    valid_lines = 0
    invalid_lines = 0
    
    print(f"🔧 Cleaning JSONL file: {input_file}")
    print(f"📝 Output file: {output_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try to parse JSON
                    obj = json.loads(line)
                    
                    # Validate required fields
                    if 'id' in obj and 'text' in obj and obj['text'].strip():
                        # Write valid line
                        json.dump(obj, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        valid_lines += 1
                    else:
                        print(f"⚠️ Line {i}: Missing required fields or empty text")
                        invalid_lines += 1
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Line {i}: JSON decode error - {e}")
                    print(f"   Preview: {line[:100]}...")
                    invalid_lines += 1
                except Exception as e:
                    print(f"❌ Line {i}: Unexpected error - {e}")
                    invalid_lines += 1
                
                # Progress indicator
                if i % 200000 == 0:
                    print(f"📊 Processed {i:,} lines - Valid: {valid_lines:,}, Invalid: {invalid_lines:,}")
    
        print(f"\n✅ Cleaning complete!")
        print(f"📊 Results:")
        print(f"   - Total lines processed: {i:,}")
        print(f"   - Valid lines: {valid_lines:,}")
        print(f"   - Invalid lines removed: {invalid_lines:,}")
        print(f"💾 Clean file saved as: {output_file}")
        
        # Replace original file if requested
        if output_file.endswith('.cleaned'):
            import shutil
            backup_file = input_file + '.backup'
            print(f"📋 Creating backup: {backup_file}")
            shutil.copy2(input_file, backup_file)
            shutil.move(output_file, input_file)
            print(f"✅ Original file updated, backup saved")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Clean invalid JSON lines from JSONL file")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("-o", "--output", help="Output file path (default: input_file.cleaned)")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        sys.exit(1)
    
    clean_jsonl_file(args.input_file, args.output)

if __name__ == "__main__":
    main()