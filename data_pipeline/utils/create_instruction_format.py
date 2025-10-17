#!/usr/bin/env python3
"""
Script to generate answers for fine-tuning data using a pre-trained model
This helps create question-context-answer triples for better fine-tuning
"""
import json
import argparse
from pathlib import Path

def create_instruction_format(input_file, output_file, format_type="alpaca"):
    """
    Convert question-context pairs to instruction-following format
    
    Args:
        input_file: Input JSONL file with question-context pairs
        output_file: Output JSONL file with instruction format
        format_type: "alpaca" or "llama2" format
    """
    print(f"🔄 Converting {input_file} to {format_type} instruction format...")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line.strip())
            
            if format_type == "alpaca":
                # Alpaca format
                instruction_data = {
                    "instruction": "Dựa vào văn bản pháp luật được cung cấp, hãy trả lời câu hỏi một cách chính xác và chi tiết.",
                    "input": f"Câu hỏi: {data['question']}\n\nVăn bản tham khảo: {data['context']}",
                    "output": "[Câu trả lời sẽ được tạo dựa trên context]",
                    "source": data.get('source', ''),
                    "id": data.get('id', '')
                }
            
            elif format_type == "llama2":
                # LLaMA-2 chat format
                instruction_data = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Bạn là một trợ lý AI chuyên về pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên văn bản pháp luật được cung cấp."
                        },
                        {
                            "role": "user", 
                            "content": f"Dựa vào văn bản sau:\n\n{data['context']}\n\nHãy trả lời câu hỏi: {data['question']}"
                        },
                        {
                            "role": "assistant",
                            "content": "[Câu trả lời sẽ được tạo dựa trên context]"
                        }
                    ],
                    "source": data.get('source', ''),
                    "id": data.get('id', '')
                }
            
            json.dump(instruction_data, outfile, ensure_ascii=False)
            outfile.write('\n')
            count += 1
            
            if count % 10000 == 0:
                print(f"   📊 Processed {count:,} entries...")
    
    print(f"✅ Converted {count:,} entries to {format_type} format")
    print(f"💾 Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert QA data to instruction format")
    parser.add_argument("input_file", help="Input JSONL file with question-context pairs")
    parser.add_argument("output_file", help="Output JSONL file with instruction format")
    parser.add_argument("--format", choices=["alpaca", "llama2"], default="alpaca",
                       help="Instruction format type")
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"❌ Input file not found: {args.input_file}")
        return
    
    create_instruction_format(args.input_file, args.output_file, args.format)

if __name__ == "__main__":
    main()