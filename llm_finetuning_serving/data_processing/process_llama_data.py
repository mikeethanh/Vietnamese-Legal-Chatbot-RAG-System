"""
Script để xử lý và cải thiện format dữ liệu cho Llama-3.1-8B
Thêm EOS token, chuyển đổi sang chat format, và tối ưu hóa instruction
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaDataProcessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.processed_data = []
        
        # Llama-3.1 special tokens
        self.system_token = "<|start_header_id|>system<|end_header_id|>"
        self.user_token = "<|start_header_id|>user<|end_header_id|>"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>"
        self.eos_token = "<|eot_id|>"
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load original JSONL data"""
        data = []
        logger.info(f"Loading data from {self.input_path}")
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_point = json.loads(line.strip())
                    data.append(data_point)
                    if line_num % 10000 == 0:
                        logger.info(f"Loaded {line_num} examples")
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    
        logger.info(f"Total loaded: {len(data)} examples")
        return data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove URLs (optional - might want to keep for legal references)
        # text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def improve_instruction(self, instruction: str, input_text: str) -> str:
        """Improve instruction to be more specific and clear"""
        
        # If instruction is generic, make it more specific
        if instruction.lower().strip() in ["trả lời câu hỏi pháp luật sau:", "trả lời câu hỏi sau:"]:
            # Analyze input to determine question type
            input_lower = input_text.lower()
            
            if any(word in input_lower for word in ["thủ tục", "hồ sơ", "giấy tờ"]):
                return "Hãy trả lời chi tiết về thủ tục và hồ sơ pháp lý được hỏi:"
            elif any(word in input_lower for word in ["quyền", "nghĩa vụ", "trách nhiệm"]):
                return "Hãy giải thích rõ về quyền, nghĩa vụ và trách nhiệm pháp lý:"
            elif any(word in input_lower for word in ["điều kiện", "quy định", "luật"]):
                return "Hãy nêu rõ các quy định và điều kiện pháp lý liên quan:"
            elif any(word in input_lower for word in ["xử lý", "xử phạt", "vi phạm"]):
                return "Hãy giải thích về việc xử lý vi phạm và các hình thức xử phạt:"
            else:
                return "Hãy trả lời chi tiết và chính xác câu hỏi pháp luật sau:"
        
        return instruction
    
    def create_llama_format(self, instruction: str, input_text: str, output_text: str) -> str:
        """Convert to Llama-3.1 chat format"""
        
        # Clean texts
        instruction = self.clean_text(instruction)
        input_text = self.clean_text(input_text)
        output_text = self.clean_text(output_text)
        
        # Improve instruction
        instruction = self.improve_instruction(instruction, input_text)
        
        # Create system message (Vietnamese legal expert)
        system_message = (
            "Bạn là một chuyên gia tư vấn pháp luật Việt Nam với nhiều năm kinh nghiệm. "
            "Hãy trả lời các câu hỏi một cách chính xác, chi tiết và dễ hiểu. "
            "Luôn dẫn nguồn từ các văn bản pháp luật cụ thể khi có thể."
        )
        
        # Combine instruction and input for user message
        if input_text:
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction
            
        # Build chat format
        chat_text = f"{self.system_token}\n\n{system_message}{self.eos_token}\n"
        chat_text += f"{self.user_token}\n\n{user_message}{self.eos_token}\n"
        chat_text += f"{self.assistant_token}\n\n{output_text}{self.eos_token}"
        
        return chat_text
    
    def process_data(self) -> List[Dict[str, str]]:
        """Process all data and convert to Llama format"""
        raw_data = self.load_data()
        processed_data = []
        
        logger.info("Processing data to Llama format...")
        
        for i, example in enumerate(raw_data):
            try:
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output_text = example.get("output", "")
                
                if not output_text:
                    logger.warning(f"Skipping example {i} - no output")
                    continue
                
                # Create Llama format
                formatted_text = self.create_llama_format(instruction, input_text, output_text)
                
                processed_example = {
                    "text": formatted_text,
                    "original_instruction": instruction,
                    "original_input": input_text,
                    "original_output": output_text
                }
                
                processed_data.append(processed_example)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1} examples")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                continue
                
        logger.info(f"Successfully processed {len(processed_data)} examples")
        self.processed_data = processed_data
        return processed_data
    
    def save_processed_data(self, include_originals: bool = False):
        """Save processed data to JSONL format"""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {self.output_path}")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for example in self.processed_data:
                if include_originals:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
                else:
                    # Only save the formatted text for training
                    f.write(json.dumps({"text": example["text"]}, ensure_ascii=False) + '\n')
                    
        logger.info(f"Saved {len(self.processed_data)} processed examples")
    
    def save_sample_for_inspection(self, n_samples: int = 5):
        """Save sample examples for manual inspection"""
        sample_path = self.output_path.parent / "sample_processed_data.json"
        
        sample_data = self.processed_data[:n_samples]
        
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {n_samples} sample examples to {sample_path}")
        
        # Also print one example
        if sample_data:
            print("\n" + "="*80)
            print("MẪU DỮ LIỆU SAU KHI XỬ LÝ:")
            print("="*80)
            print(sample_data[0]["text"])
            print("="*80)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics of processed data"""
        if not self.processed_data:
            return {}
            
        text_lengths = [len(example["text"]) for example in self.processed_data]
        
        stats = {
            "total_examples": len(self.processed_data),
            "avg_length": sum(text_lengths) / len(text_lengths),
            "max_length": max(text_lengths),
            "min_length": min(text_lengths),
            "median_length": sorted(text_lengths)[len(text_lengths)//2]
        }
        
        return stats

def main():
    # Paths
    input_path = "/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/data_pipeline/data/finetune_llm/finetune_llm_data.jsonl"
    output_path = "/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/data_processing/processed_llama_data.jsonl"
    
    # Process data
    processor = LlamaDataProcessor(input_path, output_path)
    processed_data = processor.process_data()
    
    # Save processed data
    processor.save_processed_data(include_originals=False)
    processor.save_sample_for_inspection()
    
    # Print statistics
    stats = processor.get_statistics()
    print(f"\n📊 THỐNG KÊ DỮ LIỆU ĐÃ XỬ LÝ:")
    print(f"   - Tổng số examples: {stats['total_examples']:,}")
    print(f"   - Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
    print(f"   - Độ dài median: {stats['median_length']:.1f} ký tự")
    print(f"   - Độ dài max: {stats['max_length']:,} ký tự")
    print(f"   - Độ dài min: {stats['min_length']:,} ký tự")
    
    print(f"\n✅ Hoàn thành xử lý dữ liệu!")
    print(f"   - File đầu ra: {output_path}")
    print(f"   - File mẫu: {output_path.parent / 'sample_processed_data.json'}")

if __name__ == "__main__":
    main()