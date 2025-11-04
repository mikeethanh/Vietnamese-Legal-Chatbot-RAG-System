"""
Script để phân tích dữ liệu finetune và cải thiện format cho Llama-3.1-8B
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = []
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load JSONL data"""
        logger.info(f"Loading data from {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_point = json.loads(line.strip())
                    self.data.append(data_point)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    
        logger.info(f"Loaded {len(self.data)} examples")
        return self.data
    
    def analyze_structure(self) -> Dict[str, Any]:
        """Analyze data structure and statistics"""
        if not self.data:
            self.load_data()
            
        analysis = {
            "total_examples": len(self.data),
            "fields": set(),
            "field_stats": {},
            "length_stats": {}
        }
        
        # Analyze fields
        for example in self.data:
            analysis["fields"].update(example.keys())
            
        analysis["fields"] = list(analysis["fields"])
        
        # Analyze field statistics
        for field in analysis["fields"]:
            values = [example.get(field, "") for example in self.data]
            non_empty = [v for v in values if v]
            
            analysis["field_stats"][field] = {
                "present_count": len(non_empty),
                "missing_count": len(values) - len(non_empty),
                "avg_length": sum(len(str(v)) for v in non_empty) / len(non_empty) if non_empty else 0,
                "max_length": max(len(str(v)) for v in non_empty) if non_empty else 0,
                "min_length": min(len(str(v)) for v in non_empty) if non_empty else 0
            }
            
        # Analyze text lengths for input/output
        if "input" in analysis["fields"]:
            input_lengths = [len(example.get("input", "")) for example in self.data]
            analysis["length_stats"]["input"] = {
                "avg": sum(input_lengths) / len(input_lengths),
                "max": max(input_lengths),
                "min": min(input_lengths),
                "median": sorted(input_lengths)[len(input_lengths)//2]
            }
            
        if "output" in analysis["fields"]:
            output_lengths = [len(example.get("output", "")) for example in self.data]
            analysis["length_stats"]["output"] = {
                "avg": sum(output_lengths) / len(output_lengths),
                "max": max(output_lengths),
                "min": min(output_lengths),
                "median": sorted(output_lengths)[len(output_lengths)//2]
            }
            
        return analysis
    
    def check_llama_format_compatibility(self) -> Dict[str, Any]:
        """Check if data format is compatible with Llama chat format"""
        compatibility = {
            "has_instruction": False,
            "has_input": False,
            "has_output": False,
            "format_issues": [],
            "recommendations": []
        }
        
        if not self.data:
            self.load_data()
            
        # Check required fields
        sample = self.data[0] if self.data else {}
        
        if "instruction" in sample:
            compatibility["has_instruction"] = True
        else:
            compatibility["format_issues"].append("Missing 'instruction' field")
            
        if "input" in sample:
            compatibility["has_input"] = True
        else:
            compatibility["format_issues"].append("Missing 'input' field")
            
        if "output" in sample:
            compatibility["has_output"] = True
        else:
            compatibility["format_issues"].append("Missing 'output' field")
            
        # Check for EOS token
        has_eos = any("<|im_end|>" in example.get("output", "") for example in self.data[:100])
        if not has_eos:
            compatibility["format_issues"].append("Missing EOS token in outputs")
            compatibility["recommendations"].append("Add EOS token (<|im_end|>) to outputs")
            
        # Check conversation format
        sample_texts = [example.get("output", "") for example in self.data[:10]]
        if not any("system" in text.lower() or "user" in text.lower() for text in sample_texts):
            compatibility["recommendations"].append("Consider converting to Llama chat format")
            
        return compatibility
    
    def sample_examples(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample examples for inspection"""
        if not self.data:
            self.load_data()
            
        return self.data[:n]
    
    def print_analysis_report(self):
        """Print comprehensive analysis report"""
        analysis = self.analyze_structure()
        compatibility = self.check_llama_format_compatibility()
        samples = self.sample_examples()
        
        print("=" * 80)
        print("PHÂN TÍCH DỮ LIỆU FINETUNE")
        print("=" * 80)
        
        print(f"\n1. TỔNG QUAN:")
        print(f"   - Tổng số examples: {analysis['total_examples']:,}")
        print(f"   - Các fields: {', '.join(analysis['fields'])}")
        
        print(f"\n2. THỐNG KÊ FIELDS:")
        for field, stats in analysis['field_stats'].items():
            print(f"   {field}:")
            print(f"     - Có dữ liệu: {stats['present_count']:,} examples")
            print(f"     - Thiếu dữ liệu: {stats['missing_count']:,} examples")
            print(f"     - Độ dài trung bình: {stats['avg_length']:.1f} ký tự")
            print(f"     - Độ dài max: {stats['max_length']:,} ký tự")
            
        print(f"\n3. THỐNG KÊ ĐỘ DÀI TEXT:")
        for field, stats in analysis['length_stats'].items():
            print(f"   {field}:")
            print(f"     - Trung bình: {stats['avg']:.1f} ký tự")
            print(f"     - Median: {stats['median']:.1f} ký tự")
            print(f"     - Max: {stats['max']:,} ký tự")
            print(f"     - Min: {stats['min']:,} ký tự")
            
        print(f"\n4. TƯƠNG THÍCH VỚI LLAMA:")
        print(f"   - Có instruction: {compatibility['has_instruction']}")
        print(f"   - Có input: {compatibility['has_input']}")
        print(f"   - Có output: {compatibility['has_output']}")
        
        if compatibility['format_issues']:
            print(f"   - Vấn đề format:")
            for issue in compatibility['format_issues']:
                print(f"     * {issue}")
                
        if compatibility['recommendations']:
            print(f"   - Khuyến nghị:")
            for rec in compatibility['recommendations']:
                print(f"     * {rec}")
                
        print(f"\n5. MẪU DỮ LIỆU:")
        for i, sample in enumerate(samples):
            print(f"\n   Example {i+1}:")
            for key, value in sample.items():
                preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                print(f"     {key}: {preview}")
                
        print("=" * 80)

def main():
    # Path to your data
    data_path = "/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/data_pipeline/data/finetune_llm/finetune_llm_data.jsonl"
    
    analyzer = DataAnalyzer(data_path)
    analyzer.print_analysis_report()
    
    # Save analysis to file
    analysis = analyzer.analyze_structure()
    compatibility = analyzer.check_llama_format_compatibility()
    
    output_dir = Path("/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/data_processing")
    
    with open(output_dir / "data_analysis.json", 'w', encoding='utf-8') as f:
        json.dump({
            "structure_analysis": analysis,
            "llama_compatibility": compatibility
        }, f, ensure_ascii=False, indent=2)
        
    print(f"\nPhân tích đã được lưu vào: {output_dir / 'data_analysis.json'}")

if __name__ == "__main__":
    main()