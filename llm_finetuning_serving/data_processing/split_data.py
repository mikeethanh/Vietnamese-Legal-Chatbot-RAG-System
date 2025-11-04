"""
Script để chia dữ liệu thành train/validation và tạo data loader với batching
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSplit:
    """Data split configuration"""
    train_ratio: float = 0.8
    val_ratio: float = 0.15
    test_ratio: float = 0.05
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

class DataPreprocessor:
    def __init__(self, 
                 input_path: str, 
                 output_dir: str,
                 split_config: Optional[DataSplit] = None,
                 seed: int = 42):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.split_config = split_config or DataSplit()
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        
        self.data = []
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load processed JSONL data"""
        logger.info(f"Loading data from {self.input_path}")
        
        data = []
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
        self.data = data
        return data
    
    def analyze_data_distribution(self) -> Dict[str, Any]:
        """Analyze data distribution before splitting"""
        if not self.data:
            self.load_data()
            
        text_lengths = [len(example.get("text", "")) for example in self.data]
        
        analysis = {
            "total_examples": len(self.data),
            "text_length_stats": {
                "mean": sum(text_lengths) / len(text_lengths),
                "median": sorted(text_lengths)[len(text_lengths)//2],
                "max": max(text_lengths),
                "min": min(text_lengths),
                "std": self._calculate_std(text_lengths)
            },
            "length_distribution": self._get_length_distribution(text_lengths)
        }
        
        return analysis
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _get_length_distribution(self, lengths: List[int]) -> Dict[str, int]:
        """Get distribution of text lengths in buckets"""
        buckets = {
            "0-500": 0,
            "501-1000": 0,
            "1001-2000": 0,
            "2001-3000": 0,
            "3001-5000": 0,
            "5000+": 0
        }
        
        for length in lengths:
            if length <= 500:
                buckets["0-500"] += 1
            elif length <= 1000:
                buckets["501-1000"] += 1
            elif length <= 2000:
                buckets["1001-2000"] += 1
            elif length <= 3000:
                buckets["2001-3000"] += 1
            elif length <= 5000:
                buckets["3001-5000"] += 1
            else:
                buckets["5000+"] += 1
                
        return buckets
    
    def stratified_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data with stratification based on text length"""
        if not self.data:
            self.load_data()
            
        # Sort by text length for stratification
        data_with_length = [(example, len(example.get("text", ""))) for example in self.data]
        data_with_length.sort(key=lambda x: x[1])
        
        # Shuffle within similar length groups to maintain randomness
        # while preserving stratification
        chunk_size = 100
        stratified_data = []
        
        for i in range(0, len(data_with_length), chunk_size):
            chunk = data_with_length[i:i+chunk_size]
            random.shuffle(chunk)
            stratified_data.extend(chunk)
        
        # Extract just the examples
        shuffled_data = [item[0] for item in stratified_data]
        
        # Calculate split indices
        total = len(shuffled_data)
        train_idx = int(total * self.split_config.train_ratio)
        val_idx = int(total * (self.split_config.train_ratio + self.split_config.val_ratio))
        
        # Split data
        self.train_data = shuffled_data[:train_idx]
        self.val_data = shuffled_data[train_idx:val_idx]
        self.test_data = shuffled_data[val_idx:]
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(self.train_data):,} examples ({len(self.train_data)/total*100:.1f}%)")
        logger.info(f"  Val: {len(self.val_data):,} examples ({len(self.val_data)/total*100:.1f}%)")
        logger.info(f"  Test: {len(self.test_data):,} examples ({len(self.test_data)/total*100:.1f}%)")
        
        return self.train_data, self.val_data, self.test_data
    
    def save_splits(self):
        """Save train/val/test splits to separate files"""
        if not all([self.train_data, self.val_data, self.test_data]):
            self.stratified_split()
            
        splits = {
            "train": self.train_data,
            "val": self.val_data,
            "test": self.test_data
        }
        
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.jsonl"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in split_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
                    
            logger.info(f"Saved {len(split_data)} examples to {output_path}")
    
    def create_batching_config(self, max_length: int = 2048) -> Dict[str, Any]:
        """Create configuration for batching with padding"""
        
        # Analyze length distribution for optimal batching
        all_lengths = []
        for split_data in [self.train_data, self.val_data, self.test_data]:
            all_lengths.extend([len(example.get("text", "")) for example in split_data])
        
        # Calculate percentiles for dynamic batching
        sorted_lengths = sorted(all_lengths)
        percentiles = {
            "p50": sorted_lengths[int(len(sorted_lengths) * 0.5)],
            "p75": sorted_lengths[int(len(sorted_lengths) * 0.75)],
            "p90": sorted_lengths[int(len(sorted_lengths) * 0.9)],
            "p95": sorted_lengths[int(len(sorted_lengths) * 0.95)],
            "p99": sorted_lengths[int(len(sorted_lengths) * 0.99)]
        }
        
        # Suggest batch sizes based on length distribution
        batch_config = {
            "max_length": max_length,
            "length_percentiles": percentiles,
            "suggested_batch_sizes": {
                "short_sequences": {"max_len": percentiles["p50"], "batch_size": 8},
                "medium_sequences": {"max_len": percentiles["p75"], "batch_size": 4},
                "long_sequences": {"max_len": percentiles["p90"], "batch_size": 2},
                "very_long_sequences": {"max_len": max_length, "batch_size": 1}
            },
            "padding_strategy": "longest_in_batch",
            "truncation": True,
            "gradient_accumulation_steps": 4
        }
        
        return batch_config
    
    def save_batch_config(self, max_length: int = 2048):
        """Save batching configuration"""
        config = self.create_batching_config(max_length)
        
        config_path = self.output_dir / "batch_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved batch configuration to {config_path}")
        return config
    
    def get_split_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each split"""
        if not all([self.train_data, self.val_data, self.test_data]):
            self.stratified_split()
            
        stats = {}
        
        for split_name, split_data in [("train", self.train_data), 
                                       ("val", self.val_data), 
                                       ("test", self.test_data)]:
            lengths = [len(example.get("text", "")) for example in split_data]
            
            stats[split_name] = {
                "count": len(split_data),
                "avg_length": sum(lengths) / len(lengths) if lengths else 0,
                "median_length": sorted(lengths)[len(lengths)//2] if lengths else 0,
                "max_length": max(lengths) if lengths else 0,
                "min_length": min(lengths) if lengths else 0
            }
            
        return stats
    
    def print_summary(self):
        """Print comprehensive summary"""
        analysis = self.analyze_data_distribution()
        split_stats = self.get_split_statistics()
        batch_config = self.create_batching_config()
        
        print("=" * 80)
        print("TỔNG KẾT DATA PREPROCESSING")
        print("=" * 80)
        
        print(f"\n📊 PHÂN TÍCH DỮ LIỆU TỔNG THỂ:")
        print(f"   - Tổng examples: {analysis['total_examples']:,}")
        print(f"   - Độ dài trung bình: {analysis['text_length_stats']['mean']:.1f} ký tự")
        print(f"   - Độ dài median: {analysis['text_length_stats']['median']:,} ký tự")
        print(f"   - Độ dài max: {analysis['text_length_stats']['max']:,} ký tự")
        
        print(f"\n📈 PHÂN PHỐI ĐỘ DÀI:")
        for bucket, count in analysis['length_distribution'].items():
            percentage = count / analysis['total_examples'] * 100
            print(f"   - {bucket}: {count:,} examples ({percentage:.1f}%)")
        
        print(f"\n🔀 CHIA DỮ LIỆU:")
        for split_name, stats in split_stats.items():
            print(f"   {split_name.upper()}:")
            print(f"     - Số lượng: {stats['count']:,}")
            print(f"     - Độ dài TB: {stats['avg_length']:.1f}")
            print(f"     - Độ dài median: {stats['median_length']:,}")
        
        print(f"\n🔄 CẤU HÌNH BATCHING:")
        print(f"   - Max length: {batch_config['max_length']:,}")
        print(f"   - Suggested batch sizes:")
        for seq_type, config in batch_config['suggested_batch_sizes'].items():
            print(f"     * {seq_type}: batch_size={config['batch_size']}, max_len={config['max_len']}")
        
        print("=" * 80)

def main():
    # Paths
    input_path = "/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/data_processing/processed_llama_data.jsonl"
    output_dir = "/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/data_processing/splits"
    
    # Configure split ratios
    split_config = DataSplit(
        train_ratio=0.8,
        val_ratio=0.15, 
        test_ratio=0.05
    )
    
    # Process data
    preprocessor = DataPreprocessor(
        input_path=input_path,
        output_dir=output_dir,
        split_config=split_config,
        seed=42
    )
    
    # Split and save data
    preprocessor.stratified_split()
    preprocessor.save_splits()
    
    # Save batch configuration
    batch_config = preprocessor.save_batch_config(max_length=2048)
    
    # Print summary
    preprocessor.print_summary()
    
    print(f"\n✅ HOÀN THÀNH DATA PREPROCESSING!")
    print(f"   - Train file: {output_dir}/train.jsonl")
    print(f"   - Val file: {output_dir}/val.jsonl") 
    print(f"   - Test file: {output_dir}/test.jsonl")
    print(f"   - Batch config: {output_dir}/batch_config.json")

if __name__ == "__main__":
    main()