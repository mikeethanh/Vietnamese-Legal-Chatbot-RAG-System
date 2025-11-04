"""
Script finetune Llama-3.1-8B using Unsloth with LoRA
Optimized for Vietnamese Legal QA
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

# Import DO Spaces manager
import sys
sys.path.append('..')
from do_spaces_manager import DOSpacesManager

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
except ImportError:
    print("❌ Unsloth not installed. Install with: pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
    raise

# Transformers imports
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from datasets import Dataset, load_dataset
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuneConfig:
    """Fine-tuning configuration optimized for Unsloth"""
    # Model settings - Use Unsloth models for better performance
    model_name: str = "unsloth/Llama-3.1-8B-Instruct"  # Unsloth optimized model
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None  # Auto-detect (None for optimization)
    load_in_4bit: bool = True  # Essential for H200 memory efficiency
    
    # LoRA settings - Optimized for Vietnamese legal domain
    lora_r: int = 32  # Higher rank for complex legal reasoning
    lora_alpha: int = 64  # 2 * lora_r for optimal learning
    lora_dropout: float = 0.0  # 0 for faster training (Unsloth optimized)
    bias: str = "none"  # "none" is optimized for Unsloth
    use_gradient_checkpointing: str = "unsloth"  # Use Unsloth's implementation
    random_state: int = 3407  # Unsloth recommended seed
    use_rslora: bool = False  # Can enable for rank stabilization
    
    # Training settings - Optimized for H200 GPU (80GB VRAM)
    per_device_train_batch_size: int = 4  # Increased for H200
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 4 * 8 = 32
    warmup_steps: int = 20  # More warmup for stable training
    num_train_epochs: int = 3
    max_steps: int = -1  # Use epochs instead
    learning_rate: float = 2e-4  # Conservative for legal domain
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    
    # Optimization - Unsloth optimized settings
    optim: str = "adamw_8bit"  # 8-bit optimizer for memory efficiency
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()  # Use bf16 on H200
    
    # Logging and saving
    logging_steps: int = 1
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Data settings
    dataset_text_field: str = "text"
    packing: bool = False  # Keep False for legal domain accuracy
    
    # Output settings
    output_dir: str = "./outputs"
    run_name: str = field(default_factory=lambda: f"vietnamese-legal-llama-{datetime.now().strftime('%Y%m%d-%H%M%S')}")

class LlamaFineTuner:
    def __init__(self, config: FineTuneConfig, data_dir: str):
        self.config = config
        self.data_dir = Path(data_dir)
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            config_dict = {k: v for k, v in config.__dict__.items() 
                          if not k.startswith('_')}
            # Convert non-serializable types
            if config_dict.get('dtype'):
                config_dict['dtype'] = str(config_dict['dtype'])
            json.dump(config_dict, f, indent=2)
    
    def load_model(self):
        """Load and setup model with Unsloth optimizations"""
        logger.info(f"🦥 Loading Unsloth model: {self.config.model_name}")
        
        # Load model with Unsloth optimizations
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,  # None for auto-detection
            load_in_4bit=self.config.load_in_4bit,
            # token="hf_..." # Use if model is gated
        )
        
        # Apply LoRA with Unsloth optimizations
        logger.info("🔧 Applying LoRA configuration with Unsloth")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],  # All linear layers
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,  # 0 for Unsloth optimization
            bias=self.config.bias,  # "none" for optimization
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,  # "unsloth"
            random_state=self.config.random_state,
            use_rslora=self.config.use_rslora,
            max_seq_length=self.config.max_seq_length,
        )
        
        # Setup chat template for Llama-3.1
        logger.info("📝 Setting up Llama-3.1 chat template")
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="llama-3.1",
        )
        
        logger.info("✅ Model loaded successfully with Unsloth optimizations")
        
        # Print memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"📊 GPU Memory: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB used")
        
    def load_datasets(self) -> Dict[str, Dataset]:
        """Load train and validation datasets from Digital Ocean Spaces"""
        logger.info("Loading datasets from Digital Ocean Spaces...")
        
        # Initialize DO Spaces manager
        try:
            spaces_manager = DOSpacesManager()
            
            # Download training data if not exists locally
            local_data_dir = Path("../data_processing/splits")
            local_data_dir.mkdir(parents=True, exist_ok=True)
            
            files_to_download = [
                ("process_data/finetune_data/splits/train.jsonl", str(local_data_dir / "train.jsonl")),
                ("process_data/finetune_data/splits/val.jsonl", str(local_data_dir / "val.jsonl")),
                ("process_data/finetune_data/splits/batch_config.json", str(local_data_dir / "batch_config.json"))
            ]
            
            for spaces_path, local_path in files_to_download:
                if not Path(local_path).exists():
                    logger.info(f"Downloading {spaces_path}...")
                    spaces_manager.download_file(spaces_path, local_path, show_progress=False)
                    
        except Exception as e:
            logger.warning(f"Could not download from DO Spaces: {e}")
            logger.info("Using local data files...")
        
        datasets = {}
        
        # Load train dataset
        train_path = Path("../data_processing/splits/train.jsonl")
        if train_path.exists():
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    train_data.append(json.loads(line))
            datasets['train'] = Dataset.from_list(train_data)
            logger.info(f"Loaded {len(train_data)} training examples")
        else:
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        # Load validation dataset
        val_path = Path("../data_processing/splits/val.jsonl")
        if val_path.exists():
            val_data = []
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    val_data.append(json.loads(line))
            datasets['val'] = Dataset.from_list(val_data)
            logger.info(f"Loaded {len(val_data)} validation examples")
        
        return datasets
    
    def setup_trainer(self, datasets: Dict[str, Dataset]):
        """Setup SFT trainer with Unsloth optimizations"""
        logger.info("🔧 Setting up SFT trainer with Unsloth optimizations")
        
        # Training arguments optimized for H200
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.config.run_name,
            
            # Batch size and gradient settings
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Learning rate and schedule
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_steps=self.config.warmup_steps,
            
            # Training duration
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps,
            
            # Optimization
            optim=self.config.optim,  # adamw_8bit for memory efficiency
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            
            # Logging and saving
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            
            # Reproducibility
            seed=self.config.seed,
            
            # Monitoring
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        )
        
        # Setup SFT trainer with Unsloth
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=datasets['train'],
            eval_dataset=datasets.get('val'),
            dataset_text_field=self.config.dataset_text_field,
            packing=self.config.packing,  # False for legal accuracy
            args=training_args,
            max_seq_length=self.config.max_seq_length,
        )
        
        logger.info("✅ SFT Trainer setup complete")
        logger.info(f"📊 Training dataset size: {len(datasets['train'])}")
        if 'val' in datasets:
            logger.info(f"📊 Validation dataset size: {len(datasets['val'])}")
        logger.info(f"🎯 Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
    
    def train(self):
        """Run training with Unsloth optimizations"""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call setup_trainer first.")
        
        logger.info("🚀 Starting training with Unsloth optimizations...")
        
        # Initialize wandb if API key is available
        if os.getenv("WANDB_API_KEY"):
            wandb.init(
                project="vietnamese-legal-llama-unsloth",
                name=self.config.run_name,
                config=self.config.__dict__
            )
            logger.info("📊 Weights & Biases monitoring initialized")
        
        # Print GPU memory before training
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"🔋 GPU Memory before training: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB")
        
        # Train with Unsloth speed optimizations
        trainer_stats = self.trainer.train()
        
        # Print GPU memory after training
        if torch.cuda.is_available():
            gpu_allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"🔋 GPU Memory after training: {gpu_allocated_after:.1f}GB / {gpu_memory:.1f}GB")
        
        logger.info(f"✅ Training completed! Stats: {trainer_stats}")
        
        # Save model automatically
        self.save_model()
        
        return trainer_stats
    
    def save_model(self, save_method: str = "merged_16bit"):
        """Save the fine-tuned model and upload to Digital Ocean Spaces"""
        logger.info(f"Saving model using method: {save_method}")
        
        save_dir = self.output_dir / "final_model"
        save_dir.mkdir(exist_ok=True)
        
        if save_method == "merged_16bit":
            # Save merged model in 16bit
            self.model.save_pretrained_merged(
                str(save_dir), 
                self.tokenizer, 
                save_method="merged_16bit"
            )
        elif save_method == "lora":
            # Save only LoRA adapters
            self.model.save_pretrained(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
        elif save_method == "gguf":
            # Save in GGUF format for inference
            self.model.save_pretrained_gguf(
                str(save_dir), 
                self.tokenizer,
                quantization_method="q4_k_m"
            )
        
        logger.info(f"Model saved to {save_dir}")
        
        # Upload to Digital Ocean Spaces
        try:
            logger.info("Uploading model to Digital Ocean Spaces...")
            spaces_manager = DOSpacesManager()
            
            # Create unique model name with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"vietnamese-legal-llama-{timestamp}"
            
            # Upload model directory
            if spaces_manager.upload_directory(str(save_dir), f"models/{model_name}"):
                logger.info(f"✅ Model uploaded to DO Spaces: models/{model_name}")
                
                # Save model info
                model_info = {
                    "model_name": model_name,
                    "timestamp": timestamp,
                    "config": self.config.__dict__,
                    "spaces_path": f"models/{model_name}",
                    "save_method": save_method
                }
                
                # Upload model info
                info_path = self.output_dir / "model_info.json"
                with open(info_path, 'w') as f:
                    json.dump(model_info, f, indent=2, default=str)
                
                spaces_manager.upload_file(
                    str(info_path), 
                    f"models/{model_name}/model_info.json"
                )
                
                logger.info(f"Model info saved: {model_name}")
                
            else:
                logger.error("Failed to upload model to DO Spaces")
                
        except Exception as e:
            logger.error(f"Error uploading to DO Spaces: {e}")
            logger.info("Model saved locally only")
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set"""
        if not self.trainer:
            raise ValueError("Trainer not initialized")
        
        logger.info("Running evaluation...")
        eval_results = self.trainer.evaluate()
        
        # Save evaluation results
        with open(self.output_dir / "eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Evaluation results: {eval_results}")
        return eval_results
    
    def print_trainable_parameters(self):
        """Print trainable parameters info"""
        if self.model:
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            print(f"Trainable params: {trainable_params:,} || "
                  f"All params: {all_param:,} || "
                  f"Trainable%: {100 * trainable_params / all_param:.4f}")

def main():
    """Main training function optimized for H200 GPU"""
    # Configuration optimized for H200 GPU (80GB VRAM)
    config = FineTuneConfig(
        # Model settings - Use Unsloth optimized model
        model_name="unsloth/Llama-3.1-8B-Instruct",
        max_seq_length=2048,  # Good balance for legal documents
        load_in_4bit=True,  # Essential for memory efficiency
        
        # LoRA settings - Optimized for Vietnamese legal domain
        lora_r=32,  # Higher rank for complex legal reasoning
        lora_alpha=64,  # 2 * lora_r for optimal performance
        lora_dropout=0.0,  # 0 for Unsloth optimization
        
        # Training settings - Optimized for H200 (80GB VRAM)
        per_device_train_batch_size=4,  # Increased from 2 to 4
        gradient_accumulation_steps=8,  # Effective batch size = 4 * 8 = 32
        warmup_steps=20,  # More warmup for stable training
        num_train_epochs=3,  # Conservative for legal domain
        learning_rate=2e-4,  # Conservative learning rate
        weight_decay=0.01,
        
        # Output
        output_dir="/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/finetune/outputs",
        run_name=f"vietnamese-legal-llama-unsloth-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Data directory
    data_dir = "/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/data_processing/splits"
    
    # Initialize trainer
    logger.info("🚀 Initializing Vietnamese Legal LLM Fine-tuner with Unsloth")
    logger.info(f"📋 Configuration:")
    logger.info(f"   - Model: {config.model_name}")
    logger.info(f"   - LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    logger.info(f"   - Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"   - Learning rate: {config.learning_rate}")
    logger.info(f"   - Epochs: {config.num_train_epochs}")
    
    finetuner = LlamaFineTuner(config, data_dir)
    
    try:
        # Load model with Unsloth optimizations
        finetuner.load_model()
        finetuner.print_trainable_parameters()
        
        # Load datasets from Digital Ocean Spaces
        datasets = finetuner.load_datasets()
        
        # Setup trainer with optimized settings
        finetuner.setup_trainer(datasets)
        
        # Train with Unsloth speed optimizations
        stats = finetuner.train()
        
        # Evaluate final model
        eval_results = finetuner.evaluate()
        
        print("\\n" + "="*80)
        print("🎉 VIETNAMESE LEGAL LLM FINETUNE HOÀN THÀNH!")
        print("="*80)
        print(f"🦥 Trained with Unsloth optimizations")
        print(f"📁 Model saved to: {config.output_dir}/final_model")
        print(f"📊 Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        print(f"⏱️  Training time: {stats.metrics.get('train_runtime', 0):.2f} seconds")
        print(f"🚀 Training speed: {stats.metrics.get('train_samples_per_second', 0):.2f} samples/sec")
        print(f"💾 Model uploaded to Digital Ocean Spaces")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()