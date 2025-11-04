"""
Comprehensive evaluation system for Vietnamese Legal LLM
Includes both automatic metrics and LLM-based evaluation
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
import numpy as np
from datetime import datetime

# ML libraries (will be available on GPU droplet)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from unsloth import FastLanguageModel
    import evaluate
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError as e:
    print(f"Import warning: {e}. Will be available on GPU environment.")

# API libraries
import requests
from openai import OpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Model paths
    model_path: str = ""
    base_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Data paths
    test_data_path: str = ""
    
    # Evaluation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Metrics to compute
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_perplexity: bool = True
    compute_llm_eval: bool = True
    
    # LLM-based evaluation
    openai_api_key: Optional[str] = None
    hf_api_key: Optional[str] = None
    llm_eval_model: str = "gpt-4o-mini"  # or "meta-llama/Llama-3.1-70B-Instruct"
    
    # Output
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True

class VietnameseLegalEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.base_model = None
        self.base_tokenizer = None
        
        # Initialize evaluators
        self.rouge_scorer = None
        self.smoothing_function = SmoothingFunction().method1
        
        # Initialize LLM evaluator
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = OpenAI(api_key=config.openai_api_key)
        
        # Test data
        self.test_data = []
        
        # Results
        self.predictions = []
        self.metrics = {}
        
    def load_model(self):
        """Load fine-tuned model"""
        logger.info(f"Loading fine-tuned model from {self.config.model_path}")
        
        try:
            # Try loading with Unsloth first (for LoRA models)
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
        except:
            # Fallback to standard transformers
            logger.info("Loading with standard transformers...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        logger.info("Fine-tuned model loaded successfully")
    
    def load_base_model(self):
        """Load base model for comparison"""
        if not self.config.compute_perplexity:
            return
            
        logger.info(f"Loading base model: {self.config.base_model_path}")
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        logger.info("Base model loaded successfully")
    
    def load_test_data(self):
        """Load test dataset"""
        logger.info(f"Loading test data from {self.config.test_data_path}")
        
        with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.test_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.test_data)} test examples")
    
    def extract_response_from_chat(self, generated_text: str, prompt: str) -> str:
        """Extract assistant response from generated chat format"""
        # Remove the prompt from generated text
        if prompt in generated_text:
            response = generated_text.replace(prompt, "").strip()
        else:
            response = generated_text
        
        # Extract assistant response
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        if assistant_marker in response:
            response = response.split(assistant_marker)[-1]
        
        # Remove end tokens
        end_tokens = ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
        for token in end_tokens:
            response = response.replace(token, "")
        
        return response.strip()
    
    def generate_response(self, text: str) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = self.extract_response_from_chat(generated_text, text)
        
        return response
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        if not self.config.compute_rouge:
            return {}
            
        logger.info("Computing ROUGE scores...")
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
        rouge_scores = {
            'rouge1_f': [],
            'rouge1_p': [],
            'rouge1_r': [],
            'rouge2_f': [],
            'rouge2_p': [],
            'rouge2_r': [],
            'rougeL_f': [],
            'rougeL_p': [],
            'rougeL_r': []
        }
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            
            rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge1_p'].append(scores['rouge1'].precision)
            rouge_scores['rouge1_r'].append(scores['rouge1'].recall)
            
            rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
            rouge_scores['rouge2_p'].append(scores['rouge2'].precision)
            rouge_scores['rouge2_r'].append(scores['rouge2'].recall)
            
            rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
            rouge_scores['rougeL_p'].append(scores['rougeL'].precision)
            rouge_scores['rougeL_r'].append(scores['rougeL'].recall)
        
        # Calculate averages
        avg_scores = {}
        for key, values in rouge_scores.items():
            avg_scores[f"avg_{key}"] = np.mean(values)
        
        return avg_scores
    
    def compute_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores"""
        if not self.config.compute_bleu:
            return {}
            
        logger.info("Computing BLEU scores...")
        
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU expects list of reference token lists
            
            # Compute BLEU
            try:
                bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing_function)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
        
        return {
            "avg_bleu": np.mean(bleu_scores),
            "bleu_scores": bleu_scores
        }
    
    def compute_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """Compute perplexity on reference texts"""
        if not self.config.compute_perplexity or not self.model:
            return {}
            
        logger.info("Computing perplexity...")
        
        perplexities = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        
        return {
            "avg_perplexity": np.mean(perplexities),
            "perplexities": perplexities
        }
    
    def llm_based_evaluation(self, predictions: List[str], questions: List[str], references: List[str]) -> Dict[str, Any]:
        """LLM-based evaluation using GPT-4 or other LLMs"""
        if not self.config.compute_llm_eval or not self.openai_client:
            return {}
            
        logger.info("Running LLM-based evaluation...")
        
        evaluation_prompt = """
Bạn là một chuyên gia đánh giá chất lượng câu trả lời pháp luật. Hãy đánh giá câu trả lời dưới đây theo các tiêu chí sau:

1. **Độ chính xác** (0-10): Thông tin có chính xác và đúng với pháp luật Việt Nam không?
2. **Độ đầy đủ** (0-10): Câu trả lời có đầy đủ thông tin cần thiết không?
3. **Độ rõ ràng** (0-10): Câu trả lời có dễ hiểu và rõ ràng không?
4. **Tính thực tiễn** (0-10): Câu trả lời có hữu ích trong thực tế không?

**Câu hỏi:** {question}

**Câu trả lời cần đánh giá:** {prediction}

**Câu trả lời tham khảo:** {reference}

Hãy trả lời theo format JSON:
{{
    "accuracy": <điểm_độ_chính_xác>,
    "completeness": <điểm_độ_đầy_đủ>,
    "clarity": <điểm_độ_rõ_ràng>,
    "practicality": <điểm_tính_thực_tiễn>,
    "overall": <điểm_tổng_thể>,
    "explanation": "<giải_thích_ngắn_gọn>"
}}
"""
        
        evaluations = []
        
        for i, (pred, question, ref) in enumerate(zip(predictions[:20], questions[:20], references[:20])):  # Limit for cost
            try:
                prompt = evaluation_prompt.format(
                    question=question,
                    prediction=pred,
                    reference=ref
                )
                
                response = self.openai_client.chat.completions.create(
                    model=self.config.llm_eval_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500
                )
                
                eval_text = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    eval_json = json.loads(eval_text)
                    evaluations.append(eval_json)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM evaluation for example {i}")
                    continue
                    
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"LLM evaluation failed for example {i}: {e}")
                continue
        
        if not evaluations:
            return {}
        
        # Calculate average scores
        avg_scores = {}
        for metric in ['accuracy', 'completeness', 'clarity', 'practicality', 'overall']:
            scores = [e[metric] for e in evaluations if metric in e]
            if scores:
                avg_scores[f"avg_llm_{metric}"] = np.mean(scores)
        
        return {
            **avg_scores,
            "llm_evaluations": evaluations,
            "num_evaluated": len(evaluations)
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        # Load everything
        self.load_test_data()
        self.load_model()
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = []
        questions = []
        references = []
        
        for i, example in enumerate(self.test_data):
            # Extract question and reference from chat format
            text = example.get("text", "")
            
            # Extract user question
            user_start = "<|start_header_id|>user<|end_header_id|>"
            assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
            
            if user_start in text and assistant_start in text:
                user_part = text.split(user_start)[1].split(assistant_start)[0].strip()
                user_part = user_part.replace("<|eot_id|>", "").strip()
                questions.append(user_part)
                
                # Extract reference answer
                ref_part = text.split(assistant_start)[1].strip()
                ref_part = ref_part.replace("<|eot_id|>", "").strip()
                references.append(ref_part)
                
                # Generate prediction
                prompt_for_generation = text.split(assistant_start)[0] + assistant_start + "\n\n"
                prediction = self.generate_response(prompt_for_generation)
                predictions.append(prediction)
                
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1} predictions")
        
        self.predictions = predictions
        
        # Save predictions
        if self.config.save_predictions:
            predictions_data = [
                {
                    "question": q,
                    "prediction": p,
                    "reference": r
                }
                for q, p, r in zip(questions, predictions, references)
            ]
            
            with open(self.output_dir / "predictions.json", 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, ensure_ascii=False, indent=2)
        
        # Compute metrics
        logger.info("Computing automatic metrics...")
        
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, references)
        metrics.update(rouge_scores)
        
        # BLEU scores
        bleu_scores = self.compute_bleu_scores(predictions, references)
        metrics.update(bleu_scores)
        
        # Perplexity
        perplexity_scores = self.compute_perplexity(references)
        metrics.update(perplexity_scores)
        
        # LLM-based evaluation
        llm_scores = self.llm_based_evaluation(predictions, questions, references)
        metrics.update(llm_scores)
        
        self.metrics = metrics
        
        # Save metrics
        with open(self.output_dir / "metrics.json", 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (list, dict)):
                    serializable_metrics[k] = v
                else:
                    serializable_metrics[k] = float(v) if hasattr(v, 'item') else v
            
            json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {self.output_dir}")
        
        return metrics
    
    def print_evaluation_report(self):
        """Print comprehensive evaluation report"""
        if not self.metrics:
            return
        
        print("=" * 80)
        print("📊 BÁO CÁO ĐÁNH GIÁ MODEL")
        print("=" * 80)
        
        print(f"\n🎯 THÔNG TIN ĐÁNH GIÁ:")
        print(f"   - Model: {self.config.model_path}")
        print(f"   - Test examples: {len(self.test_data)}")
        print(f"   - Predictions generated: {len(self.predictions)}")
        
        print(f"\n📈 ROUGE SCORES:")
        for metric in ['avg_rouge1_f', 'avg_rouge2_f', 'avg_rougeL_f']:
            if metric in self.metrics:
                print(f"   - {metric}: {self.metrics[metric]:.4f}")
        
        print(f"\n🔵 BLEU SCORE:")
        if 'avg_bleu' in self.metrics:
            print(f"   - Average BLEU: {self.metrics['avg_bleu']:.4f}")
        
        print(f"\n📊 PERPLEXITY:")
        if 'avg_perplexity' in self.metrics:
            print(f"   - Average Perplexity: {self.metrics['avg_perplexity']:.2f}")
        
        print(f"\n🤖 LLM-BASED EVALUATION:")
        llm_metrics = ['avg_llm_accuracy', 'avg_llm_completeness', 'avg_llm_clarity', 'avg_llm_practicality', 'avg_llm_overall']
        for metric in llm_metrics:
            if metric in self.metrics:
                print(f"   - {metric.replace('avg_llm_', '').title()}: {self.metrics[metric]:.2f}/10")
        
        if 'num_evaluated' in self.metrics:
            print(f"   - LLM evaluated examples: {self.metrics['num_evaluated']}")
        
        print("=" * 80)

def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration
    config = EvaluationConfig(
        model_path="/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/finetune/outputs/final_model",
        test_data_path="/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/data_processing/splits/test.jsonl",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        hf_api_key=os.getenv("HF_TOKEN"),
        output_dir="/home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving/evaluation/results",
        compute_llm_eval=bool(os.getenv("OPENAI_API_KEY"))
    )
    
    # Run evaluation
    evaluator = VietnameseLegalEvaluator(config)
    metrics = evaluator.run_evaluation()
    evaluator.print_evaluation_report()
    
    print(f"\n✅ EVALUATION COMPLETED!")
    print(f"📁 Results saved to: {config.output_dir}")

if __name__ == "__main__":
    main()