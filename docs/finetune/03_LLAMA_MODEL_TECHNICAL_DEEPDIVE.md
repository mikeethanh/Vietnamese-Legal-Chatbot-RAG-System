# Llama 3.1 8B Instruct - Technical Deep Dive

## Mục lục
1. [Llama Model Family Overview](#1-llama-model-family-overview)
2. [Base vs Instruct Models](#2-base-vs-instruct-models)
3. [Llama 3.1 Architecture](#3-llama-31-architecture)
4. [Instruction Tuning Process](#4-instruction-tuning-process)
5. [Llama 3.1 8B Instruct Specifications](#5-llama-31-8b-instruct-specifications)
6. [Chat Template Format](#6-chat-template-format)
7. [Tokenizer Details](#7-tokenizer-details)
8. [Use Cases & Performance](#8-use-cases--performance)

---

## 1. Llama Model Family Overview

### Meta's Llama Evolution

```
Llama 1 (Feb 2023)
├── 7B, 13B, 33B, 65B parameters
├── Research only (not commercial)
└── Pre-trained on public data

↓

Llama 2 (Jul 2023)
├── 7B, 13B, 70B parameters
├── Commercial use allowed
├── Base + Chat variants
└── Better training (2T tokens)

↓

Llama 3 (Apr 2024)
├── 8B, 70B parameters
├── Improved architecture
├── Larger vocab (128K → 128K tokens)
└── Trained on 15T tokens!

↓

Llama 3.1 (Jul 2024) ← CỦA CHÚNG TA!
├── 8B, 70B, 405B parameters
├── Extended context (8K → 128K tokens)
├── Tool use capabilities
├── Multilingual improvements
└── Better Vietnamese support!
```

### Model Variants

```
Llama-3.1-8B (Base Model)
├── Pre-trained on raw text
├── No instruction following
├── Good for: fine-tuning, research
└── Example output: continues text

vs

Llama-3.1-8B-Instruct (Instruct Model) ← CHÚNG TA DÙNG!
├── Fine-tuned on instructions
├── Follows instructions naturally
├── Good for: chatbots, Q&A, assistants
└── Example output: answers questions
```

---

## 2. Base vs Instruct Models

### 2.1. Base Model Behavior

**Llama-3.1-8B (Base)**:
```python
# Input (prompt)
"Điều 10 của Bộ luật Lao động quy định về"

# Output (continuation - NO instruction following!)
"thời giờ làm việc, thời giờ nghỉ ngơi. Theo đó, thời giờ làm việc 
bình thường không quá 8 giờ trong một ngày và 48 giờ trong một tuần..."
# ❌ Cứ viết tiếp, không trả lời câu hỏi!
```

**Problem với Base Model**:
```python
# User asks a question
Q: "Thời gian làm việc tối đa theo luật là bao nhiêu?"

# Base model continues text
A: "Thời gian làm việc tối đa theo luật là bao nhiêu giờ trong ngày?
    Thời gian làm việc tối đa theo luật là bao nhiêu giờ trong tuần?
    Các quy định..."
# ❌ Không trả lời, cứ tiếp tục đặt câu hỏi!
```

---

### 2.2. Instruct Model Behavior

**Llama-3.1-8B-Instruct**:
```python
# Input (instruction)
"Điều 10 của Bộ luật Lao động quy định về gì?"

# Output (follows instruction!)
"Điều 10 của Bộ luật Lao động quy định về thời giờ làm việc và thời 
giờ nghỉ ngơi. Cụ thể:
- Thời giờ làm việc bình thường: không quá 8 giờ/ngày và 48 giờ/tuần
- Người sử dụng lao động có thể quy định làm việc theo giờ, ngày, tuần
- Phải bảo đảm thời giờ nghỉ ngơi cho người lao động"
# ✅ Trả lời đúng câu hỏi!
```

**Question Answering**:
```python
Q: "Thời gian làm việc tối đa theo luật là bao nhiêu?"

# Base: Continues writing
A: "Thời gian làm việc tối đa theo luật là bao nhiêu giờ?..." ❌

# Instruct: Answers directly
A: "Theo Bộ luật Lao động, thời gian làm việc tối đa là:
    - 8 giờ trong một ngày
    - 48 giờ trong một tuần" ✅
```

---

### 2.3. Training Process Comparison

```
┌─────────────────────────────────────────────────────┐
│              PRE-TRAINING (Base Model)              │
│  Training Data: Raw text từ internet                │
│  Objective: Predict next word                       │
│  Size: 15 trillion tokens                          │
│  Result: Llama-3.1-8B (Base)                       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         INSTRUCTION TUNING (Instruct Model)         │
│  Training Data: Instruction-response pairs          │
│  Objective: Follow instructions                     │
│  Size: Millions of examples                        │
│  Techniques: SFT + RLHF + DPO                      │
│  Result: Llama-3.1-8B-Instruct ← WE USE THIS!      │
└─────────────────────────────────────────────────────┘
```

#### **Pre-training Phase**
```python
# Training data format
"The capital of France is Paris. Paris is known for..."
"Python is a programming language. It is widely used..."
"Điều 1. Bộ luật này quy định về quyền và nghĩa vụ..."

# Task: Predict next token
Input:  "The capital of France is"
Target: "Paris"

Input:  "Điều 1. Bộ luật này quy định"
Target: "về"
```

#### **Instruction Tuning Phase**
```python
# Training data format (instruction-following)
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "The capital of France is Paris."
}

{
  "instruction": "Điều 10 của Bộ luật Lao động quy định về gì?",
  "input": "",
  "output": "Điều 10 quy định về thời giờ làm việc và nghỉ ngơi..."
}

# Task: Generate response following instruction
Input:  Instruction + Input
Target: Expected output
```

---

## 3. Llama 3.1 Architecture

### 3.1. Transformer Architecture

```
Input Text: "Điều 10 quy định về gì?"
       ↓
┌──────────────────────────────────────┐
│         Tokenization                 │
│  "Điều" "10" "quy" "định" "về" "gì"│
│  [123,  456, 789,  012,  345, 678]  │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Embedding Layer (128K vocab)    │
│  Each token → 4096-dim vector       │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│   Transformer Block 1                │
│   ├── Multi-Head Attention           │
│   ├── MLP (Feed-Forward)             │
│   └── Layer Normalization            │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│   Transformer Block 2                │
│   ...                                │
└──────────────────────────────────────┘
       ↓
      ...  (32 blocks total)
       ↓
┌──────────────────────────────────────┐
│   Transformer Block 32               │
└──────────────────────────────────────┘
       ↓
┌──────────────────────────────────────┐
│      Output Head (LM Head)           │
│  4096-dim → 128K vocab logits        │
└──────────────────────────────────────┘
       ↓
   Predicted Token
```

### 3.2. Attention Mechanism

**Multi-Head Attention trong Llama 3.1**:
```python
# Specifications
num_heads = 32              # 32 attention heads
head_dim = 128              # 128 dimensions per head
hidden_size = 4096          # 32 × 128 = 4096

# Grouped Query Attention (GQA)
num_key_value_heads = 8     # Share K,V across heads
# Faster inference, less memory!
```

**Standard vs Grouped Query Attention**:
```
Standard Multi-Head Attention (MHA):
Head 1: Q1, K1, V1
Head 2: Q2, K2, V2
...
Head 32: Q32, K32, V32
Memory: 32 × (K + V) = lots!

Grouped Query Attention (GQA):
Group 1 (Heads 1-4):  Q1, Q2, Q3, Q4  → shared K1, V1
Group 2 (Heads 5-8):  Q5, Q6, Q7, Q8  → shared K2, V2
...
Group 8 (Heads 29-32): Q29-Q32        → shared K8, V8
Memory: 8 × (K + V) = 4x less!
```

**Tại sao GQA tốt?**
- ✅ Giảm memory (KV cache) 4x
- ✅ Faster inference
- ✅ Quality gần như không giảm
- ✅ Có thể support longer context

---

### 3.3. MLP (Feed-Forward Network)

```python
# Architecture
hidden_size = 4096
intermediate_size = 14336  # ~3.5× hidden_size

# SwiGLU activation
class MLP:
    gate_proj: Linear(4096 → 14336)  # Gate
    up_proj:   Linear(4096 → 14336)  # Up
    down_proj: Linear(14336 → 4096)  # Down
    
    def forward(x):
        # SwiGLU: Swish(gate) × up
        return down_proj(
            swish(gate_proj(x)) × up_proj(x)
        )
```

**SwiGLU vs GELU**:
```python
# GELU (old transformers)
output = GELU(W1 × x) × W2

# SwiGLU (Llama 3.1) - better performance!
output = Swish(W_gate × x) × (W_up × x) × W_down
```

---

### 3.4. RoPE (Rotary Position Embedding)

**Vấn đề**: Transformer không có khái niệm về vị trí tokens

**Giải pháp**: RoPE - encode position info vào embeddings

```python
# RoPE mechanism
def apply_rope(q, k, position):
    # Rotate query and key based on position
    θ = position / 10000^(2i/d)  # Different freq for each dim
    
    # Apply rotation matrix
    q_rotated = rotate(q, θ)
    k_rotated = rotate(k, θ)
    
    return q_rotated, k_rotated

# Properties
- Relative position encoding
- Extrapolate to longer sequences
- Better than absolute position
```

**Llama 3.1 RoPE base frequency**:
```python
rope_theta = 500000  # Increased from 10000 in Llama 2
# Allows better extrapolation to 128K context!
```

---

## 4. Instruction Tuning Process

### 4.1. Supervised Fine-Tuning (SFT)

**Stage 1: SFT trên instruction data**
```python
# Training examples
{
  "instruction": "Summarize this text",
  "input": "Long legal document...",
  "output": "Summary: The document states..."
}

{
  "instruction": "Answer the question",
  "input": "What is the maximum working hours?",
  "output": "According to labor law, maximum is 8 hours/day..."
}

# Loss function
loss = CrossEntropy(model_output, target_output)

# Result: Model learns to follow instructions
```

### 4.2. RLHF (Reinforcement Learning from Human Feedback)

**Stage 2: Learn from human preferences**
```python
# Step 1: Collect comparisons
Question: "Explain Vietnamese labor law"
Response A: "Labor law in Vietnam regulates..." (detailed)
Response B: "It's about work stuff" (vague)
Human preference: A > B ✅

# Step 2: Train reward model
reward_model(Response A) = 0.9  # High score
reward_model(Response B) = 0.3  # Low score

# Step 3: Optimize policy (PPO algorithm)
# Generate response → get reward → update model
# Maximize: E[reward(response)]

# Result: Model generates human-preferred responses
```

### 4.3. DPO (Direct Preference Optimization)

**Stage 3: Simpler alternative to RLHF**
```python
# Direct optimization without reward model
# Given: preferred response y_w, rejected y_l

loss = -log(σ(
    log(π(y_w|x) / π_ref(y_w|x)) - 
    log(π(y_l|x) / π_ref(y_l|x))
))

# Directly increase prob of y_w
# Directly decrease prob of y_l

# Result: Simpler, more stable than RLHF
```

---

## 5. Llama 3.1 8B Instruct Specifications

### 5.1. Model Architecture

```python
{
  "model_type": "llama",
  "architecture": "LlamaForCausalLM",
  
  # Size
  "num_parameters": "8.03B",  # 8 billion parameters
  "num_layers": 32,            # 32 transformer blocks
  
  # Dimensions
  "hidden_size": 4096,         # Hidden dimension
  "intermediate_size": 14336,  # FFN intermediate size
  "num_attention_heads": 32,   # Number of Q heads
  "num_key_value_heads": 8,    # Number of KV heads (GQA)
  "head_dim": 128,             # Dimension per head
  
  # Vocabulary
  "vocab_size": 128256,        # Tokenizer vocabulary
  
  # Context
  "max_position_embeddings": 131072,  # 128K tokens!
  
  # Position encoding
  "rope_theta": 500000,        # RoPE base frequency
  
  # Activation
  "hidden_act": "silu",        # SwiGLU activation
  
  # Normalization
  "rms_norm_eps": 1e-5,        # RMSNorm epsilon
  
  # Precision
  "torch_dtype": "bfloat16"    # BF16 by default
}
```

### 5.2. Model Size Breakdown

```python
# Parameter count per component
Embeddings:        128256 × 4096 = 525M params
Transformer blocks: 32 × 220M = 7.04B params
  ├── Attention:    32 × 100M = 3.2B params
  │   ├── Q proj:   4096 × 4096 × 32 = 512M
  │   ├── K proj:   4096 × 1024 × 32 = 128M (GQA!)
  │   ├── V proj:   4096 × 1024 × 32 = 128M (GQA!)
  │   └── O proj:   4096 × 4096 × 32 = 512M
  └── MLP:          32 × 120M = 3.84B params
      ├── Gate:     4096 × 14336 × 32 = 1.8B
      ├── Up:       4096 × 14336 × 32 = 1.8B
      └── Down:     14336 × 4096 × 32 = 1.8B
LM Head:           128256 × 4096 = 525M params (tied with embeddings)

Total: ~8.03 billion parameters
```

### 5.3. Memory Requirements

```python
# Model weights only
FP32:  8.03B × 4 bytes = 32.12 GB
FP16:  8.03B × 2 bytes = 16.06 GB
BF16:  8.03B × 2 bytes = 16.06 GB
INT8:  8.03B × 1 byte  = 8.03 GB
INT4:  8.03B × 0.5 byte = 4.01 GB

# Inference (with KV cache, batch=1, seq=8192)
FP16: ~20 GB
BF16: ~20 GB

# Training (LoRA r=128, batch=32, seq=8192)
Activations: ~30 GB
Gradients: ~20 GB
Optimizer: ~10 GB
Total: ~80 GB (fits H200!)

# Training (full fine-tuning)
Would need: >200 GB (không khả thi!)
```

---

## 6. Chat Template Format

### 6.1. Llama 3.1 Special Tokens

```python
# Special tokens
<|begin_of_text|>     # Start of conversation
<|end_of_text|>       # End of conversation
<|start_header_id|>   # Start of message header
<|end_header_id|>     # End of message header
<|eot_id|>            # End of turn (message)

# Token IDs
{
  "<|begin_of_text|>": 128000,
  "<|end_of_text|>": 128001,
  "<|start_header_id|>": 128006,
  "<|end_header_id|>": 128007,
  "<|eot_id|>": 128009
}
```

### 6.2. Message Format

**Single Turn**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The capital of France is Paris.<|eot_id|>
```

**Multi-turn Conversation**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is 2+2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

2+2 equals 4.<|eot_id|><|start_header_id|>user<|end_header_id|>

What about 3+3?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

3+3 equals 6.<|eot_id|>
```

### 6.3. Vietnamese Legal Format (Our Use Case)

```python
# Training format
template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions about Vietnamese law.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction} {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

# Example
{
  "instruction": "Điều 10 của Bộ luật Lao động quy định về gì?",
  "input": "",
  "output": "Điều 10 quy định về thời giờ làm việc và thời giờ nghỉ ngơi..."
}

# Formatted:
"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that answers questions about Vietnamese law.<|eot_id|><|start_header_id|>user<|end_header_id|>

Điều 10 của Bộ luật Lao động quy định về gì? <|eot_id|><|start_header_id|>assistant<|end_header_id|>

Điều 10 quy định về thời giờ làm việc và thời giờ nghỉ ngơi...<|eot_id|>"""
```

---

## 7. Tokenizer Details

### 7.1. Tokenizer Type

```python
{
  "tokenizer_type": "tiktoken",  # Fast BPE tokenizer
  "vocab_size": 128256,          # Large vocabulary
  "model_max_length": 131072,    # 128K tokens
  "pad_token": "<|finetune_right_pad_id|>",
  "eos_token": "<|eot_id|>",
  "bos_token": "<|begin_of_text|>"
}
```

### 7.2. Tokenization Examples

**English**:
```python
text = "The capital of France is Paris."
tokens = tokenizer.encode(text)
# ["The", " capital", " of", " France", " is", " Paris", "."]
# [791, 6864, 315, 9822, 374, 12366, 13]
# 7 tokens

# Compression ratio: 7 tokens / 6 words ≈ 1.17
```

**Vietnamese**:
```python
text = "Điều 10 của Bộ luật Lao động quy định về thời giờ làm việc."
tokens = tokenizer.encode(text)
# ["Đ", "i", "ều", " ", "10", " ", "c", "ủa", ...]
# Multiple tokens per word!
# ~2-3 tokens per Vietnamese word

# Example: 14 words → ~30 tokens
# Compression ratio: 30/14 ≈ 2.14
```

**Implications for max_seq_length**:
```python
# English: 8192 tokens ≈ 7000 words
# Vietnamese: 8192 tokens ≈ 3500-4000 words

# For same content coverage:
# Vietnamese needs higher max_seq_length!
max_seq_length = 8192  # Good for Vietnamese legal docs
```

---

## 8. Use Cases & Performance

### 8.1. Model Comparisons

| Model | Params | Context | Speed | Quality | VRAM | Use Case |
|-------|--------|---------|-------|---------|------|----------|
| Llama-3.1-8B | 8B | 128K | Fast | Good | 16GB | Edge, mobile |
| Llama-3.1-70B | 70B | 128K | Medium | Excellent | 140GB | Servers |
| Llama-3.1-405B | 405B | 128K | Slow | SOTA | 800GB+ | Cloud |

### 8.2. Benchmarks

**General Tasks**:
```
MMLU (Knowledge): 68.4%
HumanEval (Code): 62.2%
GSM8K (Math): 79.6%
```

**Vietnamese-Specific** (estimated):
```
Vietnamese Q&A: ~65-70%
Translation: ~70-75%
Summarization: ~70-75%
Legal Understanding: ~60-65% (before fine-tuning)
Legal Understanding: ~80-85% (after fine-tuning) ← OUR GOAL!
```

### 8.3. Why Llama-3.1-8B-Instruct for Vietnamese Legal?

**Advantages**:
1. ✅ **Size**: 8B params → fits consumer GPUs
2. ✅ **Instruct-tuned**: Follows instructions naturally
3. ✅ **Long context**: 128K tokens → long legal docs
4. ✅ **Multilingual**: Good Vietnamese support
5. ✅ **Open source**: Commercial use allowed
6. ✅ **Fine-tunable**: LoRA works excellently
7. ✅ **Unsloth optimized**: 2-5x faster training

**Trade-offs**:
1. ❌ Not as good as 70B/405B (but fine-tunable!)
2. ❌ Vietnamese tokenization less efficient than English
3. ❌ May hallucinate without retrieval (→ RAG needed!)

---

## 9. Comparison: Base vs Instruct Training

### Example: Vietnamese Legal Q&A

**Base Model (Llama-3.1-8B)**:
```python
Input: "Điều 10 của Bộ luật Lao động quy định về"
Output: "thời giờ làm việc. Điều 11 quy định về thời giờ nghỉ ngơi. 
         Điều 12 quy định về làm thêm giờ..."
# Just continues the document! Not helpful for Q&A.
```

**Instruct Model (Llama-3.1-8B-Instruct) - Before Fine-tuning**:
```python
Input: "Điều 10 của Bộ luật Lao động quy định về gì?"
Output: "Điều 10 quy định về thời giờ làm việc bình thường không quá 
         8 giờ một ngày và 48 giờ một tuần."
# Good! But may not cite exact legal text.
```

**Instruct Model - After Vietnamese Legal Fine-tuning**:
```python
Input: "Điều 10 của Bộ luật Lao động quy định về gì?"
Output: "Theo Điều 10 Bộ luật Lao động 2019:
         1. Thời giờ làm việc bình thường không quá 8 giờ trong 1 ngày 
            và 48 giờ trong 1 tuần.
         2. Người sử dụng lao động có quyền quy định thời giờ làm việc 
            theo ngày hoặc tuần nhưng phải thông báo cho người lao động.
         3. Trường hợp đặc biệt, có thể áp dụng khung giờ làm việc khác 
            theo quy định tại Điều 105 của Bộ luật này."
# Excellent! Accurate, detailed, with legal references!
```

---

## 10. Technical Innovations in Llama 3.1

### 10.1. Grouped Query Attention (GQA)

**Problem with Multi-Head Attention**:
```python
# 32 attention heads, 128K context
KV_cache = 32 heads × 128K tokens × 128 dims × 2 (K,V) × 2 bytes
         = 32 × 128000 × 128 × 2 × 2
         = 2.1 GB per sample!
# With batch_size=32: 67 GB just for KV cache!
```

**Solution: Share K,V across head groups**:
```python
# 8 KV heads (4 Q heads share 1 KV head)
KV_cache = 8 heads × 128K tokens × 128 dims × 2 (K,V) × 2 bytes
         = 8 × 128000 × 128 × 2 × 2
         = 524 MB per sample
# 4x reduction! 67 GB → 17 GB
```

### 10.2. Extended Context (128K)

**Llama 2**: 4K context
**Llama 3**: 8K context
**Llama 3.1**: 128K context (16x increase!)

**How?**
1. **RoPE scaling**: Increase rope_theta to 500K
2. **Continued pre-training**: Train on longer sequences
3. **Position interpolation**: Better extrapolation

**Use cases**:
- Full legal documents (10-50 pages)
- Entire conversations (100+ turns)
- Long-form reasoning chains

---

## 11. Unsloth Optimizations

### What is Unsloth?

```python
# Normal Llama loading
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
# Slow, memory-hungry

# Unsloth-optimized Llama
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.1-8B-Instruct"
)
# 2-5x faster, 50% less VRAM!
```

### Optimizations Applied:

1. **Flash Attention 2**:
   ```python
   # Standard attention: O(n²) memory
   # Flash Attention 2: O(n) memory, 2-4x faster
   attn_implementation = "flash_attention_2"
   ```

2. **Kernel fusion**:
   ```python
   # Fuse operations to reduce memory transfers
   # LayerNorm + Linear → Single kernel
   # RoPE + Attention → Single kernel
   ```

3. **Optimized LoRA**:
   ```python
   # Custom CUDA kernels for LoRA
   # 30% faster backward pass
   ```

4. **Memory optimization**:
   ```python
   # Gradient checkpointing with optimal strategy
   # 50% less VRAM for same batch size
   ```

**Results**:
```
Standard training: 10 hours
Unsloth training: 4 hours (2.5x faster!)

Standard VRAM: 80 GB
Unsloth VRAM: 40 GB (2x less!)
```

---

## Tổng kết

### Llama-3.1-8B-Instruct là lựa chọn tốt vì:

1. ✅ **Instruct-tuned**: Sẵn sàng follow instructions
2. ✅ **Appropriate size**: 8B params → fine-tunable on consumer GPUs
3. ✅ **Long context**: 128K tokens → full legal documents
4. ✅ **Multilingual**: Good Vietnamese tokenization
5. ✅ **Modern architecture**: GQA, RoPE, SwiGLU
6. ✅ **Open source**: Commercial use allowed
7. ✅ **Well-supported**: HuggingFace, Unsloth, TRL

### So với Base Model:
- ❌ Base: Chỉ biết tiếp tục text, không answer questions
- ✅ Instruct: Hiểu và follow instructions naturally

### So với model khác:
- Llama-3.1-70B: Better quality, but needs 8x VRAM
- GPT-4: Better, but closed-source, expensive
- Gemini: Good, but API-only
- Vietnamese models: Smaller, less capable

### Fine-tuning cho Vietnamese Legal:
```
Llama-3.1-8B-Instruct (General Vietnamese: ~65%)
            ↓ Fine-tuning với legal data
Vietnamese-Legal-Llama (Legal Vietnamese: ~85%)
            ↓ Integration với RAG
Production System (Accuracy: ~90-95%)
```

---

**Congratulations!** 🎉 Bạn đã hiểu sâu về Llama 3.1 8B Instruct model!

**Next Steps**:
1. Đọc lại 3 files để consolidate knowledge
2. Experiment với training script
3. Monitor training metrics
4. Evaluate fine-tuned model on Vietnamese legal tasks

**Happy Fine-tuning! 🚀**
