# Parameters & Hyperparameters Deep Dive

## Mục lục
1. [Model Loading Functions](#0-model-loading-functions)
2. [Model Parameters](#1-model-parameters)
3. [LoRA Parameters](#2-lora-parameters)
4. [Training Parameters](#3-training-parameters)
5. [Optimization Parameters](#4-optimization-parameters)
6. [Data Loading Parameters](#5-data-loading-parameters)
7. [Logging & Saving Parameters](#6-logging--saving-parameters)
8. [Model Loading & Quantization](#7-model-loading--quantization)

---

## 0. Model Loading Functions

### 0.1. `FastLanguageModel.from_pretrained()`

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=False,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```

**Ý nghĩa**: Load pretrained model từ HuggingFace Hub với Unsloth optimizations

**Cơ chế hoạt động**:
```python
# Step-by-step process
1. Download model weights từ HuggingFace Hub
   └─> Lưu vào cache: ~/.cache/huggingface/
   
2. Load weights vào memory
   └─> Apply dtype (FP16/BF16/FP32)
   └─> Apply quantization (nếu load_in_4bit=True)
   
3. Setup model architecture
   └─> Configure attention mechanism
   └─> Setup position embeddings (RoPE)
   └─> Initialize layers
   
4. Apply Unsloth optimizations
   └─> Flash Attention 2 (if supported)
   └─> Kernel fusion
   └─> Memory optimization
   
5. Setup tokenizer
   └─> Load vocab
   └─> Configure special tokens
   └─> Setup chat template
   
6. Map to device(s)
   └─> Single GPU: load all to GPU 0
   └─> Multi-GPU: distribute across GPUs
```

**Parameters trong from_pretrained**:

#### **`attn_implementation`**
```python
attn_implementation = "flash_attention_2"
```

**Ý nghĩa**: Chọn implementation của attention mechanism

**Attention Complexity**:
```python
# Standard Attention (vanilla)
# Complexity: O(n²) memory, O(n²) time
# Memory usage: HUGE for long sequences!

Input: [batch, seq_len, hidden]
       [32, 8192, 4096]

# Compute Q, K, V
Q = input @ W_q  # [32, 8192, 4096]
K = input @ W_k  # [32, 8192, 4096]
V = input @ W_v  # [32, 8192, 4096]

# Attention scores: Q @ K^T
scores = Q @ K.T  # [32, 8192, 8192] ← 32 × 67M = 2.1 BILLION values!
# For BF16: 2.1B × 2 bytes = 4.2 GB just for attention scores!

# Attention weights
attn = softmax(scores / √d)  # [32, 8192, 8192]

# Output
output = attn @ V  # [32, 8192, 4096]

# Total memory: ~12 GB per attention layer!
# 32 layers → ~384 GB! IMPOSSIBLE!
```

**Flash Attention 2 Solution**:
```python
# Flash Attention 2 (optimized)
# Complexity: O(n) memory!, O(n²) time (same)
# Memory reduction: 10-20x!

# Key innovations:
1. Tiling: Divide Q, K, V into blocks
2. Recomputation: Don't store intermediate attention scores
3. Fused kernels: Combine operations
4. Online softmax: Compute incrementally

# Memory usage
scores = KHÔNG LƯU! (recompute in backward)
attn = KHÔNG LƯU! (recompute in backward)

# Only store: Q, K, V, output
# Memory: ~1 GB per layer (vs ~12 GB standard)
# 32 layers → ~32 GB (vs ~384 GB)
# Reduction: 12x!
```

#### **`device_map`**
```python
device_map = "auto"
```

**Ý nghĩa**: Cách distribute model across devices (GPUs)

---

### 0.2. `FastLanguageModel.get_peft_model()`

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=256,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    max_seq_length=8192,
)
```

**Ý nghĩa**: Apply LoRA (Low-Rank Adaptation) to pretrained model

**PEFT (Parameter-Efficient Fine-Tuning)**:


### 0.3. `SFTTrainer`

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=datasets['train'],
    eval_dataset=datasets.get('val'),
    dataset_text_field="text",
    packing=False,
    args=training_args,
    max_seq_length=8192,
)
```

**Ý nghĩa**: Supervised Fine-Tuning Trainer - specialized cho instruction tuning


#### **SFTTrainer (TRL library)**
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=8192,
    packing=False,
    args=training_args,
)
```

#### **`packing`**
```python
packing = False
```

**Ý nghĩa**: Pack multiple short sequences into one long sequence

**WITHOUT packing** (packing=False):
```python
# Batch of 3 examples
Example 1: [200 tokens] + [7892 padding] = 8192 tokens
Example 2: [150 tokens] + [7942 padding] = 8192 tokens
Example 3: [300 tokens] + [7792 padding] = 8192 tokens

# Efficiency: (200+150+300) / (3×8192) = 2.7%
# Wasted: 97.3% is padding! ❌
```

**WITH packing** (packing=True):
```python
# Pack multiple into one sequence
Packed: [200 tokens][SEP][150 tokens][SEP][300 tokens] + padding = 8192
# Efficiency: 650 / 8192 = 7.9%
# Still wasteful but 3x better!

# OR pack even more:
Packed: [200][SEP][150][SEP][300][SEP][180][SEP][220]... = 8192
# Efficiency: ~90%! Much better! ✅
```

**Training flow with SFTTrainer**:
```python
# 1. Data preparation
dataset = [
    {"text": "<|begin_of_text|>...<|eot_id|>"},
    {"text": "<|begin_of_text|>...<|eot_id|>"},
    ...
]

# 2. SFTTrainer initialization
trainer = SFTTrainer(
    model=model,  # LoRA model from get_peft_model
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=8192,
)

# 3. Training loop (automatic)
for epoch in range(num_epochs):
    for batch in dataloader:
        # Tokenize
        inputs = tokenizer(batch["text"], 
                          max_length=8192,
                          truncation=True,
                          padding="max_length")
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Update (only LoRA!)
        optimizer.step()
        
        # Log
        if step % logging_steps == 0:
            log_metrics({"loss": loss.item()})

# 4. Save model
trainer.save_model("./outputs/final_model")
```

**Why use SFTTrainer for Vietnamese Legal?**
```python
# 1. Optimized for instruction following
# Legal Q&A is instruction-following task

# 2. Better handling of long sequences
# Legal documents: 4000-8000 tokens

# 3. Automatic sequence formatting
# Handles Llama chat template properly

# 4. Memory efficient
# Works well with LoRA + gradient checkpointing

# 5. Easy evaluation
# Built-in eval loop during training
```
---

## 1. Model Parameters

### 1.1. `model_name`
```python
model_name: str = "unsloth/Llama-3.1-8B-Instruct"
```

**Ý nghĩa**: Tên model trên HuggingFace Hub

**Các options phổ biến**:
- `"unsloth/Llama-3.1-8B-Instruct"` - Model đã tối ưu bởi Unsloth
- `"meta-llama/Llama-3.1-8B-Instruct"` - Model gốc từ Meta
- `"unsloth/Llama-3.2-1B-Instruct"` - Model nhỏ hơn cho GPU yếu

**Tại sao chọn Unsloth version?**
- Tối ưu hóa sẵn (Flash Attention 2, kernel fusion)
- Tốc độ training nhanh hơn 2-5x
- Sử dụng ít VRAM hơn

---

### 1.2. `max_seq_length`
```python
max_seq_length: int = 8192
```

**Ý nghĩa**: Độ dài tối đa của sequence (số tokens)

**Tại sao quan trọng?**
```
Input text: "Điều 1. Văn bản này quy định về..."
Tokenization: ["Điều", "1", ".", "Văn", "bản", ...]
Tokens: [123, 456, 789, ...]  # max_seq_length tokens
```

**Context trong Legal QA**:
- Văn bản pháp luật thường dài (nhiều điều khoản)
- 8192 tokens ≈ 6000-7000 từ tiếng Việt
- Đủ cho hầu hết documents + Q&A

**Công thức VRAM**:
```
VRAM ≈ Model_Size × Seq_Length² × Batch_Size × Precision
```
---

### 1.3. `dtype` (Data Type)
```python
dtype: Optional[torch.dtype] = None
```

**Các kiểu dtype**:

| Type | Bits | Range | Use Case |
|------|------|-------|----------|
| `torch.float32` (FP32) | 32 | ±3.4×10³⁸ | Inference chính xác cao |
| `torch.float16` (FP16) | 16 | ±65,504 | Training/inference nhanh |
| `torch.bfloat16` (BF16) | 16 | ±3.4×10³⁸ | Training ổn định hơn FP16 |
| `torch.int8` | 8 | -128 to 127 | Quantization |

**So sánh FP16 vs BF16**:

```
FP32 (32-bit):
[Sign: 1 bit][Exponent: 8 bits][Mantissa: 23 bits]
Range: ±3.4×10³⁸, Precision: ~7 decimal digits

FP16 (16-bit):
[Sign: 1 bit][Exponent: 5 bits][Mantissa: 10 bits]
Range: ±65,504, Precision: ~3 decimal digits
❌ Dễ overflow/underflow trong training!

BF16 (16-bit):
[Sign: 1 bit][Exponent: 8 bits][Mantissa: 7 bits]
Range: ±3.4×10³⁸ (giống FP32!), Precision: ~2 decimal digits
✅ Ổn định hơn cho training!
```

**Trong code**:
```python
dtype = None  # Unsloth tự chọn (thường BF16 nếu GPU hỗ trợ)
dtype = torch.bfloat16  # Force BF16 (H200, A100 support)
dtype = torch.float16   # Force FP16 (older GPUs)
```

**Tại sao H200 dùng BF16?**
- Hardware support native BF16
- Ổn định hơn FP16 cho large models
- Giảm VRAM 50% so với FP32
- Training speed gần như FP16

---

### 1.4. `load_in_4bit`
```python
load_in_4bit: bool = False
```

**Ý nghĩa**: Load model với 4-bit quantization để tiết kiệm VRAM

**Quantization là gì?**
```python
# Giảm precision của weights để save memory

# FP16 (16-bit) - Normal
Weight value: 1.234567890
Bits: 16 bits
Memory: 2 bytes per parameter

# INT4 (4-bit) - Quantized
Weight value: 12 (rounded/approximate)
Bits: 4 bits  
Memory: 0.5 bytes per parameter
Reduction: 75%! (2 bytes → 0.5 bytes)
```

**So sánh model sizes**:
```python
Llama-3.1-8B (8 billion parameters)

# FP16/BF16 (16-bit) - STANDARD cho H200
Size: 8B × 2 bytes = 16 GB
VRAM needed: ~20 GB (with overhead)

# INT4 (4-bit) - Heavily quantized
Size: 8B × 0.5 bytes = 4 GB
VRAM needed: ~5-6 GB (with overhead)
```

**QLoRA (Quantized LoRA)**:
```python
# Kết hợp 4-bit quantization + LoRA

# Normal LoRA (FP16)
Base model: 16 GB (FP16)
LoRA adapters: 0.5 GB (trainable)
Total VRAM (training): ~25-30 GB

# QLoRA (4-bit base + FP16 LoRA)
Base model: 4 GB (INT4, frozen)
LoRA adapters: 0.5 GB (FP16, trainable)
Total VRAM (training): ~8-12 GB
Reduction: 60-75%!
```


**Cách hoạt động trong Unsloth**:
```python
# With load_in_4bit=True
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
)

# Under the hood:
# 1. Load model weights
# 2. Quantize ALL weights to 4-bit (INT4)
# 3. Freeze quantized weights (không train)
# 4. Add LoRA adapters (FP16, trainable)
# 5. During training:
#    - Quantized weights: frozen, 4-bit
#    - LoRA adapters: trainable, FP16
#    - Forward pass: dequantize on-the-fly
#    - Backward pass: only update LoRA
```

---

## 2. LoRA Parameters

### 2.1. LoRA là gì?

**Full Fine-tuning (Traditional)**:
```python
# Update ALL parameters
Model Parameters: 8 billion
Trainable: 8 billion (100%)
VRAM: ~140 GB (FP16)
Time: Rất lâu
```

**LoRA (Low-Rank Adaptation)**:
```python
# Freeze original model, add small trainable matrices
Model Parameters: 8 billion
Trainable: ~100 million (1.25%)
VRAM: ~30 GB (FP16)
Time: Nhanh hơn 3-5x
```

**Cơ chế hoạt động**:
```
Original weight matrix W (frozen):
[4096 x 4096] = 16M parameters

LoRA decomposition:
W_new = W + BA
B: [4096 x r]
A: [r x 4096]
Total LoRA params: 4096×r + r×4096 = 2×4096×r

Example with r=128:
LoRA params: 2 × 4096 × 128 = 1M parameters
Reduction: 16M → 1M (94% reduction!)
```

---

### 2.2. `lora_r` (LoRA Rank)
```python
lora_r: int = 128
```

**Ý nghĩa**: Rank của decomposition matrices


**Công thức số parameters**:
```python
# Cho mỗi target module (q_proj, k_proj, v_proj, o_proj)
module_size = 4096  # Hidden size của Llama-3.1-8B
params_per_module = 2 × module_size × lora_r

# Total với 4 attention modules + 3 MLP modules
total_lora_params = 7 × 2 × 4096 × lora_r
                  = 57,344 × lora_r

# With r=128:
total = 57,344 × 128 = 7,340,032 parameters ≈ 7.3M parameters

# With r=64:
total = 57,344 × 64 = 3,670,016 parameters ≈ 3.7M parameters
```

---

### 2.3. `lora_alpha`
```python
lora_alpha: int = 256  # Usually 2 × lora_r
```

**Ý nghĩa**: Scaling factor cho LoRA updates

**Công thức**:
```python
# LoRA update
ΔW = (lora_alpha / lora_r) × B × A

# With lora_r=128, lora_alpha=256:
scaling = 256 / 128 = 2.0
ΔW = 2.0 × B × A

# With lora_r=64, lora_alpha=128:
scaling = 128 / 64 = 2.0
ΔW = 2.0 × B × A
```

**Best practice**:
```python
lora_alpha = 2 × lora_r  # Standard ratio
```

**Tại sao cần lora_alpha?**
- Normalize learning magnitude across different ranks
- Higher alpha → stronger LoRA influence
- Lower alpha → more conservative updates

---

### 2.4. `lora_dropout`
```python
lora_dropout: float = 0.0
```

**Ý nghĩa**: Dropout rate trong LoRA layers

**Dropout mechanism**:
```python
# During training
output = dropout(x, p=0.1)  # Randomly zero 10% of values
# [1.2, 0.8, 1.5, 0.9] → [1.2, 0.0, 1.5, 0.9]

# Prevents overfitting by forcing network to be robust
```

**Trade-offs**:
| Dropout | Overfitting Prevention | Training Speed | Convergence |
|---------|----------------------|----------------|------------|
| 0.0 | Thấp | Nhanh nhất | Nhanh |
| 0.05 | Trung bình | Nhanh | Trung bình |
| 0.1 | Cao | Chậm | Chậm hơn |

**Unsloth recommendation**:
```python
lora_dropout = 0.0  # Unsloth tối ưu cho dropout=0
# Faster training, ít regularization
```

**Khi nào dùng dropout?**
```python
# Small dataset (<10k examples)
lora_dropout = 0.1  # Prevent overfitting

# Large dataset (>100k examples)
lora_dropout = 0.0  # No need, faster training
```

---

### 2.5. `use_rslora` (Rank-Stabilized LoRA)
```python
use_rslora: bool = True
```

**Vấn đề với standard LoRA**:
```python
# Với rank cao (r=128), updates có thể quá lớn
ΔW = (alpha/r) × B × A
# Có thể unstable, diverge

# Với rank thấp (r=16), updates có thể quá nhỏ
# Convergence chậm
```

**RSLoRA solution**:
```python
# Thêm 1/√r scaling
ΔW = (alpha / r) × (1 / √r) × B × A
    = (alpha / r^1.5) × B × A

# With r=128:
Standard LoRA: scaling = alpha / 128
RSLoRA: scaling = alpha / (128^1.5) = alpha / 1448

# More stable với high rank!
```

**Khi nào dùng?**
```python
# High rank (r >= 64)
use_rslora = True  # Recommended!

# Low rank (r < 64)
use_rslora = False  # Không cần thiết
```

---

### 2.6. `bias` (LoRA Bias)
```python
bias: str = "none"
```

**Ý nghĩa**: Có train bias terms trong LoRA không?

**Bias trong Neural Networks**:
```python
# Linear layer with bias
output = W × input + b
         ^^^^^^^^   ^^^
         weights    bias term
```

**Bias trong LoRA context**:
```python
# Original layer (frozen)
output = W × input + b_original
         ^           ^^^^^^^^^^^
         frozen      frozen

# LoRA modification
output = (W + ΔW) × input + b
         ^^^^^^^^           ^
         W frozen           bias options:
         ΔW = BA trainable  - "none": b = b_original (frozen)
                            - "all": b = b_original + Δb (trainable)
                            - "lora_only": new b for LoRA only
```

**Options**:

#### **1. `bias = "none"` (RECOMMENDED!)** ✅
```python
bias = "none"
```

#### **2. `bias = "all"`**
```python
bias = "all"
```

#### **3. `bias = "lora_only"`**
```python
bias = "lora_only"

```

---

### 2.7. `use_gradient_checkpointing`
```python
use_gradient_checkpointing: str = "unsloth"
```

**Ý nghĩa**: Gradient checkpointing strategy để save VRAM

**Gradient Checkpointing là gì?**
```python
# Normal training (NO checkpointing)
Forward pass:
  Input → Layer1 → Layer2 → Layer3 → ... → Layer32 → Output
          ↓ SAVE  ↓ SAVE  ↓ SAVE      ↓ SAVE
        Act1     Act2     Act3       Act32

Backward pass:
  Use saved activations to compute gradients
  
Memory usage: HUGE!
# Save ALL 32 layers' activations
# Llama-3.1-8B, seq=8192: ~100 GB just for activations!
```

**With Gradient Checkpointing**:
```python
# Checkpointing strategy
Forward pass:
  Input → Layer1-8 → Layer9-16 → Layer17-24 → Layer25-32 → Output
          ↓ SAVE    ↓ SAVE      ↓ SAVE       ↓ SAVE
         CP1        CP2         CP3          CP4
         
Backward pass:
  Recompute activations between checkpoints when needed
  
Memory usage: Much lower!
# Only save 4 checkpoints instead of 32 layers
# Llama-3.1-8B, seq=8192: ~30 GB for activations
# Reduction: 70%!

Trade-off: +20-30% training time (recomputation cost)
```

**Options trong Unsloth**:

#### **1. `use_gradient_checkpointing = "unsloth"` (RECOMMENDED!)** ✅
```python
use_gradient_checkpointing = "unsloth"

# Unsloth's optimized checkpointing strategy
# Tối ưu hóa đặc biệt cho LoRA training

# Features:
# - Smart checkpoint placement
# - Optimized recomputation
# - Minimal slowdown (~10-15% instead of 20-30%)
# - Maximum VRAM savings

# Best cho:
# - LoRA fine-tuning (our case!)
# - Long sequences (8192 tokens)
# - Limited VRAM scenarios
```

#### **2. `use_gradient_checkpointing = True`**
```python
use_gradient_checkpointing = True

# Standard gradient checkpointing
# Uses HuggingFace/PyTorch default strategy

# Features:
# - Generic checkpointing
# - Works but not optimized for LoRA
# - ~20-30% slowdown

# Kém hơn "unsloth" option
```

#### **3. `use_gradient_checkpointing = False`**
```python
use_gradient_checkpointing = False

# NO checkpointing
# Save ALL activations

# Use when:
# - Plenty of VRAM (>120 GB)
# - Need maximum speed
# - Short sequences (<2048 tokens)

# Our H200 case: Still use "unsloth"!
# Because 8192 seq length needs checkpointing
```

---

### 2.8. `target_modules`
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
    "embed_tokens", "lm_head"                 # Embeddings
]
```

**Llama Architecture**:
```
Input
  ↓
Embeddings (embed_tokens)
  ↓
[Transformer Block] × 32 layers
  ├── Attention
  │   ├── q_proj (Query)
  │   ├── k_proj (Key)
  │   ├── v_proj (Value)
  │   └── o_proj (Output)
  └── MLP
      ├── gate_proj (Gate)
      ├── up_proj (Up)
      └── down_proj (Down)
  ↓
LM Head (lm_head)
  ↓
Output logits
```

---

## 3. Training Parameters

### 3.1. `per_device_train_batch_size`
```python
per_device_train_batch_size: int = 32
```

**Ý nghĩa**: Số examples xử lý đồng thời trên 1 GPU

**Ví dụ**:
```python
# Batch size = 2
Batch = [
    "Điều 1. Văn bản này...",
    "Điều 2. Phạm vi..."
]
# Process cùng lúc 2 examples

# Batch size = 32
# Process cùng lúc 32 examples
```

**Trade-offs**:
| Batch Size | VRAM | Speed | Gradient Quality | Convergence |
|-----------|------|-------|-----------------|-------------|
| 1 | Thấp nhất | Chậm nhất | Noisy | Chậm |
| 8 | Thấp | Chậm | OK | Trung bình |
| 16 | Trung bình | Trung bình | Tốt | Tốt |
| 32 | Cao | Nhanh | Rất tốt | Nhanh |
| 64 | Rất cao | Rất nhanh | Smooth | Rất nhanh |

---

### 3.2. `gradient_accumulation_steps`
```python
gradient_accumulation_steps: int = 16
```

**Ý nghĩa**: Tích lũy gradients qua nhiều mini-batches

**Cơ chế**:
```python
# WITHOUT gradient accumulation
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Update sau mỗi batch
    optimizer.zero_grad()

# WITH gradient accumulation (steps=4)
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / gradient_accumulation_steps  # Scale loss
    loss.backward()  # Accumulate gradients
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()  # Update sau 4 batches
        optimizer.zero_grad()
```

**Effective batch size**:
```python
effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × num_gpus

# Example: H200 config
per_device = 32
accumulation = 16
num_gpus = 1
effective = 32 × 16 × 1 = 512  # HUGE!
```

---

### 3.3. `num_train_epochs`
```python
num_train_epochs: int = 4
```

**Ý nghĩa**: Số lần duyệt qua toàn bộ training data

**Ví dụ**:
```python
# Dataset: 10,000 examples
# Batch size: 32
# Steps per epoch: 10,000 / 32 = 313 steps

# 1 epoch: 313 steps
# 3 epochs: 939 steps
# 5 epochs: 1,565 steps
```

**Chọn số epochs**:
```python
# Small dataset (<1k examples)
num_train_epochs = 10-20  # Nhiều epochs để học kỹ

# Medium dataset (1k-10k)
num_train_epochs = 5-10

# Large dataset (10k-100k)
num_train_epochs = 3-5

# Very large dataset (>100k)
num_train_epochs = 1-3

# Legal domain (complex reasoning)
num_train_epochs = 4  # Balance giữa learning và overfitting
```

**Overfitting check**:
```python
# Monitor validation loss
Epoch 1: train_loss=2.5, val_loss=2.6  # Good
Epoch 2: train_loss=2.0, val_loss=2.1  # Good
Epoch 3: train_loss=1.5, val_loss=1.6  # Good
Epoch 4: train_loss=1.0, val_loss=1.8  # ⚠️ Overfitting!
# Stop training or reduce epochs
```

---

### 3.4. `learning_rate`
```python
learning_rate: float = 3e-4  # 0.0003
```

**Ý nghĩa**: Tốc độ cập nhật weights

**Cơ chế**:
```python
# Gradient descent
weight_new = weight_old - learning_rate × gradient

# Example
weight = 1.0
gradient = 0.5
lr = 0.01

weight_new = 1.0 - 0.01 × 0.5 = 0.995
```

**Trade-offs**:
| Learning Rate | Convergence Speed | Stability | Final Performance |
|--------------|------------------|-----------|------------------|
| 1e-5 (0.00001) | Rất chậm | Rất stable | Tốt |
| 1e-4 (0.0001) | Chậm | Stable | Rất tốt |
| 3e-4 (0.0003) | Trung bình | Tốt | Tốt nhất |
| 5e-4 (0.0005) | Nhanh | Trung bình | Tốt |
| 1e-3 (0.001) | Rất nhanh | Unstable | Có thể kém |

---

### 3.5. `warmup_steps`
```python
warmup_steps: int = 100
```

**Ý nghĩa**: Số steps tăng dần learning rate từ 0 → target

**Tại sao cần warmup?**
```python
# WITHOUT warmup
# Training starts với full LR ngay từ đầu
# Model chưa stable → gradients lớn → divergence risk

# WITH warmup
# LR tăng dần: 0 → 3e-4 trong 100 steps
# Model có thời gian stabilize → safer training
```

**Warmup schedule**:
```python
# Linear warmup
step = 0:   lr = 0.0
step = 25:  lr = 3e-4 × (25/100) = 7.5e-5
step = 50:  lr = 3e-4 × (50/100) = 1.5e-4
step = 75:  lr = 3e-4 × (75/100) = 2.25e-4
step = 100: lr = 3e-4 × (100/100) = 3e-4  # Full LR
step > 100: lr follows lr_scheduler
```

---

### 3.6. `lr_scheduler_type`
```python
lr_scheduler_type: str = "cosine"
```

**Các loại schedulers**:

#### **1. Constant**
```python
lr = 3e-4  # Không đổi suốt training
```

#### **2. Linear Decay**
```python
# Giảm tuyến tính từ max_lr → 0
step = 0:    lr = 3e-4
step = 500:  lr = 1.5e-4
step = 1000: lr = 0
```

#### **3. Cosine Annealing** (RECOMMENDED!)
```python
# Giảm theo cosine curve
lr(t) = lr_min + (lr_max - lr_min) × (1 + cos(πt/T)) / 2

# Visualization:
3e-4 |     ╱╲
     |    ╱  ╲
     |   ╱    ╲___
     |  ╱         ╲___
0    |_________________
     0  steps      T

# Đặc điểm:
# - Start: giảm nhanh (explore)
# - Middle: giảm chậm (refine)
# - End: giảm rất chậm (fine-tune)
```

**Tại sao cosine tốt nhất?**
1. **Explore early**: High LR ban đầu cho convergence nhanh
2. **Refine later**: Low LR cuối cho fine-tuning
3. **Smooth decay**: Không có sudden drops như linear
4. **SOTA results**: Best practice trong research

---

### 3.7. `weight_decay`
```python
weight_decay: float = 0.01
```

**Ý nghĩa**: L2 regularization - penalty cho weights lớn

**Cơ chế**:
```python
# Normal loss
loss = model_output_loss

# With weight_decay
loss = model_output_loss + weight_decay × Σ(w²)
# Penalty weights lớn → prefer smaller weights

# Weight update
w_new = w_old - lr × gradient - lr × weight_decay × w_old
      = w_old × (1 - lr × weight_decay) - lr × gradient
```

**Tại sao cần?**
```python
# Prevent overfitting
# Large weights → model memorizes training data
# Small weights → model generalizes better
```

---

## 4. Optimization Parameters

### 4.1. `optim` (Optimizer)
```python
optim: str = "adamw_torch"
```

**Các optimizers**:

#### **1. SGD (Stochastic Gradient Descent)**
```python
w_new = w_old - lr × gradient
# Simple, stable, nhưng chậm converge
```

#### **2. Adam**
```python
# Adaptive learning rate cho mỗi parameter
# Momentum + RMSProp
m = β1 × m + (1-β1) × gradient  # Momentum
v = β2 × v + (1-β2) × gradient²  # RMSProp
w = w - lr × m / √v

# Fast convergence, popular
```

**So sánh**:
| Optimizer | Speed | Stability | Memory | Use Case |
|-----------|-------|-----------|--------|----------|
| SGD | Chậm | Rất stable | Thấp | Research |
| Adam | Nhanh | Tốt | Trung bình | General |
| AdamW | Nhanh | Tốt nhất | Trung bình | SOTA (recommended!) |

**Variants trong code**:
```python
"adamw_torch"       # PyTorch implementation (standard)
"adamw_hf"          # HuggingFace implementation
"adamw_8bit"        # 8-bit AdamW (save memory)
"adamw_bnb_8bit"    # BitsAndBytes 8-bit (even less memory)
"adafactor"         # Memory-efficient alternative
```

---

---

### 4.3. `gradient_checkpointing`
```python
gradient_checkpointing: bool = True
```

**Cơ chế**:

**WITHOUT gradient checkpointing**:
```python
# Forward pass: lưu ALL activations
Input → Layer1 → Layer2 → ... → Layer32 → Output
        ↓ save  ↓ save       ↓ save
        Act1    Act2         Act32

# Backward pass: dùng saved activations
# Memory: O(num_layers × seq_length × hidden_size)
# For Llama-3.1-8B: ~100 GB với seq=8192!
```

**WITH gradient checkpointing**:
```python
# Forward pass: CHỈ lưu checkpoints (mỗi N layers)
Input → Layer1 → ... → Layer8 → ... → Layer16 → ... → Output
                       ↓ save          ↓ save
                       CP1             CP2

# Backward pass: recompute activations từ checkpoints
# Memory: O(√num_layers × seq_length × hidden_size)
# For Llama-3.1-8B: ~30 GB với seq=8192!
# Saving: 70%!

# Trade-off: +20-30% training time
```

**Khi nào dùng?**
```python
# Limited VRAM (<40GB)
gradient_checkpointing = True  # Essential!

# Plenty VRAM (>80GB) và cần speed
gradient_checkpointing = False

# H200 (141GB) với long sequences
gradient_checkpointing = True  # Still recommended!
```
---

## 5. Data Loading Parameters

### 5.1. `dataloader_num_workers`
```python
dataloader_num_workers: int = 8
```

**Ý nghĩa**: Số CPU processes load data song song

```

```

**Trade-offs**:
- **Too low** (0-2): Data loading bottleneck, GPU idle
- **Optimal** (4-8): GPU utilization 100%
- **Too high** (>8): Overhead, no benefit, RAM pressure

---

### 5.2. `dataloader_pin_memory`
```python
dataloader_pin_memory: bool = True
```

**Ý nghĩa**: Pin data trong RAM để transfer CPU→GPU nhanh hơn

---

## 6. Logging & Saving Parameters

### 6.1. `logging_steps`
```python
logging_steps: int = 1
```

**Ý nghĩa**: Log metrics mỗi N steps

```python
# logging_steps = 1
Step 1: loss=2.5
Step 2: loss=2.3
Step 3: loss=2.1
...

# logging_steps = 10
Step 10: loss=2.1
Step 20: loss=1.9
Step 30: loss=1.7
...
```

**Trade-offs**:
```python
# Frequent logging (1-10)
# + Real-time monitoring
# + Detailed loss curves
# - Slower (logging overhead)
# - Large log files

# Infrequent logging (50-100)
# + Faster training
# + Smaller logs
# - Miss details
# - Harder to debug
```

---

### 6.2. `save_strategy` và `evaluation_strategy`
```python
save_strategy: str = "epoch"
evaluation_strategy: str = "epoch"
```

**Options**:
```python
"no"       # Không save/eval
"steps"    # Save/eval mỗi N steps
"epoch"    # Save/eval mỗi epoch (RECOMMENDED!)
```

**Example với "epoch"**:
```python
Epoch 1: train → evaluate → save checkpoint
Epoch 2: train → evaluate → save checkpoint
Epoch 3: train → evaluate → save checkpoint
...
```

**Example với "steps" (save_steps=500)**:
```python
Step 500: save checkpoint
Step 1000: save checkpoint
Step 1500: save checkpoint
...
```

---

### 6.3. `save_total_limit`
```python
save_total_limit: int = 3
```

**Ý nghĩa**: Giữ tối đa N checkpoints gần nhất

```python
# save_total_limit = 3

# After epoch 1
checkpoints/
  ├── checkpoint-1/

# After epoch 2
checkpoints/
  ├── checkpoint-1/
  └── checkpoint-2/

# After epoch 3
checkpoints/
  ├── checkpoint-1/
  ├── checkpoint-2/
  └── checkpoint-3/

# After epoch 4 (delete oldest!)
checkpoints/
  ├── checkpoint-2/
  ├── checkpoint-3/
  └── checkpoint-4/  # checkpoint-1 deleted!
```

**Disk space calculation**:
```python
# Llama-3.1-8B với LoRA r=128
checkpoint_size ≈ 1 GB

# save_total_limit = 3
total_disk = 3 × 1 GB = 3 GB

# save_total_limit = 10
total_disk = 10 × 1 GB = 10 GB
```

---
### 7.2. `from_pretrained()`
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=8192,
    dtype=None,
    load_in_4bit=False,
)
```

**Cơ chế**:
1. **Download weights** từ HuggingFace Hub (nếu chưa có local)
2. **Load weights** vào memory
3. **Setup model architecture**
4. **Apply optimizations** (Flash Attention, kernel fusion)

**Bình thường load bao nhiêu bit?**
```python
# Mặc định: Precision của pretrained weights
"meta-llama/Llama-3.1-8B"  # Usually FP16 (16-bit)

# With dtype parameter
dtype=None          # Auto-detect (usually FP16)
dtype=torch.bfloat16  # Force BF16
dtype=torch.float16   # Force FP16
dtype=torch.float32   # Force FP32 (huge VRAM!)

# With quantization
load_in_4bit=True   # Quantize to 4-bit after loading
load_in_8bit=True   # Quantize to 8-bit after loading
```

---

