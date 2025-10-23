# Phân Tích Chi Tiết Model Fine-tuning Setup

## 📊 Dữ Liệu Training

### Thống kê dữ liệu:
- **Số lượng documents**: 1,875,511 documents
- **Độ dài trung bình**: ~345 ký tự/document
- **Format**: JSONL với structure `{"id": "...", "text": "..."}`
- **Ngôn ngữ**: Tiếng Việt - Legal domain
- **Chất lượng**: Đã được preprocess trong data pipeline

### Đánh giá dữ liệu:
✅ **TÍCH CỰC**:
- Số lượng rất lớn (1.8M+ documents) - đủ để fine-tune hiệu quả
- Specific domain (Vietnamese legal) - phù hợp cho specialized embedding
- Độ dài vừa phải (~345 chars) - không quá dài, không quá ngắn
- Đã được clean và preprocess

⚠️ **CẦN LƯU Ý**:
- Cần kiểm tra distribution độ dài text (có documents quá dài không?)
- Cần verify quality của một số samples random
- Nên có validation set để đánh giá performance

## 🤖 Base Model Analysis

### Model hiện tại:
```
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

### Đặc điểm model:
- **Architecture**: MiniLM-L12 (12-layer transformer)
- **Language support**: Multilingual (104 languages including Vietnamese)
- **Vector dimension**: 384
- **Max sequence length**: 512 tokens
- **Size**: ~118MB
- **Performance**: Good balance between quality & speed

### Đánh giá lựa chọn:
✅ **TUYỆT VỜI**:
- Support Vietnamese tốt
- Đã được pre-train trên multilingual data
- Kích thước vừa phải, inference nhanh
- Proven performance trên sentence similarity tasks
- Compatible với sentence-transformers framework

📈 **ALTERNATIVES NÊN XEMXÉT**:
```python
# Nếu cần performance cao hơn (nhưng chậm hơn):
'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'  # 768 dims

# Nếu cần model nhỏ hơn (nhưng accuracy thấp hơn):
'sentence-transformers/paraphrase-MiniLM-L6-v2'  # 384 dims, 6 layers
```

## ⚙️ Hyperparameters Analysis

### Current settings:
```python
hyperparameters = {
    'base-model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'max-seq-length': 512,
    'batch-size': 16,
    'epochs': 3,
    'learning-rate': 2e-5,
}
```

### Detailed Analysis:

#### 1. **Max Sequence Length: 512**
✅ **PERFECT**: 
- Phù hợp với average text length (~345 chars ≈ ~100-150 tokens)
- Không waste computation trên padding
- Capture được context đầy đủ

#### 2. **Batch Size: 16**
⚠️ **CẦN ĐIỀU CHỈNH**:
```python
# Recommendations based on instance type:
ml.g4dn.xlarge (16GB GPU): batch_size = 32-64  # Có thể tăng
ml.g4dn.2xlarge (32GB GPU): batch_size = 64-128
ml.p3.2xlarge (16GB GPU): batch_size = 32-64

# Current batch_size = 16 là conservative, có thể tăng để:
# - Tăng training speed
# - Better gradient estimation
# - Improved model convergence
```

**RECOMMENDED**: Tăng lên `batch_size = 32` hoặc `48`

#### 3. **Epochs: 3**
⚠️ **CÓ THỂ CHƯA ĐỦ**:
```python
# Với 1.8M documents, mỗi epoch sẽ có rất nhiều steps
# Nhưng với specialized domain (legal), cần nhiều epochs hơn

# Recommendations:
epochs = 5-8  # Cho specialized domain
# Với early stopping based on validation loss
```

**RECOMMENDED**: Tăng lên `epochs = 5` với early stopping

#### 4. **Learning Rate: 2e-5**
✅ **TỐT**: 
- Standard learning rate cho fine-tuning sentence transformers
- Không quá cao (avoid catastrophic forgetting)
- Không quá thấp (training sẽ chậm)

**ALTERNATIVE**: Có thể thử learning rate schedule:
```python
# Warmup + cosine decay
warmup_steps = 1000
total_steps = (len(dataset) // batch_size) * epochs
```

## 🔧 Training Strategy Analysis

### Current approach:
```python
# Tạo positive pairs từ same documents (giả định documents liên quan)
# Và negative pairs từ random sampling
for i in range(len(texts)):
    for j in range(i + 1, min(i + 10, len(texts))):  # Positive pairs
        examples.append(InputExample(texts=[texts[i], texts[j]], label=0.8))
    
    # Negative pairs
    if i + 50 < len(texts):
        examples.append(InputExample(texts=[texts[i], texts[i + 50]], label=0.2))
```

⚠️ **VẤN ĐỀ NGHIÊM TRỌNG**:

1. **Weak positive pairs**: Giả định documents liên tiếp có similarity cao là KHÔNG chính xác
2. **Arbitrary labels**: 0.8 và 0.2 không có cơ sở thực tế
3. **Poor negative sampling**: Documents cách xa 50 positions không đảm bảo là negative

### 🚀 RECOMMENDED IMPROVEMENTS:

#### Option 1: Self-supervised approach
```python
# Sử dụng các techniques tốt hơn:
from sentence_transformers.losses import MultipleNegativesRankingLoss

# Không cần manual labeling, model tự học
# Mỗi sentence là positive với chính nó, negative với others trong batch
```

#### Option 2: Domain-specific approach  
```python
# Tạo better training pairs based on:
# - Overlapping legal concepts
# - Similar document types
# - Semantic similarity using existing embeddings
```

## 📈 RECOMMENDED CONFIGURATION

```python
# Optimized hyperparameters
hyperparameters = {
    'base-model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'max-seq-length': 512,
    'batch-size': 48,  # Tăng từ 16
    'epochs': 6,       # Tăng từ 3  
    'learning-rate': 2e-5,
    'warmup-steps': 1000,
    'evaluation-steps': 500,
    'save-steps': 1000
}

# Training strategy
training_strategy = "MultipleNegativesRankingLoss"  # Thay vì CosineSimilarityLoss
```

## 💰 Cost & Time Estimation

### With current config (batch_size=16, epochs=3):
- **Training time**: ~6-8 hours trên ml.g4dn.xlarge
- **Cost**: ~$10-15

### With recommended config (batch_size=48, epochs=6):
- **Training time**: ~8-12 hours  
- **Cost**: ~$15-20

### Cost optimization:
- Sử dụng Spot instances: giảm 50-70% cost
- Mixed precision training: tăng speed ~30%

## 🎯 Expected Results

### Với current setup:
- Moderate improvement trên Vietnamese legal domain
- Better than base multilingual model nhưng không optimal

### Với recommended setup:  
- Significant improvement trên legal terminology
- Better semantic understanding của legal concepts
- Improved retrieval performance cho RAG system

## ✅ Action Items

1. **Immediate fixes**:
   - Tăng batch_size lên 48
   - Tăng epochs lên 6
   - Thêm early stopping

2. **Training strategy improvement**:
   - Thay CosineSimilarityLoss bằng MultipleNegativesRankingLoss
   - Remove manual positive/negative pair creation

3. **Evaluation setup**:
   - Tạo evaluation dataset
   - Setup proper metrics (cosine similarity, retrieval accuracy)

4. **Monitoring**:
   - Track loss curves
   - Validate trên legal query examples
   - Compare với base model performance