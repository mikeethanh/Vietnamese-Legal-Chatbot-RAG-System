# BGE-M3 Model - Comprehensive Technical Documentation

## 📋 Tổng quan về BGE-M3

**BGE-M3** (BAAI General Embedding - Multilingual, Multi-Granularity, Multi-Functionality) là một breakthrough trong lĩnh vực text embedding, được phát triển bởi Beijing Academy of Artificial Intelligence (BAAI). Model này đại diện cho sự tiến bộ vượt bậc trong việc tạo ra unified embedding space cho multiple languages và functionalities.

## 📄 Research Foundation

### Paper Reference
**Title**: "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
**Authors**: Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, Zheng Liu
**Publication**: arXiv:2402.03216, February 2024
**Link**: https://arxiv.org/abs/2402.03216

## 🎯 Ba Trụ Cột Chính (The 3 M's)

### 1. Multi-Lingual (Đa ngôn ngữ)

**Khái niệm**: Khả năng hiểu và tạo embeddings chất lượng cao cho **100+ ngôn ngữ** khác nhau, bao gồm tiếng Việt.

**Đặc điểm nổi bật**:
- **Cross-lingual Retrieval**: Có thể search bằng tiếng Việt trong corpus tiếng Anh và ngược lại
- **Language-Agnostic Training**: Training methodology không bias towards specific languages
- **Unified Embedding Space**: Tất cả languages được map vào cùng một vector space

**Vietnamese Performance**:
```
📊 BGE-M3 Vietnamese Metrics:
- MTEB Vietnamese: 68.2% (excellent performance)
- Cross-lingual EN→VI: 59.3% (best in class)
- Cross-lingual VI→EN: 61.1% (competitive)
```

### 2. Multi-Functionality (Đa chức năng)

BGE-M3 không chỉ là một embedding model mà integrates **3 retrieval paradigms** khác nhau:

#### 🔸 **Dense Embedding (Dense Vector Retrieval)**

**Khái niệm**: Traditional semantic embedding approach sử dụng dense vectors để capture semantic meaning.

**Cơ chế hoạt động**:
```python
# Dense embedding generation
dense_embedding = model.encode(text, return_dense=True)
# Output: [1024] dimensional vector with semantic information
```

**Đặc điểm**:
- **Semantic Understanding**: Hiểu ý nghĩa sâu của text
- **Context Aware**: Capture contextual relationships
- **Similarity Based**: Sử dụng cosine similarity cho ranking
- **Dimension**: 1024-dimensional vectors

**Use Cases**: 
- Semantic search
- Document similarity
- Clustering based on meaning

#### 🔸 **Sparse Embedding (Lexical/Keyword Retrieval)**

**Khái niệm**: Simulates traditional keyword-based retrieval (như BM25) nhưng learnable và có thể optimization.

**Cơ chế hoạt động**:
```python
# Sparse embedding generation  
sparse_embedding = model.encode(text, return_sparse=True)
# Output: Sparse vector với learned term importance weights
```

**Technical Implementation**:
- **Learned Term Weighting**: Thay vì TF-IDF, sử dụng neural network để weight terms
- **Vocabulary Expansion**: Có thể assign weights cho terms không xuất hiện trong text gốc
- **Sparsity Control**: Automatic sparsity regulation để balance performance vs efficiency

**Advantages over BM25**:
- **Learnable Weights**: Weights được optimize cho specific domain
- **Semantic Term Expansion**: Có thể weight related terms cao hơn
- **Cross-lingual**: Hoạt động across languages

**Use Cases**:
- Exact match requirements
- Legal document search (exact term matching)
- Hybrid search systems

#### 🔸 **Multi-Vector Embedding (Fine-grained Interaction)**

**Khái niệm**: Advanced approach sử dụng **multiple vectors per text** để capture fine-grained semantic interactions.

**Cơ chế hoạt động**:
```python
# Multi-vector embedding generation
multi_vectors = model.encode(text, return_multi_vector=True)
# Output: Multiple vectors representing different aspects của text
```

**Technical Details**:
- **Token-level Representations**: Mỗi important token có riêng vector representation
- **Interaction Modeling**: Model interactions giữa query tokens và document tokens
- **Maximum Inner Product Search (MIPS)**: Sử dụng MIPS thay vì cosine similarity

**Advantages**:
- **Fine-grained Matching**: Detailed token-to-token interactions
- **Higher Accuracy**: Better performance cho complex queries
- **Interpretability**: Có thể trace matching reasons

**Disadvantages**:
- **Storage Overhead**: Requires multiple vectors per document
- **Computational Cost**: More expensive similarity computation
- **Index Complexity**: More complex indexing requirements

### 3. Multi-Granularity (Đa độ chi tiết)

**Khái niệm**: Khả năng xử lý text ở **multiple levels of granularity** từ tokens đến documents.

#### **Granularity Levels**:

1. **Token Level**:
   - Individual word/subword representations
   - Fine-grained semantic analysis
   - Token-token interactions

2. **Sentence Level**:
   - Sentence embeddings
   - Intra-sentence relationships
   - Standard use case cho most applications

3. **Passage Level**:
   - Paragraph/passage representations
   - Long-form content understanding
   - Document section analysis

4. **Document Level**:
   - Entire document embeddings
   - Global semantic representation
   - Document-level similarity

**Technical Implementation**:
```python
# Multi-granularity processing
embeddings = model.encode(
    text,
    granularity=['token', 'sentence', 'passage', 'document']
)
```

## 🧠 Self-Knowledge Distillation

### Khái niệm Core

**Self-Knowledge Distillation** là breakthrough technique trong BGE-M3, cho phép model học từ chính bản thân nó để improve performance across multiple functionalities.

### Traditional vs Self-Knowledge Distillation

**Traditional Knowledge Distillation**:
```
Teacher Model (Large) → Student Model (Small)
                     ↓
               Knowledge Transfer
```

**Self-Knowledge Distillation**:
```
Model (Dense) ← → Model (Sparse) ← → Model (Multi-Vector)
    ↓              ↓                    ↓
      Self-Teaching & Cross-Functionality Learning
```

### Technical Mechanism

#### **Cross-Functionality Learning**:

1. **Dense → Sparse Knowledge Transfer**:
   ```python
   # Dense embeddings teach sparse embeddings
   dense_loss = compute_dense_loss(query_dense, doc_dense)
   sparse_loss = compute_sparse_loss(query_sparse, doc_sparse)
   
   # Distillation loss
   distill_loss = KL_divergence(dense_similarity, sparse_similarity)
   total_loss = dense_loss + sparse_loss + λ * distill_loss
   ```

2. **Sparse → Multi-Vector Knowledge Transfer**:
   ```python
   # Sparse weights guide multi-vector attention
   sparse_weights = get_sparse_weights(text)
   multi_vector_attention = compute_attention(
       tokens, 
       guided_by=sparse_weights
   )
   ```

3. **Multi-Vector → Dense Knowledge Transfer**:
   ```python
   # Multi-vector interactions improve dense representations
   fine_grained_signals = aggregate_multi_vector_interactions(
       query_vectors, doc_vectors
   )
   dense_embedding = enhance_dense_with_fine_grained(
       original_dense, fine_grained_signals
   )
   ```

#### **Iterative Self-Improvement**:

```python
# Pseudo-code for self-distillation process
for epoch in range(num_epochs):
    # Forward pass với all functionalities
    dense_emb = model.encode_dense(batch)
    sparse_emb = model.encode_sparse(batch) 
    multi_vec_emb = model.encode_multi_vector(batch)
    
    # Compute individual losses
    loss_dense = compute_dense_loss(dense_emb, labels)
    loss_sparse = compute_sparse_loss(sparse_emb, labels)
    loss_multi = compute_multi_vector_loss(multi_vec_emb, labels)
    
    # Cross-functionality distillation
    distill_loss = (
        kl_divergence(dense_sim, sparse_sim) +
        kl_divergence(dense_sim, multi_sim) + 
        kl_divergence(sparse_sim, multi_sim)
    )
    
    # Total loss with self-distillation
    total_loss = (
        loss_dense + loss_sparse + loss_multi + 
        λ * distill_loss
    )
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
```

### Ưu điểm của Self-Knowledge Distillation

1. **Unified Training**: Single model learns multiple functionalities simultaneously
2. **Cross-Functionality Synergy**: Each functionality improves others
3. **No External Teacher**: Không cần separate teacher models
4. **Consistent Performance**: All functionalities benefit from shared knowledge
5. **Efficiency**: One model serves multiple purposes

## 📊 Training Data và Methodology

### Dataset Composition

BGE-M3 được train trên massive multilingual dataset với diverse text types:

#### **Core Training Data**:

1. **Retrieval Datasets**:
   ```
   📚 MS MARCO Passages (English)
   📚 Natural Questions (English)
   📚 mMARCO (Multilingual version)
   📚 Mr. TyDi (Multilingual QA)
   ```

2. **Multilingual Corpora**:
   ```
   🌍 Wikipedia (100+ languages)
   🌍 Common Crawl (multilingual web data)
   🌍 News Corpora (multiple languages)
   🌍 Academic Papers (multilingual)
   ```

3. **Specialized Domains**:
   ```
   ⚖️  Legal Documents (multiple jurisdictions)
   🏥 Medical Literature
   💼 Business Documents
   🔬 Scientific Papers
   ```

### Training Format: Triplet vs Instruction

BGE-M3 sử dụng **combined training approach** với multiple formats:

#### **Triplet Format (Primary)**:

```python
# Triplet training format
triplet = {
    "query": "Luật lao động Việt Nam quy định về thời gian làm việc",
    "positive": "Theo Bộ luật Lao động 2019, thời gian làm việc bình thường không quá 8 giờ một ngày và không quá 48 giờ một tuần...",
    "negative": "Quy định về an toàn lao động trong môi trường làm việc..."
}
```

**Tại sao dùng Triplet Format?**:
- **Contrastive Learning**: Learn to distinguish between relevant và irrelevant content
- **Ranking Optimization**: Directly optimize cho retrieval ranking
- **Hard Negative Mining**: Improve model's ability to handle difficult cases

#### **Instruction Format (Secondary)**:

```python
# Instruction format for multi-functionality
instruction_data = {
    "instruction": "Generate dense embedding for semantic search",
    "input": "Vietnamese legal document about labor law",
    "output": "[dense_embedding_vector]"
},
{
    "instruction": "Generate sparse embedding for keyword matching", 
    "input": "Vietnamese legal document about labor law",
    "output": "{term_weights_sparse_vector}"
}
```

**Purpose của Instruction Format**:
- **Functionality Control**: Teach model when to use which functionality
- **Task Awareness**: Model learns to adapt behavior based on instructions
- **Multi-Task Learning**: Single model handles multiple tasks

### Training Stages

#### **Stage 1: Foundation Training**
```python
# Massive multilingual pre-training
for batch in multilingual_corpus:
    # Contrastive learning với large batch sizes
    embeddings = model.encode(batch)
    loss = contrastive_loss(embeddings, similarity_labels)
    loss.backward()
```

#### **Stage 2: Multi-Functionality Learning**
```python
# Joint training cho all functionalities
for batch in retrieval_data:
    dense_emb = model.encode_dense(batch)
    sparse_emb = model.encode_sparse(batch)
    multi_emb = model.encode_multi_vector(batch)
    
    # Individual losses
    loss_dense = contrastive_loss(dense_emb)
    loss_sparse = sparse_ranking_loss(sparse_emb) 
    loss_multi = multi_vector_loss(multi_emb)
    
    # Self-distillation loss
    distill_loss = cross_functionality_distillation(
        dense_emb, sparse_emb, multi_emb
    )
    
    total_loss = loss_dense + loss_sparse + loss_multi + distill_loss
```

#### **Stage 3: Fine-tuning và Specialization**
```python
# Domain-specific fine-tuning
for domain_batch in specialized_domains:
    # Fine-tune trên specific domains (legal, medical, etc.)
    embeddings = model.encode(domain_batch)
    domain_loss = domain_specific_loss(embeddings, domain_labels)
    
    # Maintain general capabilities
    general_loss = general_capability_loss(embeddings)
    
    total_loss = domain_loss + λ * general_loss
```

## 🏗️ Model Architecture Deep Dive

### Base Architecture

```python
# BGE-M3 Architecture Overview
class BGEM3Model(nn.Module):
    def __init__(self):
        # Backbone: XLM-RoBERTa Large (560M parameters)
        self.backbone = XLMRobertaModel.from_pretrained(
            'xlm-roberta-large'
        )
        
        # Dense embedding head
        self.dense_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.Tanh()
        )
        
        # Sparse embedding head
        self.sparse_head = nn.Sequential(
            nn.Linear(1024, vocab_size),
            nn.ReLU(),  # Ensure positive weights
            nn.Dropout(0.1)
        )
        
        # Multi-vector head
        self.multi_vector_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024)
        )
        
    def forward(self, input_ids, attention_mask, return_type='all'):
        # Backbone encoding
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        
        results = {}
        
        if return_type in ['dense', 'all']:
            # Dense embedding: CLS token pooling
            dense_emb = self.dense_head(pooler_output)
            results['dense'] = dense_emb
            
        if return_type in ['sparse', 'all']:
            # Sparse embedding: Token-level weights
            token_weights = self.sparse_head(last_hidden_state)
            # Apply attention mask
            token_weights = token_weights * attention_mask.unsqueeze(-1)
            results['sparse'] = token_weights
            
        if return_type in ['multi_vector', 'all']:
            # Multi-vector: Multiple representations
            multi_vectors = self.multi_vector_head(last_hidden_state)
            # Select top-k important tokens
            importance_scores = torch.norm(multi_vectors, dim=-1)
            top_k_indices = torch.topk(importance_scores, k=32).indices
            results['multi_vector'] = multi_vectors[top_k_indices]
            
        return results
```

### Embedding Mechanisms Chi Tiết

#### **Dense Embedding Process**:

```python
def encode_dense(self, texts):
    """Dense embedding generation"""
    # Tokenization
    inputs = self.tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=8192,
        return_tensors='pt'
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = self.model(**inputs, return_type='dense')
        dense_embeddings = outputs['dense']
        
        # L2 normalization for cosine similarity
        dense_embeddings = F.normalize(dense_embeddings, p=2, dim=1)
        
    return dense_embeddings
```

#### **Sparse Embedding Process**:

```python
def encode_sparse(self, texts):
    """Sparse embedding generation"""
    inputs = self.tokenizer(texts, ...)
    
    with torch.no_grad():
        outputs = self.model(**inputs, return_type='sparse')
        token_weights = outputs['sparse']  # [batch, seq_len, vocab_size]
        
        # Aggregate token weights cho each vocab term
        sparse_embeddings = []
        for i, text in enumerate(texts):
            text_tokens = inputs['input_ids'][i]
            text_weights = token_weights[i]
            
            # Create sparse vector
            sparse_vector = torch.zeros(self.vocab_size)
            for j, token_id in enumerate(text_tokens):
                if token_id != self.pad_token_id:
                    # Aggregate weights for each vocabulary term
                    sparse_vector[token_id] += text_weights[j, token_id]
                    
            sparse_embeddings.append(sparse_vector)
            
    return torch.stack(sparse_embeddings)
```

#### **Multi-Vector Embedding Process**:

```python
def encode_multi_vector(self, texts):
    """Multi-vector embedding generation"""
    inputs = self.tokenizer(texts, ...)
    
    with torch.no_grad():
        outputs = self.model(**inputs, return_type='multi_vector')
        all_vectors = outputs['multi_vector']  # [batch, seq_len, hidden]
        
        multi_vector_embeddings = []
        for i, text in enumerate(texts):
            text_vectors = all_vectors[i]  # [seq_len, hidden]
            attention_mask = inputs['attention_mask'][i]
            
            # Select important vectors based on attention weights
            importance_scores = torch.norm(text_vectors, dim=-1)
            masked_scores = importance_scores * attention_mask
            
            # Top-k selection
            top_k = min(32, torch.sum(attention_mask).item())
            top_indices = torch.topk(masked_scores, k=top_k).indices
            
            selected_vectors = text_vectors[top_indices]
            multi_vector_embeddings.append(selected_vectors)
            
    return multi_vector_embeddings
```

## ✅ Ưu điểm của BGE-M3

### 1. **Unified Architecture**
- **Single Model**: Thay vì multiple models cho different functionalities
- **Consistent Interface**: Same API cho dense, sparse, multi-vector
- **Resource Efficiency**: One model deployment thay vì multiple

### 2. **State-of-the-Art Performance**
```
📊 Performance Comparison:
Model               | MTEB Avg | Multilingual | Vietnamese
--------------------|----------|--------------|----------
BGE-M3             | 70.46%   | ✅ Excellent | 68.2%
e5-mistral-7b-instruct | 69.00% | ❌ Limited | ~45%
multilingual-e5-large | 65.79% | ✅ Good | 62.1%
```

### 3. **Flexibility và Adaptability**
- **Multi-Modal Retrieval**: Dense + Sparse + Multi-Vector
- **Domain Adaptation**: Fine-tuning cho specific domains
- **Language Flexibility**: Cross-lingual capabilities

### 4. **Production Ready**
- **Optimized Inference**: Efficient serving implementations
- **Scalable**: Supports batch processing
- **Memory Efficient**: Reasonable model size (2.3GB)

## ❌ Nhược điểm của BGE-M3

### 1. **Complexity Issues**

#### **Training Complexity**:
- **Multi-Objective Optimization**: Balancing multiple loss functions
- **Hyperparameter Tuning**: Complex hyperparameter space
- **Computational Cost**: Requires significant compute resources

```python
# Complex loss function
total_loss = (
    α * dense_loss + 
    β * sparse_loss + 
    γ * multi_vector_loss +
    δ * distillation_loss_dense_sparse +
    ε * distillation_loss_dense_multi +
    ζ * distillation_loss_sparse_multi
)
# Tuning α, β, γ, δ, ε, ζ is challenging
```

#### **Deployment Complexity**:
- **Multiple Inference Modes**: Need to support 3 different output types
- **Storage Overhead**: Multi-vector approach requires more storage
- **API Complexity**: More complex serving interface

### 2. **Resource Requirements**

#### **Memory Usage**:
```
💾 Memory Requirements:
- Model Parameters: 560M (XLM-RoBERTa backbone)
- Dense Embeddings: 1024 dims per text
- Sparse Embeddings: vocab_size dims per text (~250K)
- Multi-Vector: 32 × 1024 dims per text
```

#### **Computational Overhead**:
- **Multi-Vector Similarity**: More expensive than simple cosine similarity
- **Sparse Vector Processing**: Additional computation for sparse weights
- **Cross-Functionality**: Higher inference cost when using all modes

### 3. **Fine-tuning Challenges**

#### **Knowledge Distillation Issues**:
- **Balancing Act**: Hard to balance between different functionalities
- **Catastrophic Forgetting**: Risk of losing general capabilities during specialization
- **Domain Adaptation**: Difficult to adapt all functionalities simultaneously

```python
# Example fine-tuning challenge
def fine_tune_domain_specific(model, domain_data):
    # Risk: Dense performance improves, sparse performance degrades
    for batch in domain_data:
        dense_loss = compute_dense_loss(batch)  # Improves
        sparse_loss = compute_sparse_loss(batch)  # May degrade
        
        # Challenging to maintain balance
        total_loss = dense_loss + λ * sparse_loss  # How to set λ?
```

### 4. **Limited Specialization**

#### **Jack of All Trades Problem**:
- **General Purpose**: May not excel in highly specialized domains
- **Trade-offs**: Performance compromises across functionalities
- **Domain Expertise**: May lack deep domain-specific optimizations

### 5. **Interpretability Issues**

#### **Black Box Nature**:
- **Multi-Vector Selection**: Why certain vectors are selected?
- **Sparse Weight Assignment**: How weights are assigned to terms?
- **Cross-Functionality Interactions**: Complex internal dynamics

## 🎓 Training Process Chi Tiết

### Phase 1: Multilingual Foundation

```python
# Stage 1: Massive multilingual pre-training
def stage1_multilingual_training():
    """
    Foundation training trên diverse multilingual corpus
    Goal: Establish strong multilingual representations
    """
    
    # Data preparation
    multilingual_corpus = load_multilingual_data([
        'wikipedia_100_languages',
        'common_crawl_multilingual', 
        'news_corpora_multilingual'
    ])
    
    # Training loop
    for epoch in range(foundation_epochs):
        for batch in multilingual_corpus:
            # Simple contrastive learning
            embeddings = model.encode_dense(batch['texts'])
            
            # In-batch negatives
            labels = create_contrastive_labels(batch['similarities'])
            loss = contrastive_loss(embeddings, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
        # Evaluation on multiple languages
        evaluate_multilingual_performance(model)
```

### Phase 2: Multi-Functionality Integration

```python
# Stage 2: Joint functionality training
def stage2_multifunctionality_training():
    """
    Joint training để integrate dense, sparse, multi-vector
    Goal: Learn unified representations across functionalities
    """
    
    retrieval_data = load_retrieval_datasets([
        'ms_marco', 'natural_questions', 'mmarco', 'mr_tydi'
    ])
    
    for epoch in range(integration_epochs):
        for batch in retrieval_data:
            # Multi-functionality forward pass
            dense_emb = model.encode_dense(batch['queries'], batch['docs'])
            sparse_emb = model.encode_sparse(batch['queries'], batch['docs'])
            multi_emb = model.encode_multi_vector(batch['queries'], batch['docs'])
            
            # Individual functionality losses
            loss_dense = ranking_loss(dense_emb, batch['labels'])
            loss_sparse = sparse_ranking_loss(sparse_emb, batch['labels'])
            loss_multi = multi_vector_loss(multi_emb, batch['labels'])
            
            # Self-knowledge distillation
            sim_dense = compute_similarity(dense_emb)
            sim_sparse = compute_similarity(sparse_emb)  
            sim_multi = compute_similarity(multi_emb)
            
            distill_loss = (
                kl_divergence(sim_dense, sim_sparse) +
                kl_divergence(sim_dense, sim_multi) +
                kl_divergence(sim_sparse, sim_multi)
            )
            
            # Combined loss
            total_loss = (
                loss_dense + loss_sparse + loss_multi + 
                λ * distill_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
        # Multi-functionality evaluation
        evaluate_all_functionalities(model)
```

### Phase 3: Domain Specialization

```python
# Stage 3: Domain-specific fine-tuning
def stage3_domain_specialization():
    """
    Fine-tune trên specific domains while maintaining general capabilities
    Goal: Specialize cho target domains without catastrophic forgetting
    """
    
    domain_datasets = {
        'legal': load_legal_corpus(),
        'medical': load_medical_corpus(), 
        'scientific': load_scientific_corpus()
    }
    
    # Gradual domain adaptation
    for domain, dataset in domain_datasets.items():
        print(f"Specializing for {domain} domain...")
        
        for epoch in range(specialization_epochs):
            for batch in dataset:
                # Domain-specific training
                domain_loss = compute_domain_loss(model, batch)
                
                # General capability preservation
                general_batch = sample_general_data()
                general_loss = compute_general_loss(model, general_batch)
                
                # Regularization để prevent forgetting
                regularization_loss = compute_regularization_loss(
                    model, previous_model_weights
                )
                
                total_loss = (
                    domain_loss + 
                    α * general_loss + 
                    β * regularization_loss
                )
                
                total_loss.backward()
                optimizer.step()
                
            # Monitor both domain and general performance
            domain_performance = evaluate_domain_specific(model, domain)
            general_performance = evaluate_general_capabilities(model)
            
            # Early stopping if general performance degrades
            if general_performance < threshold:
                print(f"Stopping {domain} specialization to prevent forgetting")
                break
```

### Advanced Training Techniques

#### **Hard Negative Mining**:

```python
def hard_negative_mining(model, queries, documents):
    """
    Dynamically mine hard negatives during training
    Goal: Improve model's discrimination ability
    """
    
    with torch.no_grad():
        # Generate embeddings
        query_embs = model.encode_dense(queries)
        doc_embs = model.encode_dense(documents)
        
        # Compute similarity matrix
        similarities = torch.mm(query_embs, doc_embs.t())
        
        # Find hard negatives (high similarity but wrong labels)
        hard_negatives = []
        for i, query in enumerate(queries):
            # Get documents with high similarity but low relevance
            query_sims = similarities[i]
            
            # Sort by similarity
            sorted_indices = torch.argsort(query_sims, descending=True)
            
            # Select hard negatives
            for j in sorted_indices:
                if j not in positive_docs[i] and query_sims[j] > threshold:
                    hard_negatives.append((i, j))
                    
    return hard_negatives

# Training với hard negatives
for batch in training_data:
    # Mine hard negatives for current batch
    hard_negs = hard_negative_mining(model, batch['queries'], batch['docs'])
    
    # Add hard negatives to training batch
    augmented_batch = add_hard_negatives(batch, hard_negs)
    
    # Train with augmented data
    loss = compute_loss(model, augmented_batch)
    loss.backward()
```

#### **Multi-Task Learning Schedule**:

```python
def adaptive_loss_weighting(epoch, performance_metrics):
    """
    Dynamically adjust loss weights based on performance
    Goal: Balance learning across functionalities
    """
    
    # Performance tracking
    dense_perf = performance_metrics['dense']
    sparse_perf = performance_metrics['sparse'] 
    multi_perf = performance_metrics['multi_vector']
    
    # Adaptive weighting
    if dense_perf < target_dense:
        weight_dense = 1.2
    else:
        weight_dense = 0.8
        
    if sparse_perf < target_sparse:
        weight_sparse = 1.2
    else:
        weight_sparse = 0.8
        
    if multi_perf < target_multi:
        weight_multi = 1.2
    else:
        weight_multi = 0.8
        
    return {
        'dense': weight_dense,
        'sparse': weight_sparse, 
        'multi': weight_multi
    }

# Training với adaptive weights
for epoch in range(num_epochs):
    # Evaluate current performance
    current_performance = evaluate_model(model)
    
    # Update loss weights
    weights = adaptive_loss_weighting(epoch, current_performance)
    
    for batch in training_data:
        # Compute losses
        loss_dense = compute_dense_loss(model, batch)
        loss_sparse = compute_sparse_loss(model, batch)
        loss_multi = compute_multi_loss(model, batch)
        
        # Weighted combination
        total_loss = (
            weights['dense'] * loss_dense +
            weights['sparse'] * loss_sparse + 
            weights['multi'] * loss_multi
        )
        
        total_loss.backward()
        optimizer.step()
```

## 🔬 Research Impact và Future Directions

### Current Impact

1. **Benchmark Performance**: SOTA results trên multiple multilingual benchmarks
2. **Industry Adoption**: Wide adoption trong production systems
3. **Research Foundation**: Basis cho subsequent embedding research
4. **Open Source**: Availability promotes broader research

### Future Research Directions

1. **Efficiency Improvements**:
   - Model compression techniques
   - Quantization strategies
   - Distillation to smaller models

2. **Domain Specialization**:
   - Better domain adaptation methods
   - Few-shot learning capabilities
   - Transfer learning improvements

3. **Architectural Innovations**:
   - Integration with newer transformer architectures
   - Attention mechanism improvements
   - Memory-efficient designs

---

*Tài liệu này cung cấp comprehensive understanding về BGE-M3 model architecture, training methodology, và practical implications cho Vietnamese Legal RAG applications.*