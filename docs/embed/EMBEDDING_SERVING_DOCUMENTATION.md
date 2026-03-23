# Tài liệu Embedding Serving - Vietnamese Legal Chatbot RAG System

## 📋 Tổng quan

Module Embedding Serving của Vietnamese Legal Chatbot RAG System cung cấp **RESTful API** để serve embedding model **BGE-M3** cho việc tạo vector representations của text tiếng Việt trong domain pháp luật. Hệ thống được thiết kế để chạy hiệu quả trên **GPU servers** với performance cao và cost-effective.

## 🎯 Vấn đề cần giải quyết

### Bài toán Semantic Search trong RAG
Embedding Serving giải quyết việc **chuyển đổi text thành vector** để phục vụ:

1. **🔍 Semantic Search**: Tìm kiếm documents liên quan dựa trên ý nghĩa, không chỉ keywords
2. **📊 Similarity Computation**: Tính toán độ tương tự giữa câu hỏi user và legal documents
3. **⚡ Real-time Inference**: Serving embedding với latency thấp cho chatbot

## 🤖 BGE-M3 Model - Lựa chọn Embedding Model

### Tại sao chọn BGE-M3?

**BGE-M3** (BAAI General Embedding - Multilingual, Multi-Granularity, Multi-Functionality) là state-of-the-art embedding model được phát triển bởi Beijing Academy of Artificial Intelligence.

#### 📄 **Paper và Research Background**

**Research Paper**: [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216)

**Key Innovation**: Self-knowledge distillation technique để tạo ra unified embedding space cho multiple languages và functionalities.

#### 🎯 **Ưu điểm vượt trội của BGE-M3**

**1. Multi-Lingual Excellence**
- **Cross-lingual retrieval** - search tiếng Việt trong corpus đa ngôn ngữ

**2. Multi-Functionality**
- **Dense Retrieval**: Traditional semantic similarity
- **Sparse Retrieval**: Keyword-based matching (tương tự BM25)

**5. Efficiency**
- **Model size**: 2.3GB (compact cho production)
- **Embedding dim**: 1024 (optimal cho speed/quality balance)

#### 🔬 **Technical Architecture**

**Base Architecture**: 
- **Backbone**: XLM-RoBERTa-large (560M parameters)
- **Self-Knowledge Distillation**: Novel training approach
- **Multi-task Learning**: Joint training cho dense + sparse + multi-vector


**Input Processing**:
- **Max sequence length**: 8192 tokens (excellent cho legal documents)

## 🏗️ Kiến trúc Serving System

### Framework và Technology Stack

#### 🌐 **Tại sao chọn Flask Framework?**

**Flask** được chọn làm serving framework thay vì FastAPI hay alternatives:

**Ưu điểm của Flask**:
1. **Simplicity**: Minimal boilerplate, easy debugging
2. **Lightweight**: Low memory footprint (quan trọng cho CPU serving)
3. **Mature Ecosystem**: Extensive libraries và community support
4. **Production Proven**: Được sử dụng rộng rãi trong production
5. **Threading Support**: Built-in multi-threading cho concurrent requests

#### 🧠 **Model Loading và Optimization**

### API Design và Endpoints

#### 🔧 **Core API Endpoints**

**1. Health Check Endpoint**
```python
@app.route("/health", methods=["GET"])
```

**Chức năng**: Service discovery, health monitoring, load balancer integration.

**2. Embedding Generation Endpoint**
```python
@app.route("/embed", methods=["POST"])
def embed():
```

**3. Similarity Computation Endpoint**
```python
@app.route("/similarity", methods=["POST"])
```
#### ⚡ **Performance Optimizations**

**Batch Size Management**:
```python
max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "32"))

# Optimal batch sizes cho different CPU configurations:
# 4-core CPU: batch_size = 16
# 8-core CPU: batch_size = 32  
# 16-core CPU: batch_size = 64
```
---

*Tài liệu này mô tả comprehensive architecture và implementation của Embedding Serving Module, sử dụng BGE-M3 model với Flask framework để cung cấp high-performance, cost-effective embedding API cho Vietnamese Legal RAG System.*