# Giải Thích Chi Tiết Hybrid Search - Tìm Kiếm Lai

## Tổng Quan
File `search.py` triển khai **Hybrid Search** - kết hợp 2 phương pháp tìm kiếm để tăng độ chính xác:
1. **Vector Search** (Semantic) - Tìm theo nghĩa
2. **BM25 Search** (Keyword) - Tìm theo từ khóa

---

## 1. Hybrid Search Là Gì?

### Định nghĩa
**Hybrid Search** = Kết hợp nhiều phương pháp tìm kiếm khác nhau để tận dụng ưu điểm của từng loại.

### So sánh 3 loại tìm kiếm:

| Loại | Cách hoạt động | Ưu điểm | Nhược điểm | Ví dụ |
|------|---------------|---------|-----------|-------|
| **Keyword Search (BM25)** | Đếm tần suất từ khóa | Chính xác với từ chính xác | Bỏ lỡ đồng nghĩa | "phạt nồng độ cồn" → tìm đúng từ "phạt nồng độ cồn" |
| **Vector Search (Semantic)** | So sánh nghĩa bằng embeddings | Hiểu đồng nghĩa, ngữ cảnh | Có thể trả về kết quả không chính xác | "phạt uống rượu lái xe" → tìm "xử phạt nồng độ cồn" |
| **Hybrid Search** | Kết hợp cả 2 | Vừa chính xác vừa linh hoạt | Phức tạp hơn | Kết hợp kết quả từ cả 2 phương pháp |

### Ví dụ cụ thể:

#### Query: "Phạt vượt đèn đỏ bao nhiêu tiền?"

**BM25 tìm được**:
- ✅ Doc 1: "Mức phạt vượt đèn đỏ từ 4-6 triệu đồng" (chứa từ chính xác)
- ❌ Bỏ lỡ: "Vi phạm tín hiệu giao thông" (không có từ "đèn đỏ")

**Vector Search tìm được**:
- ✅ Doc 1: "Mức phạt vượt đèn đỏ..." (similar meaning)
- ✅ Doc 2: "Vi phạm tín hiệu giao thông đường bộ" (đồng nghĩa)
- ❌ Doc 3: "Phạt vi phạm tốc độ" (có từ "phạt" nhưng khác chủ đề)

**Hybrid Search kết quả**:
- 🏆 Doc 1: Score cao nhất (cả 2 phương pháp đều tìm thấy)
- ✅ Doc 2: Score trung bình (chỉ vector tìm thấy)
- ❌ Doc 3: Score thấp, bị loại

---

## 2. Kiến Trúc Hệ Thống

### 2.1. Components (Thành phần)

```python
# Global variables - Shared state
_docstore = None              # Lưu trữ documents cho BM25
_bm25_retriever = None        # BM25 search engine
_search_engine_initialized = False  # Trạng thái khởi tạo
```

#### Giải thích:

- **`_docstore`** (SimpleDocumentStore): 
  - Lưu trữ documents dưới dạng nodes
  - Cung cấp cho BM25Retriever để search
  - Như một "database in-memory" cho keyword search

- **`_bm25_retriever`** (BM25Retriever):
  - Thuật toán BM25 (Best Matching 25) - chuẩn công nghiệp cho keyword search
  - Tính điểm dựa trên TF-IDF (Term Frequency - Inverse Document Frequency)
  - Ưu tiên documents có từ khóa xuất hiện nhiều nhưng hiếm trong corpus

- **`_search_engine_initialized`** (bool):
  - Flag để kiểm tra hệ thống đã sẵn sàng chưa
  - Tránh gọi search khi chưa khởi tạo

### 2.2. Workflow Tổng Quan

```
┌─────────────────────────────────────────────────────────┐
│          HYBRID SEARCH WORKFLOW                         │
└─────────────────────────────────────────────────────────┘

1. Initialization (1 lần khi start)
   ┌──────────────┐
   │ Load Raw     │
   │ Documents    │ → [{"question": "...", "content": "..."}]
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Convert to   │
   │ LlamaIndex   │ → [Document(text="...", metadata={...})]
   │ Format       │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │ Split into   │
   │ Chunks       │ → [Node1, Node2, Node3, ...]
   │ (2048 tokens)│
   └──────┬───────┘
          ▼
   ┌──────────────────┬──────────────────┐
   │  Initialize      │  Initialize      │
   │  Docstore        │  BM25 Retriever  │
   └──────────────────┴──────────────────┘

2. Search Time (mỗi query)
   ┌──────────────┐
   │ User Query   │ → "Phạt vượt đèn đỏ?"
   └──────┬───────┘
          ▼
   ┌─────────────────────────────────────┐
   │        PARALLEL SEARCH              │
   ├──────────────────┬──────────────────┤
   │  BM25 Search     │  Vector Search   │
   │  (Keyword)       │  (Semantic)      │
   │                  │                  │
   │  - Tokenize      │  - Get embedding │
   │  - Match tokens  │  - Cosine sim    │
   │  - BM25 scoring  │  - Top K results │
   └────────┬─────────┴─────────┬────────┘
            ▼                   ▼
   ┌────────────────────────────────────┐
   │    COMBINE & SCORE RESULTS         │
   │                                    │
   │  - Deduplicate by content hash     │
   │  - Merge scores for overlaps       │
   │  - Calculate hybrid score          │
   │  - Sort by hybrid score            │
   └────────────────┬───────────────────┘
                    ▼
            ┌──────────────┐
            │ Top K Final  │
            │ Results      │
            └──────────────┘
```

---

## 3. Chi Tiết Từng Hàm

### 3.1. `initialize_search_index(documents)` - Khởi Tạo

**Mục đích**: Chuẩn bị BM25 search index từ documents

##### **Bước 2: Convert sang LlamaIndex Document format**
```python
llama_docs = []
for i, doc in enumerate(documents):
    text = f"{doc.get('question', '')} {doc.get('content', '')}"
    llama_doc = Document(
        text=text,
        metadata={
            "question": doc.get('question', ''),
            "content": doc.get('content', ''),
            "source": doc.get('source', 'unknown'),
            "doc_id": doc.get('doc_id', i)
        }
    )
    llama_docs.append(llama_doc)
```

**Tại sao combine question + content?**
- BM25 search cả question lẫn content để coverage tốt hơn
- Question thường chứa keywords quan trọng

##### **Bước 3: Split thành chunks (nodes)**
```python
splitter = SentenceSplitter(chunk_size=2048)
nodes = splitter.get_nodes_from_documents(llama_docs)
```

**Tại sao chunk_size=2048?**
- Documents pháp luật thường dài (nhiều điều khoản)
- 2048 tokens ≈ 1500-1800 từ tiếng Việt
- Đủ lớn để giữ ngữ cảnh, không quá lớn để search chính xác

##### **Bước 4: Khởi tạo Docstore**
```python
_docstore = SimpleDocumentStore()
_docstore.add_documents(nodes)
```

**SimpleDocumentStore** là gì?
- In-memory storage cho documents
- Cho phép BM25 retriever truy cập nhanh
- Lưu dưới dạng dict: `{node_id: node_object}`

##### **Bước 5: Khởi tạo BM25 Retriever**
```python
_bm25_retriever = BM25Retriever.from_defaults(
    docstore=_docstore,
    similarity_top_k=5,
)
```

**BM25Retriever Parameters**:
- `docstore`: Nguồn documents để search
- `similarity_top_k`: Trả về top 5 kết quả mặc định

### 3.2. `hybrid_search(query, limit=10)` - Hàm Tìm Kiếm Chính

**Mục đích**: Thực hiện hybrid search kết hợp BM25 + Vector


#### Luồng xử lý chi tiết:

##### **Bước 2: BM25 keyword search**
```python
bm25_results = _bm25_retriever.retrieve(query)
logger.info(f"🔍 BM25 search returned {len(bm25_results)} results")
```

**BM25 hoạt động như thế nào?**

```python
Query: "Phạt vượt đèn đỏ"

# Step 1: Tokenize
tokens = ["phạt", "vượt", "đèn", "đỏ"]

# Step 2: Tính TF-IDF cho mỗi document
Document 1: "Mức phạt vượt đèn đỏ từ 4-6 triệu"
- TF(phạt) = 2/10 = 0.2 (xuất hiện 2 lần trong 10 từ)
- IDF(phạt) = log(1000/500) = 0.3 (500/1000 docs có từ "phạt")
- TF-IDF(phạt) = 0.2 * 0.3 = 0.06

Document 2: "Quy định về đèn tín hiệu giao thông"
- Không có từ "phạt", "vượt" → Score thấp

# Step 3: BM25 scoring (cải tiến của TF-IDF)
# Công thức: score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * docLen/avgDocLen))
# k1=1.5, b=0.75 (hyperparameters)

Document 1 BM25 score: 8.5
Document 2 BM25 score: 2.1
```

**Kết quả BM25**:
```python
bm25_results = [
    NodeWithScore(
        node=Node(
            text="Mức phạt vượt đèn đỏ từ 4-6 triệu đồng",
            metadata={"question": "Phạt vượt đèn đỏ bao nhiêu?", ...}
        ),
        score=8.5
    ),
]
```

##### **Bước 3: Vector semantic search**
```python
vector = get_embedding(query)
vector_results = search_vector(DEFAULT_COLLECTION_NAME, vector, limit)
logger.info(f"🔍 Vector search returned {len(vector_results)} results")
```

**Vector search hoạt động thế nào?**

```python
Query: "Phạt vượt đèn đỏ"

# Step 1: Get embedding
embedding_model = "BAAI/bge-m3"  
query_vector = get_embedding(query)

# Step 2: Cosine similarity search in Qdrant
# So sánh query_vector với tất cả document vectors trong DB

Doc 1: "Mức phạt vượt đèn đỏ..." 
  → vector1 = [0.120, -0.450, 0.780, ...]
  → cosine_sim(query_vector, vector1) = 0.92 (Very similar!)

Doc 2: "Vi phạm tín hiệu giao thông" 
  → vector2 = [0.115, -0.440, 0.770, ...]
  → cosine_sim(query_vector, vector2) = 0.85 (Similar semantically)

Doc 3: "Phạt vi phạm tốc độ"
  → vector3 = [0.080, -0.200, 0.400, ...]
  → cosine_sim(query_vector, vector3) = 0.65 (Less similar)
```

**Kết quả Vector**:
```python
vector_results = [
    {
        "content": "Mức phạt vượt đèn đỏ từ 4-6 triệu đồng",
        "question": "Phạt vượt đèn đỏ bao nhiêu?",
        "similarity_score": 0.92,
        "source": "nghi_dinh_100"
    },
    {
        "content": "Vi phạm tín hiệu giao thông...",
        "question": "Xử phạt không chấp hành tín hiệu",
        "similarity_score": 0.85,
        "source": "luat_gtdb"
    }
]
```

##### **Bước 4: Combine và tính hybrid score**
```python
combined_results = combine_search_results(bm25_results, vector_results, query)
```

**Chi tiết trong hàm `combine_search_results()` - xem mục 3.3**

##### **Bước 5: Sort và limit results**


### 3.3. `combine_search_results(bm25_results, vector_results, query)` - Kết Hợp Kết Quả

**Mục đích**: Merge kết quả từ 2 nguồn và tính hybrid score

#### Luồng xử lý chi tiết:

##### **Bước 1: Convert BM25 results sang dict format**


##### **Bước 2: Convert Vector results sang dict format**


##### **Bước 3: Merge results**
```python
all_docs = {}
overlap_count = 0

# Add BM25 results
for content_hash, doc in bm25_docs.items():
    all_docs[content_hash] = doc

# Add vector results and merge if overlap
for content_hash, doc in vector_docs.items():
    if content_hash in all_docs:
        # Found by both methods - MERGE!
        all_docs[content_hash]["vector_score"] = doc["vector_score"]
        all_docs[content_hash]["search_method"] = "hybrid"
        overlap_count += 1
    else:
        # Only found by vector
        all_docs[content_hash] = doc
```


##### **Bước 4: Tính hybrid score**

#### Giải thích công thức scoring:

| Search Method | Formula | Reasoning |
|---------------|---------|-----------|
| **Hybrid** (cả 2 tìm thấy) | `0.5*BM25 + 0.5*Vector + 0.1` | Kết quả tốt nhất, thưởng +0.1 bonus |
| **BM25 only** | `0.6 * BM25` | Giảm 40% vì chỉ 1 phương pháp tìm thấy |
| **Vector only** | `0.6 * Vector` | Giảm 40% vì chỉ 1 phương pháp tìm thấy |

**Tại sao thiết kế này?**
- ✅ **Hybrid results ưu tiên cao**: Nếu cả 2 phương pháp đều tìm thấy → rất relevant
- ✅ **Balance**: Weight 50-50 giữa BM25 và Vector
- ✅ **Penalty cho single-method**: Giảm score nếu chỉ 1 phương pháp tìm thấy

#### Ví dụ tính toán:

```python
Doc A (Hybrid):
- BM25 score: 8.5
- Vector score: 0.92
- Hybrid score = (8.5 * 0.5) + (0.92 * 0.5) + 0.1
                = 4.25 + 0.46 + 0.1
                = 4.81 ✅ HIGHEST

Doc B (BM25 only):
- BM25 score: 7.0
- Vector score: 0
- Hybrid score = 7.0 * 0.6
                = 4.2

Doc C (Vector only):
- BM25 score: 0
- Vector score: 0.85
- Hybrid score = 0.85 * 0.6
                = 0.51

# Ranking: Doc A > Doc B > Doc C
```

##### **Bước 5: Sort và log top results**
```python
sorted_docs = sorted(all_docs.values(), key=lambda x: x.get("hybrid_score", 0), reverse=True)

logger.info(f"🏆 Top 3 combined results:")
for i, doc in enumerate(sorted_docs[:3], 1):
    logger.info(f"   {i}. {doc['question'][:50]}... (Score: {doc['hybrid_score']:.3f}, Method: {doc['search_method']})")
```

---
