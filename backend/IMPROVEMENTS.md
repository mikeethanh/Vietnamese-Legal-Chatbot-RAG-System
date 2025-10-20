# Backend RAG System Improvements

## Tổng Quan Cải Tiến

Hệ thống RAG cho chatbot tư vấn pháp luật Việt Nam đã được cải tiến với các kỹ thuật hiện đại để đạt hiệu suất cao hơn.

---

## 1. Semantic Chunking (Phân Đoạn Ngữ Nghĩa)

### File: `splitter.py`

**Cải tiến:**
- Thay thế `TokenTextSplitter` bằng `SemanticSplitterNodeParser`
- Sử dụng embeddings để xác định ranh giới ngữ nghĩa tự nhiên
- Tránh cắt đứt câu hoặc đoạn văn ở vị trí không phù hợp

**Lợi ích:**
- Giữ nguyên ngữ cảnh pháp lý quan trọng trong mỗi chunk
- Cải thiện độ chính xác của retrieval
- Phù hợp với văn bản pháp luật có cấu trúc phức tạp

**Cách sử dụng:**
```python
from splitter import split_document

# Semantic splitting (mặc định, khuyến nghị)
nodes = split_document(text, metadata={"title": "Luật Doanh Nghiệp"}, use_semantic=True)

# Token-based splitting (fallback)
nodes = split_document(text, metadata={"title": "..."}, use_semantic=False)
```

**Tham số:**
- `buffer_size=1`: Nhóm câu khi đánh giá similarity
- `breakpoint_percentile_threshold=95`: Ngưỡng để tạo điểm cắt (95% = cắt ít, giữ context dài)

---

## 2. Multi-Query Retrieval (Truy Xuất Đa Truy Vấn)

### File: `query_rewriter.py`

**Chức năng:**

### 2.1. `rewrite_query_to_multi_queries()`
Tạo 3 câu hỏi đa dạng từ 1 câu hỏi gốc để tăng phủ sóng retrieval.

**Ví dụ:**
```python
Input: "Thủ tục ly hôn như thế nào?"

Output:
1. "Quy trình giải quyết ly hôn theo pháp luật Việt Nam"
2. "Các bước tiến hành thủ tục chấm dứt hôn nhân"
3. "Hồ sơ và trình tự ly hôn được quy định ra sao"
```

**Lợi ích:**
- Bắt được nhiều góc độ của câu hỏi
- Tăng recall (số lượng tài liệu liên quan được tìm thấy)
- Xử lý tốt các câu hỏi mơ hồ

### 2.2. `rewrite_query_with_context()`
Viết lại câu hỏi có ngữ cảnh thành câu độc lập.

**Ví dụ:**
```python
History: "User: Thủ tục ly hôn như thế nào?"
Query: "Còn chi phí thì sao?"

Output: "Chi phí thủ tục ly hôn theo pháp luật Việt Nam là bao nhiêu?"
```

---

## 3. Improved Follow-Up Question Handling

### File: `brain.py` - `detect_user_intent()`

**Cải tiến:**
- Phát hiện các đại từ chỉ định (đó, này, kia, thế, vậy)
- Thay thế đại từ bằng danh từ cụ thể từ lịch sử
- Prompt tiếng Việt tối ưu cho ngữ cảnh pháp luật

**Ví dụ:**
```python
History:
User: "Thủ tục thành lập công ty TNHH như thế nào?"
Assistant: "Thủ tục thành lập công ty TNHH gồm..."

Current Query: "Vậy chi phí là bao nhiêu?"

Rephrased: "Chi phí thủ tục thành lập công ty TNHH theo quy định là bao nhiêu?"
```

**Kỹ thuật:**
- Kiểm tra follow-up indicators
- Trích xuất entities từ lịch sử
- Bổ sung ngữ cảnh vào câu hỏi hiện tại

---

## 4. Enhanced Routing System

### File: `brain.py` - `detect_route()`

**Các Route:**

1. **`legal_rag`** - Câu hỏi về pháp luật Việt Nam
   - Thủ tục pháp lý
   - Quy định, luật, nghị định
   - Quyền và nghĩa vụ
   - Xử lý vi phạm

2. **`web_search`** - Thông tin thời sự
   - Luật mới ban hành
   - Vụ án đang diễn ra
   - Tin tức pháp luật

3. **`general_chat`** - Trò chuyện thông thường
   - Chào hỏi
   - Hỏi về khả năng chatbot
   - Câu hỏi ngoài chủ đề

**Prompt Engineering:**
- Hướng dẫn rõ ràng cho từng route
- Ví dụ cụ thể cho mỗi trường hợp
- Validation và fallback

---

## 5. Integrated RAG Pipeline

### File: `tasks.py`

**Luồng xử lý:**

```
User Query
    ↓
1. Follow-up Question Handling
   (detect_user_intent)
    ↓
2. Query Rewriting
   (rewrite_query_to_multi_queries)
   → 3 query variations
    ↓
3. Multi-Query Retrieval
   (retrieve_with_multi_query)
   → 15-20 documents (3 queries × 5 docs)
    ↓
4. Reranking
   (rerank_documents with Cohere)
   → Top 5 most relevant
    ↓
5. Enhanced Prompting
   (Vietnamese legal context)
    ↓
6. Answer Generation
   (OpenAI GPT-4)
    ↓
Response
```

**Các hàm chính:**

### `bot_rag_answer_message()`
Pipeline RAG hoàn chỉnh với multi-query và reranking.

### `bot_route_answer_message()`
Routing thông minh đến handler phù hợp:
- `legal_rag`: Sử dụng RAG system
- `web_search`: Google Custom Search
- `general_chat`: Simple conversation

### `llm_handle_message()`
Entry point chính với:
- Conversation management
- Automatic routing
- Response summarization

---

## 6. System Prompt Improvements

**Legal RAG System Prompt:**
```
Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam. Nhiệm vụ của bạn là:
1. Trả lời câu hỏi dựa trên các tài liệu pháp luật được cung cấp
2. Trích dẫn chính xác các điều khoản, khoản, điểm
3. Giải thích rõ ràng, dễ hiểu cho người không chuyên
4. Nếu thông tin không đủ, hãy nói rõ điều đó
5. Luôn đưa ra câu trả lời có căn cứ pháp lý
```

**Lợi ích:**
- Hướng dẫn rõ ràng về cách trả lời
- Yêu cầu trích dẫn nguồn
- Tránh hallucination

---

## Configuration & Environment

### Required Environment Variables

```bash
# OpenAI API
OPENAI_API_KEY=sk-...

# Cohere for Reranking
COHERE_API_KEY=...

# Google Custom Search (for web_search route)
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...

# Database
MYSQL_USER=root
MYSQL_PASSWORD=...
MYSQL_HOST=mariadb-db
MYSQL_PORT=3306

# Redis/Celery
CELERY_BROKER_URL=redis://valkey-db:6379
CELERY_RESULT_BACKEND=redis://valkey-db:6379

# Vector Database
QDRANT_URL=http://qdrant-db:6333
```

---

## Performance Metrics

### Improvements Expected:

1. **Retrieval Quality:**
   - ↑ Recall: +15-25% (multi-query)
   - ↑ Precision: +10-20% (reranking)
   - ↑ Context preservation (semantic chunking)

2. **User Experience:**
   - Better follow-up handling
   - More accurate routing
   - Contextually aware responses

3. **Response Quality:**
   - More relevant citations
   - Better Vietnamese legal terminology
   - Fewer hallucinations

---

## Usage Examples

### Example 1: Simple Legal Query
```python
User: "Thủ tục đăng ký kết hôn như thế nào?"

System:
1. Route → legal_rag
2. Generate 3 queries
3. Retrieve & rerank documents
4. Generate answer with legal citations
```

### Example 2: Follow-up Question
```python
User: "Thủ tục ly hôn như thế nào?"
Bot: "Thủ tục ly hôn gồm..."

User: "Vậy thời gian giải quyết là bao lâu?"

System:
1. Detect follow-up
2. Rephrase: "Thời gian giải quyết thủ tục ly hôn là bao lâu?"
3. Continue RAG pipeline
```

### Example 3: Web Search
```python
User: "Luật giao thông mới nhất năm 2025 có gì thay đổi?"

System:
1. Route → web_search
2. Google search with rephrased query
3. Synthesize results
4. Return answer with sources
```

---

## Monitoring & Debugging

### Log Points

```python
# Check routing decision
logger.info(f"Detected route: {route}")

# View query variations
logger.info(f"Query variations: {queries}")

# Monitor retrieval
logger.info(f"Retrieved {len(docs)} documents")

# Track reranking
logger.info(f"Top {len(ranked_docs)} after rerank")
```

### Debug Mode

Set logging level to DEBUG for detailed information:
```python
logging.basicConfig(level=logging.DEBUG)
```

---

## Best Practices

### 1. Chunking
- Sử dụng semantic splitting cho văn bản pháp luật
- Đặt `breakpoint_percentile_threshold=95` để giữ context dài
- Test với các loại văn bản khác nhau

### 2. Query Rewriting
- Luôn generate 3 queries (balance giữa diversity và efficiency)
- Review generated queries định kỳ để cải thiện prompt
- Cache frequent queries để tiết kiệm API calls

### 3. Reranking
- Sử dụng Cohere `rerank-multilingual-v3.0` cho tiếng Việt
- Top_n=5 là optimal cho hầu hết trường hợp
- Monitor relevance scores để điều chỉnh threshold

### 4. Routing
- Default route: `legal_rag` (an toàn cho legal chatbot)
- Review routing decisions thường xuyên
- Thêm custom routes khi cần thiết

---

## Troubleshooting

### Issue 1: Low Retrieval Quality
**Solution:**
- Kiểm tra embedding model
- Tăng số queries (num_queries=5)
- Điều chỉnh `breakpoint_percentile_threshold`

### Issue 2: Incorrect Routing
**Solution:**
- Review routing prompt
- Thêm examples cụ thể
- Implement fallback logic

### Issue 3: Slow Performance
**Solution:**
- Cache embeddings
- Reduce num_queries
- Use async operations
- Optimize reranking threshold

---

## Future Improvements

1. **Hybrid Search:** Kết hợp semantic + keyword search
2. **Query Classification:** ML model thay vì LLM routing
3. **Caching Layer:** Redis cache cho frequent queries
4. **Fine-tuned Embeddings:** Train on Vietnamese legal corpus
5. **Feedback Loop:** Thu thập user feedback để cải thiện

---

## References

- LlamaIndex Documentation: https://docs.llamaindex.ai/
- Cohere Rerank: https://docs.cohere.com/docs/reranking
- Semantic Chunking: https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/
- Multi-Query Retrieval: https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83

---

## Contact & Support

For questions or issues, please refer to the project documentation or contact the development team.
