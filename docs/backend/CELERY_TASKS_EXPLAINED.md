# Giải Thích Chi Tiết File tasks.py - Celery Tasks

## Tổng Quan
File `tasks.py` chứa các **background tasks** (tác vụ chạy nền) sử dụng **Celery** để xử lý các công việc tốn thời gian như:
- Tương tác với LLM (Large Language Model)
- Tìm kiếm và xếp hạng tài liệu
- Lập chỉ mục (indexing) documents

---

## 1. @shared_task() - Decorator Của Celery

### Celery là gì?
**Celery** là một hệ thống **distributed task queue** (hàng đợi tác vụ phân tán) cho Python. Nó cho phép:
- Chạy các tác vụ **bất đồng bộ** (asynchronous) trong background
- Phân tán công việc cho nhiều **worker processes**
- Xử lý hàng triệu tasks mà không làm chậm server chính

### @shared_task() Decorator

```python
@shared_task()
def llm_handle_message(bot_id, user_id, question):
    # Code xử lý...
    return response
```

---

## 2. Chi Tiết Từng Hàm

### 2.1. `follow_up_question(history, question)`

**Mục đích**: Xử lý câu hỏi tiếp theo (follow-up) bằng cách diễn giải lại với ngữ cảnh


#### Giải thích:
- **Input**:
  - `history`: Lịch sử hội thoại trước đó
  - `question`: Câu hỏi hiện tại của user
  
- **Process**: 
  - Gọi `detect_user_intent()` để hiểu câu hỏi trong ngữ cảnh
  - Log kết quả sau khi diễn giải
  
- **Output**: Câu hỏi độc lập (standalone question)

**Tại sao cần?**: LLM và vector search cần câu hỏi đầy đủ, không phụ thuộc ngữ cảnh.

---

### 2.2. `retrieve_with_hybrid_search(queries, top_k=5)`

**Mục đích**: Tìm kiếm tài liệu sử dụng **hybrid search** (kết hợp semantic + keyword)

- **Process**:
  1. Duyệt qua từng query variation
  2. Gọi `hybrid_search()` - kết hợp:
     - **Vector search** (semantic): Tìm theo nghĩa
     - **BM25** (keyword): Tìm theo từ khóa chính xác
  3. **Deduplicate**: Loại bỏ tài liệu trùng lặp bằng hash content
  4. Gắn metadata `retrieval_query` để biết doc nào từ query nào
  
- **Output**: List các documents unique với hybrid scores

#### Ví dụ:
```python
queries = [
    "Phạt nồng độ cồn khi lái xe",
    "Mức phạt vi phạm nồng độ cồn người điều khiển phương tiện",
    "Xử phạt uống rượu bia lái xe"
]

docs = retrieve_with_hybrid_search(queries, top_k=3)
# Kết quả: 7 documents unique (không trùng lặp) từ 3 queries
```

**Ưu điểm Hybrid Search**:
- Semantic search: Bắt được đồng nghĩa ("phạt" = "xử phạt")
- Keyword search: Bắt được từ chính xác ("nồng độ cồn")
- Kết hợp → Coverage cao hơn

---

### 2.4. `bot_rag_answer_message(history, question)` - @shared_task()

**Mục đích**: Pipeline RAG (Retrieval-Augmented Generation) hoàn chỉnh để trả lời câu hỏi pháp luật

```python
@shared_task()
def bot_rag_answer_message(history, question):
    """
    Enhanced RAG pipeline with:
    1. Follow-up question handling
    2. Multi-query retrieval
    3. Reranking
    4. Improved prompting for Vietnamese legal context
    """
```

**Tại sao là @shared_task()?**: 
- LLM generation mất 3-10 giây
- Không muốn block FastAPI request
- Chạy trên Celery worker riêng

---

---

### 2.6. `get_summarized_response(response)`

**Mục đích**: Tóm tắt câu trả lời dài thành version ngắn gọn


#### Khi nào dùng?
- Lưu vào database (tiết kiệm storage)
- Hiển thị preview trong lịch sử chat
- Response quá dài (> 1000 từ)

---

### 2.7. `bot_route_answer_message(history, question)` - @shared_task()

**Mục đích**: **Định tuyến** (routing) câu hỏi đến handler phù hợp dựa trên intent


#### Chi tiết 4 routes:

##### **Route 1: legal_rag**
**Khi nào**: Câu hỏi về luật pháp có trong knowledge base

```python
Question: "Phạt vượt đèn đỏ bao nhiêu?"
→ Route: legal_rag
→ Handler: bot_rag_answer_message()
→ Flow: Retrieve docs từ Qdrant → Rerank → LLM generate
```

##### **Route 2: agent_tools**
**Khi nào**: Cần tính toán, kiểm tra logic, dùng tools

```python
Question: "Tôi 17 tuổi 11 tháng, có được đăng ký kết hôn không?"
→ Route: agent_tools
→ Handler: ai_agent_handle()
→ Tools: check_legal_age_for_marriage(age=17, months=11)
→ Output: "Không, tuổi kết hôn tối thiểu là 18 (nữ) và 20 (nam) theo Luật Hôn nhân..."
```

**Ví dụ tools khác**:
- `calculate_inheritance(total_assets, heirs)` - Tính chia thừa kế
- `calculate_fine(violation_type, vehicle_type)` - Tính tiền phạt
- `validate_contract_terms(terms)` - Kiểm tra hợp đồng

##### **Route 3: web_search**
**Khi nào**: Hỏi về thông tin mới, chưa có trong database

```python
Question: "Luật giao thông mới 2025 có gì thay đổi?"
→ Route: web_search
→ Handler: tavily_search_legal()
→ Flow: Search Google/Bing → Lấy top 5 results → LLM tổng hợp
```

**Tại sao cần**: Knowledge base có thể lỗi thời, web search bổ sung thông tin mới.

##### **Route 4: general_chat**
**Khi nào**: Chào hỏi, hỏi về hệ thống, không liên quan pháp luật

```python
Question: "Bạn có thể giúp gì cho tôi?"
→ Route: general_chat
→ Handler: vietnamese_llm_chat_complete() với general prompt
→ Output: "Tôi là trợ lý AI tư vấn pháp luật. Tôi có thể giúp bạn:
           1. Tìm hiểu các quy định pháp luật
           2. Tính toán phí phạt, chia thừa kế
           3. Hướng dẫn thủ tục pháp lý..."
```

#### Tại sao cần routing?
- ✅ **Tối ưu tài nguyên**: Không dùng RAG cho câu "Xin chào"
- ✅ **Độ chính xác cao**: Dùng đúng tool cho đúng task
- ✅ **Trải nghiệm tốt**: Response nhanh hơn, chính xác hơn

---

### 2.8. `llm_handle_message(bot_id, user_id, question)` - @shared_task()

**Mục đích**: Handler chính orchestrate toàn bộ luồng xử lý message

```python
@shared_task()
def llm_handle_message(bot_id, user_id, question):
    """
    Main message handler with intelligent routing.

    Flow:
    1. Save user message to conversation history
    2. Load conversation context
    3. Route to appropriate handler (RAG, web search, or general chat)
    4. Generate and save response
    """
```

#### Luồng chi tiết:

##### **Bước 1: Lưu message của user vào database**
```python
conversation_id = update_chat_conversation(bot_id, user_id, question, True)
```

##### **Bước 2: Load lịch sử hội thoại**
```python
messages = get_conversation_messages(conversation_id)

history = messages[:-1]  # Bỏ message cuối (vừa gửi) để làm context
```

##### **Bước 3: Route và xử lý**
```python
response = bot_route_answer_message(history, question)
# Tự động chọn: legal_rag / agent_tools / web_search / general_chat
```

##### **Bước 4: Summarize (optional)**
```python
try:
    summarized_response = get_summarized_response(response)
except Exception as e:
    summarized_response = response  # Fallback nếu summarize lỗi
```

##### **Bước 5: Lưu response vào database**
```python
update_chat_conversation(bot_id, user_id, summarized_response, False)
# False = is_assistant_message
```

##### **Bước 6: Trả về kết quả**
```python
return {"role": "assistant", "content": response}  # Full response (không summarized)
```

#### Khi nào task này được gọi?

**Từ FastAPI endpoint** (trong `app.py`):
```python
@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    if data.sync_request:
        # Gọi đồng bộ (KHÔNG khuyến khích)
        response = llm_handle_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        # Gọi bất đồng bộ (KHUYẾN KHÍCH)
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}
```


