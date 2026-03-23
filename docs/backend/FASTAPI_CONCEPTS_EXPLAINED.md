# Giải Thích Các Khái Niệm Trong FastAPI Backend

## Tổng Quan
File `app.py` sử dụng FastAPI framework để xây dựng REST API cho hệ thống chatbot pháp lý. Tài liệu này giải thích các khái niệm chính được sử dụng trong file.

---

## 1. Pydantic Model

### Pydantic là gì?
Pydantic là một thư viện Python dùng để **validate dữ liệu** và **định nghĩa schema** (cấu trúc dữ liệu).

### Ví dụ trong code:
```python
class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "botLawyer"
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False
```

### Giải thích:
- **`CompleteRequest`**: Định nghĩa cấu trúc dữ liệu mà API endpoint `/chat/complete` sẽ nhận
- **Các trường (fields)**:
  - `bot_id`: Optional (không bắt buộc), mặc định là "botLawyer"
  - `user_id`: Required (bắt buộc), kiểu string
  - `user_message`: Required (bắt buộc), kiểu string - nội dung tin nhắn người dùng
  - `sync_request`: Optional, mặc định False - quyết định xử lý đồng bộ hay bất đồng bộ

### Lợi ích:
- ✅ **Tự động validate**: Nếu client gửi thiếu `user_id` hoặc sai kiểu dữ liệu → FastAPI tự động trả về lỗi 422
---

## 2. HTTP Methods: GET vs POST

### GET Method
**Mục đích**: Lấy dữ liệu từ server (READ operation)

#### Ví dụ trong code:
```python
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    # Lấy kết quả của task đã submit trước đó
    ...
```

#### Đặc điểm:
- 📖 **Chỉ đọc dữ liệu**, không thay đổi trạng thái server
- 🔗 **Parameters trong URL**: `/chat/complete/abc123` → `task_id = "abc123"`
- 💾 **Có thể cache**: Trình duyệt có thể cache kết quả
- 🔄 **Idempotent**: Gọi nhiều lần cho cùng kết quả

### POST Method
**Mục đích**: Gửi dữ liệu để tạo mới hoặc xử lý (CREATE/PROCESS operation)

#### Ví dụ trong code:
```python
@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    # Xử lý tin nhắn chat từ user
    ...

@app.post("/collection/create")
async def create_vector_collection(data: Dict):
    # Tạo collection mới trong vector database
    ...

@app.post("/document/create")
async def create_document(data: Dict):
    # Tạo document mới
    ...
```

#### Đặc điểm:
- 📝 **Gửi dữ liệu phức tạp**: Data nằm trong request body (JSON)
- ⚙️ **Thay đổi trạng thái**: Tạo mới, cập nhật, xử lý dữ liệu
- 🚫 **Không cache được**: Mỗi request có thể cho kết quả khác nhau
- 🔄 **Không idempotent**: Gọi nhiều lần có thể tạo nhiều bản ghi

---

## 3. Luồng Hoạt Động: Frontend → Backend

### Kịch bản: Người dùng gửi tin nhắn chat

#### Bước 1: Frontend gửi POST request
```javascript
// Frontend code (ví dụ)
const response = await fetch('http://backend:8002/chat/complete', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        user_id: "user123",
        user_message: "Luật giao thông quy định gì về mũ bảo hiểm?",
        sync_request: false
    })
});

const data = await response.json();
// Nhận được: { "task_id": "abc-123-def" }
```

#### Bước 2: Backend xử lý (Asynchronous Mode)
```python
@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    # FastAPI tự động parse JSON → CompleteRequest object
    # Validate các trường theo Pydantic model
    
    if data.sync_request:
        # Xử lý đồng bộ: chờ xong mới trả response
        response = llm_handle_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        # Xử lý bất đồng bộ: trả task_id ngay lập tức
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}  # ← Trả về ngay
```

#### Bước 3: Frontend poll kết quả (với async mode)
```javascript
// Sau khi có task_id, frontend gọi GET để lấy kết quả
const checkResult = async (taskId) => {
    const response = await fetch(`http://backend:8002/chat/complete/${taskId}`, {
        method: 'GET'
    });
    
    const result = await response.json();
    // Nhận được:
    // {
    //     "task_id": "abc-123-def",
    //     "task_status": "SUCCESS",
    //     "task_result": "Theo Luật Giao thông đường bộ 2008..."
    // }
};
```

#### Bước 4: Backend trả kết quả
```python
@app.get("/chat/complete/{task_id}")
async def get_response(task_id: str):
    # Kiểm tra trạng thái task trong Celery
    task_result = AsyncResult(task_id)
    
    if task_status == "PENDING":
        # Vẫn đang xử lý, đợi thêm
        time.sleep(0.5)
    else:
        # Đã xong, trả kết quả
        return {
            "task_id": task_id,
            "task_status": "SUCCESS",
            "task_result": task_result.result  # ← Câu trả lời từ LLM
        }
```

---

## 4. Tại Sao Dùng POST cho `/chat/complete`?

### Lý do chính:

1. **Dữ liệu phức tạp**: 
   - Cần gửi nhiều trường: `user_id`, `user_message`, `bot_id`, `sync_request`
   - Tin nhắn có thể rất dài, không phù hợp với URL query parameters

2. **Thay đổi trạng thái hệ thống**:
   - Tạo task mới trong Celery
   - Lưu lịch sử chat vào database
   - Gọi LLM để xử lý (tốn tài nguyên)

3. **Bảo mật**:
   - POST body được mã hóa trong HTTPS
   - Không lưu trong browser history như GET

4. **Không idempotent**:
   - Mỗi lần gửi tin nhắn giống nhau → tạo ra conversation khác nhau
   - Khác với GET `/health` (gọi 10 lần = 1 lần)

---

## 5. So Sánh Sync vs Async Request

### Synchronous Request (`sync_request: true`)
```
Frontend → POST /chat/complete (sync_request=true)
              ↓
          Backend xử lý ngay
              ↓ (chờ 5-10 giây)
          LLM trả lời
              ↓
Frontend ← Nhận response trực tiếp
```

**Ưu điểm**: Đơn giản, 1 request duy nhất  
**Nhược điểm**: Frontend bị block, user phải chờ, timeout nếu lâu

### Asynchronous Request (`sync_request: false`) - **MẶC ĐỊNH**
```
Frontend → POST /chat/complete (sync_request=false)
              ↓
Frontend ← Nhận task_id ngay lập tức (0.1s)
              ↓
          [Celery worker xử lý background]
              ↓
Frontend → GET /chat/complete/{task_id} (polling mỗi 0.5s)
              ↓
Frontend ← Nhận "PENDING" hoặc "SUCCESS" với kết quả
```

**Ưu điểm**: 
- Frontend không bị block
- User thấy loading indicator, biết hệ thống đang xử lý
- Xử lý được request lâu (> 60s)

**Nhược điểm**: Phức tạp hơn, cần polling hoặc WebSocket

---

