# Giải Thích Chi Tiết AI Agent - ReAct Workflow

## Tổng Quan
File `agent.py` triển khai **ReAct Agent** (Reasoning + Acting) - một AI agent có khả năng:
- **Suy luận** (Reasoning): Phân tích câu hỏi, lập kế hoạch
- **Hành động** (Acting): Sử dụng tools (công cụ) để thực hiện tác vụ
- **Tự điều chỉnh**: Dựa vào kết quả để quyết định bước tiếp theo

---

## 1. ReAct Agent Là Gì?

### Định nghĩa
**ReAct** = **Re**asoning + **Act**ing

## 2. Các Tools (Công Cụ) Có Sẵn

Hệ thống có **8 tools** chia làm 2 nhóm:

### 2.1. Legal Calculation Tools (5 tools)

#### Tool 1: `contract_penalty_calculator`

**Mục đích**: Tính tiền phạt vi phạm hợp đồng

```python
def contract_penalty_calculator(
    contract_value: float,    # Giá trị hợp đồng (VNĐ)
    penalty_rate: float,      # Tỷ lệ phạt (%/ngày)
    days_late: int           # Số ngày chậm
) -> str:

```
---

#### Tool 2: `legal_age_checker`

**Mục đích**: Kiểm tra tuổi pháp lý để thực hiện hành vi dân sự


#### Tool 3: `inheritance_calculator`

**Mục đích**: Tính phần thừa kế theo pháp luật (hàng thừa kế thứ nhất)

**Nguyên tắc chia thừa kế**:
- Hàng thứ nhất: Vợ/chồng, con, cha/mẹ → **Chia đều**
- Nếu có di chúc → Ưu tiên di chúc
- Người chưa thành niên cần người quản lý tài sản

---

#### Tool 4: `business_name_validator`

**Mục đích**: Kiểm tra tên doanh nghiệp hợp lệ theo Luật Doanh nghiệp


---

### 2.2. Web Search Tools (3 tools)

#### Tool 7: `tavily_search_tool`

**Mục đích**: Tìm kiếm thông minh với Tavily AI

```python
def tavily_search_tool(query: str, max_results: int = 5) -> str:
```

**Ưu điểm so với Google**:
- AI tự động tóm tắt kết quả
- Lọc nguồn tin cậy
- Trả về kết quả có cấu trúc

---

#### Tool 8: `quick_answer_tool`

**Mục đích**: Trả lời nhanh bằng Tavily Q&A

```python
def quick_answer_tool(question: str) -> str:
```

**Ví dụ**:
```
Question: "Mức lương tối thiểu vùng 1 năm 2024?"
Answer: "4,960,000 VNĐ/tháng theo Nghị định 24/2023/NĐ-CP"
```

---

## 3. Workflow Chi Tiết - `ai_agent_handle()`

### 3.1. Cấu trúc Agent

```python
# Step 1: Khởi tạo LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
# temperature=0.1 → Ít creative, consistent hơn

# Step 2: Tạo agent với system prompt
agent_system_prompt = """Bạn là trợ lý AI chuyên về tư vấn pháp luật Việt Nam..."""

ai_agent = ReActAgent.from_tools(
    all_tools,              # 8 tools đã định nghĩa
    llm=llm,
    verbose=True,           # Log chi tiết
    max_iterations=10,      # Tối đa 10 vòng lặp
    context=agent_system_prompt
)
```

### 3.2. Hàm Main - ai_agent_handle()

```python
@shared_task()
def ai_agent_handle(question: str) -> str:
    """
    Handle user question using ReAct agent with tools.
    """
    try:
        logger.info(f"[AGENT] Processing question: {question}")
        response = ai_agent.chat(question)  # ← ReAct loop ở đây
        logger.info(f"[AGENT] Response generated successfully")
        return response.response
    except Exception as e:
        logger.error(f"[AGENT] Error: {e}")
        return f"Xin lỗi, đã xảy ra lỗi: {str(e)}"
```

