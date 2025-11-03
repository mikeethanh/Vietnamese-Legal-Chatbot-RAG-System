# Agent Tools và Routing - Hướng Dẫn

## Tổng Quan

Hệ thống chatbot pháp luật Việt Nam đã được nâng cấp với:
1. **ReAct Agent với 8 tools** - xử lý tính toán và validation pháp lý
2. **Routing thông minh 4 routes** - phân loại và xử lý câu hỏi tối ưu
3. **Tavily AI search** - tìm kiếm web thông minh với AI

---

## 1. Agent Tools (8 Tools)

### Legal Calculation Tools (5 tools)

#### 1.1 Contract Penalty Calculator
**Công dụng:** Tính tiền phạt vi phạm hợp đồng theo Bộ luật Dân sự 2015

**Ví dụ:**
```
User: "Tính tiền phạt hợp đồng 100 triệu chậm 30 ngày với lãi 0.1%/ngày"
Agent: Sử dụng contract_penalty_calculator(100000000, 0.1, 30)
→ Kết quả: Phạt 3 triệu VNĐ
```

#### 1.2 Legal Age Checker
**Công dụng:** Kiểm tra tuổi pháp lý cho các hành vi dân sự

**Ví dụ:**
```
User: "Năm sinh 2005 có đủ tuổi ký hợp đồng không?"
Agent: Sử dụng legal_age_checker(2005, "sign_contract")
→ Kết quả: Đủ 18 tuổi, hợp lệ theo Điều 21 Bộ luật Dân sự
```

**Các loại hành vi:**
- `sign_contract` - ký hợp đồng (18 tuổi)
- `marriage` - kết hôn (Nam 20, Nữ 18)
- `work` - làm việc (15 tuổi)
- `criminal_responsibility` - chịu trách nhiệm hình sự (16 tuổi)

#### 1.3 Inheritance Calculator
**Công dụng:** Tính chia thừa kế theo pháp luật (hàng thứ nhất)

**Ví dụ:**
```
User: "Tài sản 500 triệu chia cho 3 người: vợ, 2 con"
Agent: Sử dụng inheritance_calculator với heirs_json
→ Mỗi người: 166.67 triệu (chia đều theo Điều 651 Bộ luật Dân sự)
```

#### 1.4 Business Name Validator
**Công dụng:** Kiểm tra tên doanh nghiệp hợp lệ theo Luật Doanh nghiệp 2020

**Ví dụ:**
```
User: "Tên 'Công ty Việt Nam ABC' có hợp lệ không?"
Agent: Sử dụng business_name_validator("Công ty Việt Nam ABC")
→ Không hợp lệ (chứa từ cấm "Việt Nam")
```

#### 1.5 Statute of Limitations Lookup
**Công dụng:** Tra cứu thời hiệu khởi kiện

**Ví dụ:**
```
User: "Thời hiệu khởi kiện tranh chấp lao động là bao lâu?"
Agent: Sử dụng statute_lookup("labor")
→ 1 năm (Điều 193 Bộ luật Lao động 2019)
```

**Các loại vụ việc:**
- `civil` - dân sự (3 năm)
- `labor` - lao động (1 năm)
- `administrative` - hành chính (1 năm)
- `criminal` - hình sự (tùy mức độ tội phạm)

### Web Search Tools (3 tools)

#### 1.6 Google Search Tool
**Công dụng:** Tìm kiếm thông tin trên Google

#### 1.7 Tavily Search Tool
**Công dụng:** Tìm kiếm thông minh với AI, tự động tóm tắt

#### 1.8 Quick Answer Tool
**Công dụng:** Trả lời nhanh câu hỏi từ web search

---

## 2. Routing System (4 Routes)

### Route 1: `legal_rag`
**Khi nào dùng:**
- Câu hỏi về nội dung luật, nghị định, thông tư
- Thủ tục pháp lý
- Quyền lợi, nghĩa vụ pháp lý
- Giải thích khái niệm pháp lý

**Ví dụ:**
```
"Thủ tục ly hôn theo Bộ luật Dân sự?"
"Quyền của người lao động khi bị sa thải?"
"Điều 651 Bộ luật Dân sự quy định gì?"
```

**Cách xử lý:**
1. Detect follow-up question → standalone question
2. Multi-query retrieval (3 queries)
3. Retrieve from Qdrant vector DB
4. Rerank with Cohere
5. Generate answer with context

### Route 2: `agent_tools`
**Khi nào dùng:**
- Câu hỏi có từ khóa: tính, kiểm tra, hợp lệ, đủ tuổi, chia
- Cần tính toán số liệu
- Cần validation theo quy định
- Cần suy luận nhiều bước

**Ví dụ:**
```
"Tính tiền phạt hợp đồng 200 triệu chậm 45 ngày lãi 0.2%"
"Kiểm tra tên 'Công ty ABC' có hợp lệ không?"
"Năm sinh 2008 có đủ tuổi làm việc không?"
"Thời hiệu khởi kiện hành chính là bao lâu?"
```

**Cách xử lý:**
1. Detect follow-up question
2. Pass to ReAct agent
3. Agent reasoning + tool picking
4. Execute tools
5. Generate final answer

### Route 3: `web_search`
**Khi nào dùng:**
- Từ khóa: mới nhất, gần đây, hiện nay, năm 2024
- Tin tức pháp luật
- Thống kê hiện tại
- Văn bản pháp luật vừa ban hành

**Ví dụ:**
```
"Luật Đất đai 2024 có gì mới?"
"Lương tối thiểu vùng 1 năm 2024"
"Tin tức về sửa đổi Luật Giao thông gần đây"
```

**Cách xử lý:**
1. Detect follow-up question
2. Google Search hoặc Tavily Search
3. Generate answer from search results

### Route 4: `general_chat`
**Khi nào dùng:**
- Chào hỏi, cảm ơn
- Hỏi về khả năng của bot
- Off-topic

**Ví dụ:**
```
"Xin chào"
"Bạn có thể giúp gì cho tôi?"
"Cảm ơn bạn"
```

---

## 3. Flow Diagram

```
User Question
     ↓
[Routing System] - Phân loại câu hỏi
     ↓
     ├─→ legal_rag → RAG Pipeline (vector search + rerank)
     ├─→ agent_tools → ReAct Agent (reasoning + tools)
     ├─→ web_search → Google/Tavily Search
     └─→ general_chat → Direct LLM response
     ↓
Response to User
```

---

## 4. Cấu Hình và Environment Variables

### Required API Keys:
```bash
# OpenAI (bắt buộc)
OPENAI_API_KEY=sk-xxx

# Cohere for reranking (bắt buộc)
COHERE_API_KEY=xxx

# Google Search (optional)
GOOGLE_API_KEY=xxx
GOOGLE_CSE_ID=xxx

# Tavily AI Search (optional nhưng khuyến nghị)
TAVILY_API_KEY=tvly-xxx
```

### Cài đặt:
```bash
pip install tavily-python>=0.3.0
```

---

## 5. Ví Dụ Sử Dụng End-to-End

### Case 1: Tính toán pháp lý (agent_tools)
```
Input: "Hợp đồng 150 triệu, phạt 0.15%/ngày, chậm 20 ngày, tính tiền phạt"

Routing: agent_tools
Agent thinking:
  - Cần dùng contract_penalty_calculator
  - Parameters: 150000000, 0.15, 20
Tool execution:
  - Result: 4,500,000 VNĐ
  - Note: Theo quy định Bộ luật Dân sự 2015

Output: "Tiền phạt vi phạm hợp đồng là 4.5 triệu VNĐ..."
```

### Case 2: Tra cứu luật (legal_rag)
```
Input: "Quyền của người lao động khi bị sa thải trái luật"

Routing: legal_rag
Process:
  - Generate 3 queries
  - Search Qdrant vector DB
  - Rerank top 5 docs
  - Generate answer with legal citations

Output: "Theo Điều 42 Bộ luật Lao động 2019, người lao động bị sa thải trái luật có quyền..."
```

### Case 3: Tìm kiếm tin mới (web_search)
```
Input: "Lương tối thiểu vùng 1 năm 2024 là bao nhiêu?"

Routing: web_search
Process:
  - Tavily search: "Việt Nam pháp luật: lương tối thiểu vùng 1 2024"
  - Get AI summary + sources

Output: "Theo Nghị định xxx/2024, lương tối thiểu vùng 1 từ 01/7/2024 là..."
```

---

## 6. Best Practices

### Khi nào dùng Agent Tools:
✅ Có số liệu cần tính toán
✅ Có tiêu chí cần kiểm tra/validation
✅ Câu hỏi dạng "có hợp lệ không", "đủ tuổi không"
✅ Cần tra cứu giá trị cố định (thời hiệu, tuổi pháp lý)

❌ KHÔNG dùng khi:
- Hỏi về nội dung văn bản pháp luật (dùng legal_rag)
- Cần thông tin cập nhật từ internet (dùng web_search)

### Tối ưu Routing:
- Ưu tiên: agent_tools > legal_rag > web_search > general_chat
- Nếu có từ "tính", "kiểm tra" → agent_tools
- Nếu có năm 2024, "mới nhất" → web_search
- Còn lại về pháp luật → legal_rag

---

## 7. Monitoring và Debug

### Logs để theo dõi:
```python
[ROUTE] Selected route: agent_tools
[AGENT] Processing question: ...
[TOOL] Contract penalty calculated: {...}
[AGENT] Response generated successfully
```

### Common Issues:
1. **Agent không chọn tool đúng**: Cải thiện docstring của tool
2. **Routing sai**: Kiểm tra prompt trong `detect_route()`
3. **Tavily error**: Kiểm tra TAVILY_API_KEY

---

## 8. Future Improvements

- [ ] Thêm tool tra cứu tên doanh nghiệp từ database thật
- [ ] Tool tính lãi suất vay ngân hàng
- [ ] Tool tra cứu bộ luật theo năm ban hành
- [ ] Cải thiện agent reasoning với CoT prompting
- [ ] A/B test routing accuracy
