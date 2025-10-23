# Tóm Tắt Các Thay Đổi - Agent Tools & Routing

## ✅ Đã Hoàn Thành

### 1. Legal Tools (legal_tools.py) - MỚI
Tạo 5 tools pháp lý thực tế và đơn giản:

- **calculate_contract_penalty()** - Tính phạt vi phạm hợp đồng
  - Input: giá trị HĐ, tỷ lệ phạt, số ngày chậm
  - Logic: áp dụng giới hạn tối đa 12% theo thông lệ
  
- **check_legal_entity_age()** - Kiểm tra tuổi pháp lý
  - Support: ký hợp đồng, kết hôn, làm việc, chịu trách nhiệm hình sự
  - Dựa trên: Bộ luật Dân sự, Luật Hôn nhân, Bộ luật Lao động, Hình sự
  
- **calculate_inheritance_share()** - Tính chia thừa kế
  - Áp dụng: hàng thừa kế thứ nhất chia đều
  - Theo: Điều 651 Bộ luật Dân sự 2015
  
- **check_business_name_rules()** - Kiểm tra tên doanh nghiệp
  - Validate: từ ngữ cấm, độ dài, ký tự đặc biệt
  - Theo: Điều 36 Luật Doanh nghiệp 2020
  
- **get_statute_of_limitations()** - Tra cứu thời hiệu
  - Support: dân sự, lao động, hành chính, hình sự
  - Trả về: thời hạn, căn cứ pháp lý, trường hợp ngoại lệ

### 2. Tavily Search Tool (tavily_tool.py) - MỚI
Tích hợp Tavily AI search với 3 functions:

- **tavily_search()** - Core search function
  - Features: AI-generated summary, relevance scoring
  - Fallback: graceful degradation nếu library chưa cài
  
- **tavily_search_legal()** - Optimized cho legal queries
  - Enhancement: thêm "Việt Nam pháp luật" vào query
  - Search depth: advanced mode
  
- **tavily_qna()** - Quick Q&A
  - Use case: factual questions cần web search
  - Return: direct answer hoặc summary từ top results

### 3. Agent Enhancement (agent.py) - CẬP NHẬT
Nâng cấp từ 3 tools → 8 tools với Vietnamese legal context:

**Legal Tools (5):**
- contract_penalty_calculator
- legal_age_checker  
- inheritance_calculator
- business_name_validator
- statute_lookup

**Search Tools (3):**
- web_search_tool (Google)
- tavily_search_tool (Tavily AI)
- quick_answer_tool (Tavily Q&A)

**Agent Config:**
- Model: gpt-4o-mini, temperature=0.1
- Max iterations: 10
- System prompt: Vietnamese legal assistant với tool usage guidelines
- Verbose logging: enabled

### 4. Routing Improvement (brain.py) - CẬP NHẬT
Mở rộng từ 3 routes → 4 routes:

**Routes:**
1. `legal_rag` - RAG system cho tra cứu văn bản pháp luật
2. `agent_tools` - **MỚI** - ReAct agent cho tính toán/validation
3. `web_search` - Google/Tavily cho tin tức mới
4. `general_chat` - Trò chuyện thông thường

**Detection Logic:**
- Enhanced prompt với ví dụ rõ ràng cho từng route
- Fallback logic: detect từ khóa "tính", "kiểm tra" → agent_tools
- Priority: agent_tools > legal_rag > web_search > general_chat

### 5. Task Pipeline (tasks.py) - CẬP NHẬT
Thêm xử lý cho `agent_tools` route:

```python
elif route == 'agent_tools':
    standalone_question = follow_up_question(history, question)
    agent_response = ai_agent_handle(standalone_question)
    return agent_response
```

### 6. Dependencies (requirements.txt) - CẬP NHẬT
Thêm:
```
tavily-python>=0.3.0
```

### 7. Documentation (AGENT_TOOLS_GUIDE.md) - MỚI
Tài liệu đầy đủ về:
- Cách sử dụng 8 tools
- Routing system và khi nào dùng route nào
- Flow diagram
- Ví dụ end-to-end
- Best practices
- Troubleshooting

---

## 🔧 Cấu Hình Cần Thiết

### Environment Variables Mới:
```bash
# Optional nhưng khuyến nghị
TAVILY_API_KEY=tvly-xxx
```

### Existing (không thay đổi):
```bash
OPENAI_API_KEY=sk-xxx
COHERE_API_KEY=xxx
GOOGLE_API_KEY=xxx  # optional
GOOGLE_CSE_ID=xxx   # optional
```

---

## 📊 So Sánh Trước/Sau

### Trước:
```
3 routes: legal_rag | web_search | general_chat
3 tools: multiply | add | search_engine
Capabilities: Chỉ tra cứu và search
```

### Sau:
```
4 routes: legal_rag | agent_tools | web_search | general_chat
8 tools: 5 legal + 3 search
Capabilities: 
  ✅ Tra cứu văn bản pháp luật
  ✅ Tính toán pháp lý (phạt HĐ, thừa kế)
  ✅ Validation (tuổi, tên DN, thời hiệu)
  ✅ Web search thông minh (Tavily AI)
  ✅ Multi-step reasoning (ReAct)
```

---

## 🚀 Cách Deploy

### 1. Build lại Docker image:
```bash
cd backend
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 2. Kiểm tra logs:
```bash
docker logs -f chatbot-api
docker logs -f chatbot-worker
```

### 3. Test routing:
```bash
# Test agent_tools route
curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "user_message": "Tính phạt hợp đồng 100 triệu chậm 30 ngày lãi 0.1%/ngày"
  }'

# Test legal_rag route  
curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "user_message": "Thủ tục ly hôn theo luật Việt Nam"
  }'
```

---

## 🐛 Troubleshooting

### Issue 1: Tavily import error
```
ImportError: No module named 'tavily'
```
**Fix:** 
```bash
docker-compose exec chatbot-worker pip install tavily-python
# hoặc rebuild image
```

### Issue 2: Agent không chọn tool đúng
```
Agent returns text answer thay vì dùng tool
```
**Fix:**
- Check tool docstrings có rõ ràng không
- Check agent verbose logs
- Có thể cần điều chỉnh temperature (hiện tại: 0.1)

### Issue 3: Routing sai
```
Câu hỏi tính toán nhưng route về legal_rag
```
**Fix:**
- Check logs: `[ROUTE] Selected route: xxx`
- Xem response từ OpenAI có hợp lệ không
- Fallback keywords có trigger không

---

## 📈 Metrics Đề Xuất

Monitor các metrics sau:
- Route distribution: % của mỗi route được sử dụng
- Agent tool usage: tool nào được dùng nhiều nhất
- Agent success rate: % agent hoàn thành vs error
- Response time: avg time cho mỗi route
- User satisfaction: feedback từ users

---

## 🎯 Next Steps (Optional)

1. **A/B Testing**: So sánh routing accuracy
2. **Tool Expansion**: Thêm tools cho:
   - Tính lãi suất vay
   - Tra cứu bộ luật theo năm
   - Tính thuế thu nhập cá nhân
3. **Agent Improvement**: 
   - Chain-of-Thought prompting
   - Self-reflection mechanism
4. **Caching**: Cache agent responses cho câu hỏi tương tự

---

## ✨ Highlights

**Đơn giản, thực tế:**
- Không phụ thuộc vào external APIs phức tạp (trừ Tavily optional)
- Tools dựa trên logic Python thuần, không cần database
- Fallback gracefully khi thiếu dependencies

**Tối ưu cho Vietnamese legal:**
- All prompts in Vietnamese
- Dựa trên luật Việt Nam (Dân sự, Lao động, Hình sự, Doanh nghiệp)
- Examples và use cases cụ thể cho người Việt

**Production-ready:**
- Comprehensive error handling
- Detailed logging
- Clear documentation
- Easy to extend với tools mới
