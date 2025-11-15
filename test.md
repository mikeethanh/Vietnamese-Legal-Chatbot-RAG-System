# Vietnamese Legal Chatbot Backend - Comprehensive Test Queries

## 📋 Tổng quan

File này chứa các câu test query toàn diện cho hệ thống Vietnamese Legal Chatbot RAG System, bao gồm các tính năng:

- **Follow-up Questions & Query Rewriting** - Xử lý câu hỏi tiếp theo và viết lại truy vấn
- **Route Detection** - Phân loại và định tuyến truy vấn (legal_rag, agent_tools, web_search, general_chat)
- **Legal RAG** - Tìm kiếm và trả lời dựa trên cơ sở dữ liệu pháp luật
- **Agent Tools** - Các công cụ tính toán và validation pháp lý
- **Web Search** - Tìm kiếm thông tin mới trên web
- **Multi-Query & Hybrid Search** - Tìm kiếm lai kết hợp semantic + keyword

## 🔧 API Endpoints

### Health Check

```bash
curl http://localhost:8002/health
```

### Chat Complete (Sync)

```bash
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_001",
    "user_message": "Thủ tục ly hôn như thế nào?",
    "sync_request": true
  }'
```

### Chat Complete (Async)

```bash
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_001",
    "user_message": "Quyền lợi của người lao động khi nghỉ việc là gì?"
  }'
```

## 🤖 1. Follow-up Questions & Query Rewriting

### Câu hỏi gốc và follow-up

```json
{
  "user_id": "followup_test_001",
  "conversation_history": [
    { "role": "user", "content": "Thủ tục ly hôn như thế nào?" },
    {
      "role": "assistant",
      "content": "Theo Luật Hôn nhân và Gia đình 2014, thủ tục ly hôn gồm: 1. Nộp đơn ly hôn tại UBND xã/phường..."
    }
  ],
  "user_message": "Còn chi phí thì sao?"
}
```

### Test cases cho Query Rewriting

```json
[
  {
    "context": "Người dùng vừa hỏi về thủ tục thành lập công ty",
    "follow_up": "Còn điều kiện đăng ký thì sao?",
    "expected_rewrite": "Điều kiện đăng ký thành lập doanh nghiệp theo pháp luật Việt Nam"
  },
  {
    "context": "Thảo luận về hợp đồng lao động",
    "follow_up": "Nó có hiệu lực bao lâu?",
    "expected_rewrite": "Thời hạn hiệu lực của hợp đồng lao động theo Bộ luật Lao động"
  },
  {
    "context": "Câu hỏi về thuế thu nhập cá nhân",
    "follow_up": "Làm sao để khai báo đây?",
    "expected_rewrite": "Cách thức khai báo thuế thu nhập cá nhân theo quy định"
  },
  {
    "context": "Hỏi về quyền thừa kế",
    "follow_up": "Có bao nhiêu hàng thừa kế vậy?",
    "expected_rewrite": "Số hàng thừa kế theo Bộ luật Dân sự Việt Nam"
  }
]
```

## 🧭 2. Route Detection Tests

### Legal RAG Route

```json
[
  {
    "query": "Quyền và nghĩa vụ của người lao động theo Bộ luật Lao động 2019",
    "expected_route": "legal_rag",
    "description": "Tra cứu văn bản pháp luật cụ thể"
  },
  {
    "query": "Thủ tục đăng ký kết hôn tại UBND",
    "expected_route": "legal_rag",
    "description": "Thủ tục hành chính theo quy định"
  },
  {
    "query": "Điều kiện để được ly hôn đơn phương",
    "expected_route": "legal_rag",
    "description": "Điều kiện pháp lý cụ thể"
  },
  {
    "query": "Trách nhiệm hình sự của người chưa thành niên",
    "expected_route": "legal_rag",
    "description": "Quy định về trách nhiệm hình sự"
  }
]
```

### Agent Tools Route

```json
[
  {
    "query": "Tính tiền phạt hợp đồng 500 triệu chậm 45 ngày với lãi suất 0.15% mỗi ngày",
    "expected_route": "agent_tools",
    "description": "Tính toán phạt hợp đồng"
  },
  {
    "query": "Kiểm tra người sinh năm 2006 có đủ tuổi ký hợp đồng lao động không?",
    "expected_route": "agent_tools",
    "description": "Kiểm tra tuổi pháp lý"
  },
  {
    "query": "Chia thừa kế cho 3 con với tài sản 2 tỷ đồng theo luật",
    "expected_route": "agent_tools",
    "description": "Tính toán chia thừa kế"
  },
  {
    "query": "Công ty ABC có hợp lệ theo quy định đặt tên doanh nghiệp không?",
    "expected_route": "agent_tools",
    "description": "Kiểm tra quy tắc đặt tên"
  }
]
```

### Web Search Route

```json
[
  {
    "query": "Luật Đất đai 2024 có những thay đổi gì mới nhất?",
    "expected_route": "web_search",
    "description": "Thông tin pháp luật mới"
  },
  {
    "query": "Mức lương tối thiểu vùng năm 2024 hiện tại",
    "expected_route": "web_search",
    "description": "Thông tin cập nhật gần đây"
  },
  {
    "query": "Vụ án tham nhũng ở Quảng Ninh vừa xét xử gần đây",
    "expected_route": "web_search",
    "description": "Tin tức pháp lý hiện tại"
  }
]
```

### General Chat Route

```json
[
  {
    "query": "Xin chào, bạn có thể giúp tôi được không?",
    "expected_route": "general_chat",
    "description": "Chào hỏi"
  },
  {
    "query": "Cảm ơn bạn đã hỗ trợ",
    "expected_route": "general_chat",
    "description": "Cảm ơn"
  },
  {
    "query": "Hôm nay thời tiết Hà Nội thế nào?",
    "expected_route": "general_chat",
    "description": "Chủ đề ngoài pháp luật"
  }
]
```

## 📚 3. Legal RAG System Tests

### Test Multi-Query Generation

```json
{
  "original_query": "Quyền lợi khi bị sa thải",
  "expected_variations": [
    "Quyền lợi khi bị sa thải",
    "Quyền lợi khi bị chấm dứt hợp đồng lao động",
    "Bồi thường khi người lao động bị sa thải trái luật",
    "Trợ cấp thôi việc cho người lao động"
  ]
}
```

### Test Hybrid Search

```json
[
  {
    "query": "hợp đồng lao động",
    "expected_semantic": "tìm documents về employment contract",
    "expected_keyword": "tìm exact match 'hợp đồng lao động'"
  },
  {
    "query": "ly hôn đơn phương",
    "expected_semantic": "tìm về unilateral divorce",
    "expected_keyword": "tìm exact phrase trong documents"
  }
]
```

### Test Query Expansion

```json
{
  "query": "vi phạm hợp đồng",
  "expanded_terms": [
    "vi phạm hợp đồng",
    "phạm hợp đồng",
    "trái hợp đồng",
    "sai phạm hợp đồng",
    "không tuân thủ hợp đồng"
  ]
}
```

### Complex Legal Questions

```json
[
  {
    "query": "Người nước ngoài có thể sở hữu nhà ở Việt Nam không?",
    "complexity": "high",
    "expected_docs": ["Luật Nhà ở", "Luật Đầu tư", "Nghị định 99/2015"]
  },
  {
    "query": "Điều kiện để được miễn thuế thu nhập doanh nghiệp",
    "complexity": "medium",
    "expected_docs": ["Luật Thuế TNDN", "Nghị định 218/2013"]
  }
]
```

## 🛠️ 4. Agent Tools Tests

### Contract Penalty Calculator

```json
[
  {
    "tool": "contract_penalty_calculator",
    "params": {
      "contract_value": 1000000000,
      "penalty_rate": 0.1,
      "days_late": 30
    },
    "expected_result": {
      "penalty_amount": "30,000,000 VNĐ",
      "note": "Tính theo tỷ lệ phạt đã thỏa thuận"
    }
  },
  {
    "tool": "contract_penalty_calculator",
    "params": {
      "contract_value": 500000000,
      "penalty_rate": 0.5,
      "days_late": 365
    },
    "expected_result": {
      "penalty_amount": "60,000,000 VNĐ",
      "note": "Đã áp dụng mức phạt tối đa 12% giá trị hợp đồng"
    }
  }
]
```

### Legal Age Checker

```json
[
  {
    "tool": "legal_age_checker",
    "params": {
      "birth_year": 2005,
      "action_type": "sign_contract"
    },
    "expected_result": {
      "eligible": true,
      "age": 19,
      "requirement": "Đủ 18 tuổi để ký hợp đồng"
    }
  },
  {
    "tool": "legal_age_checker",
    "params": {
      "birth_year": 2010,
      "action_type": "marriage"
    },
    "expected_result": {
      "eligible": false,
      "age": 14,
      "requirement": "Nam đủ 20 tuổi, Nữ đủ 18 tuổi"
    }
  }
]
```

### Inheritance Calculator

```json
[
  {
    "tool": "inheritance_calculator",
    "params": {
      "total_estate": 2000000000,
      "heirs": [
        { "name": "Con 1", "relationship": "con", "share_ratio": 1 },
        { "name": "Con 2", "relationship": "con", "share_ratio": 1 },
        { "name": "Vợ", "relationship": "vợ/chồng", "share_ratio": 1 }
      ]
    },
    "expected_result": {
      "total_shares": 3,
      "share_value": "666,666,667 VNĐ",
      "distribution": "Mỗi người được 666.67 triệu VNĐ"
    }
  }
]
```

### Business Name Validator

```json
[
  {
    "tool": "business_name_validator",
    "params": {
      "business_name": "Công ty TNHH ABC"
    },
    "expected_result": {
      "valid": true,
      "analysis": "Tên phù hợp với quy định"
    }
  },
  {
    "tool": "business_name_validator",
    "params": {
      "business_name": "Ngân hàng XYZ"
    },
    "expected_result": {
      "valid": false,
      "issues": ["Từ 'Ngân hàng' cần có giấy phép đặc biệt"]
    }
  }
]
```

### Statute of Limitations Checker

```json
[
  {
    "tool": "statute_checker",
    "params": {
      "case_type": "tranh chấp hợp đồng",
      "incident_date": "2022-01-01"
    },
    "expected_result": {
      "time_limit": "3 năm",
      "deadline": "2025-01-01",
      "status": "Còn thời hiệu"
    }
  }
]
```

## 🌐 5. Web Search Integration Tests

### Tavily Search Tests

```json
[
  {
    "query": "Nghị định mới về giao thông 2024",
    "search_type": "tavily_search_legal",
    "expected_sources": ["thuvienphapluat.vn", "baochinhphu.vn"],
    "expected_content": "Thông tin về văn bản pháp luật mới"
  },
  {
    "query": "Lương tối thiểu vùng 1 năm 2024",
    "search_type": "tavily_qna",
    "expected_answer": "Mức lương tối thiểu cụ thể"
  }
]
```

### Search Result Integration

```json
{
  "query": "Luật Đất đai 2024 thay đổi gì",
  "expected_flow": [
    "1. Phát hiện từ khóa 'mới nhất', '2024' → route: web_search",
    "2. Gọi tavily_search_legal()",
    "3. Tổng hợp kết quả từ web",
    "4. Tạo câu trả lời dựa trên thông tin tìm được"
  ]
}
```

## 🔀 6. Multi-Query & Reranking Tests

### Query Variations

```json
{
  "original": "Thủ tục thành lập công ty",
  "variations": [
    "Thủ tục thành lập công ty",
    "Quy trình đăng ký doanh nghiệp",
    "Các bước thành lập doanh nghiệp",
    "Giấy tờ cần thiết để thành lập công ty"
  ]
}
```

### Document Reranking

```json
{
  "query": "quyền lợi người lao động",
  "initial_results": [
    { "score": 0.85, "content": "Bài về quyền lao động" },
    { "score": 0.82, "content": "Bài về nghĩa vụ lao động" },
    { "score": 0.8, "content": "Bài về hợp đồng lao động" }
  ],
  "expected_rerank": "Prioritize documents about worker rights specifically"
}
```

## 📊 7. Performance & Edge Case Tests

### Large Document Handling

```json
{
  "query": "Toàn bộ quy định về thuế",
  "expected_behavior": "Handle large document retrieval efficiently",
  "max_response_time": "5 seconds",
  "max_tokens": 4000
}
```

### Multilingual Queries

```json
[
  {
    "query": "What are labor rights in Vietnam?",
    "expected_handling": "Detect English → translate/handle appropriately"
  },
  {
    "query": "Luật lao động Việt Nam (Vietnam Labor Law)",
    "expected_handling": "Handle mixed language query"
  }
]
```

### Error Handling

```json
[
  {
    "scenario": "Empty query",
    "input": "",
    "expected": "Request validation error"
  },
  {
    "scenario": "Extremely long query",
    "input": "A very long legal question that exceeds normal limits...",
    "expected": "Truncate or handle gracefully"
  },
  {
    "scenario": "Special characters",
    "input": "Quyền @#$% người lao động ???",
    "expected": "Clean and process normally"
  }
]
```

## 🧪 8. Integration Test Scenarios

### Full Conversation Flow

```json
{
  "conversation": [
    {
      "step": 1,
      "user": "Tôi muốn thành lập công ty",
      "expected_route": "legal_rag",
      "expected_response": "Thông tin về thủ tục thành lập doanh nghiệp"
    },
    {
      "step": 2,
      "user": "Chi phí là bao nhiêu?",
      "expected_route": "legal_rag",
      "expected_rewrite": "Chi phí thủ tục thành lập doanh nghiệp",
      "expected_response": "Thông tin về lệ phí đăng ký"
    },
    {
      "step": 3,
      "user": "Tính lệ phí cho vốn điều lệ 10 tỷ đồng",
      "expected_route": "agent_tools",
      "expected_tool": "business_registration_fee_calculator",
      "expected_response": "Tính toán cụ thể lệ phí"
    }
  ]
}
```

### Mixed Route Conversation

```json
{
  "conversation": [
    {
      "user": "Xin chào!",
      "expected_route": "general_chat"
    },
    {
      "user": "Quy định về hợp đồng lao động",
      "expected_route": "legal_rag"
    },
    {
      "user": "Tính phạt chậm lương 30 ngày với mức lương 10 triệu",
      "expected_route": "agent_tools"
    },
    {
      "user": "Chính sách lương mới nhất 2024",
      "expected_route": "web_search"
    }
  ]
}
```

## 🚀 9. Load Testing Queries

### High Volume Tests

```bash
# Concurrent requests test
for i in {1..50}; do
  curl -X POST http://localhost:8002/chat/complete \
    -H "Content-Type: application/json" \
    -d "{\"user_id\": \"load_test_${i}\", \"user_message\": \"Quy định về thuế TNCN\", \"sync_request\": true}" &
done
```

### Memory Stress Tests

```json
[
  {
    "query": "Toàn bộ Bộ luật Dân sự 2015",
    "purpose": "Test large document retrieval"
  },
  {
    "query": "So sánh tất cả luật về lao động từ 1995 đến 2024",
    "purpose": "Test complex multi-document analysis"
  }
]
```

## 📝 10. Custom Test Scripts

### Backend Health Check Script

```python
import requests
import json
import time

def test_backend_health():
    """Test all backend endpoints"""

    # Health check
    health = requests.get("http://localhost:8002/health")
    print(f"Health: {health.status_code} - {health.json()}")

    # Legal RAG test
    legal_query = {
        "user_id": "test_001",
        "user_message": "Quyền lợi của người lao động khi bị sa thải",
        "sync_request": True
    }

    legal_response = requests.post(
        "http://localhost:8002/chat/complete",
        json=legal_query
    )
    print(f"Legal RAG: {legal_response.status_code}")
    print(f"Response: {legal_response.json()}")

    # Agent tools test
    agent_query = {
        "user_id": "test_002",
        "user_message": "Tính phạt hợp đồng 100 triệu chậm 15 ngày với lãi 0.1%/ngày",
        "sync_request": True
    }

    agent_response = requests.post(
        "http://localhost:8002/chat/complete",
        json=agent_query
    )
    print(f"Agent Tools: {agent_response.status_code}")
    print(f"Response: {agent_response.json()}")

if __name__ == "__main__":
    test_backend_health()
```

### Async Task Testing

```python
import requests
import time

def test_async_processing():
    """Test async task processing"""

    # Start async task
    query = {
        "user_id": "async_test",
        "user_message": "Thủ tục ly hôn và chia tài sản"
    }

    response = requests.post("http://localhost:8002/chat/complete", json=query)
    task_id = response.json().get("task_id")
    print(f"Task started: {task_id}")

    # Poll for result
    while True:
        result = requests.get(f"http://localhost:8002/chat/complete/{task_id}")
        status = result.json().get("task_status")
        print(f"Status: {status}")

        if status != "PENDING":
            print(f"Final result: {result.json()}")
            break

        time.sleep(1)

if __name__ == "__main__":
    test_async_processing()
```

## 📋 11. Test Data Management

### Create Test Collection

```bash
curl -X POST http://localhost:8002/collection/create \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "test_legal_docs"}'
```

### Import Test Documents

```bash
curl -X POST http://localhost:8002/document/create \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_doc_001",
    "question": "Quy định về thời giờ làm việc",
    "content": "Theo Bộ luật Lao động 2019, thời giờ làm việc bình thường không quá 8 giờ một ngày và không quá 48 giờ một tuần..."
  }'
```

### Bulk Import Test Data

```bash
curl -X POST http://localhost:8002/data/import \
  -H "Content-Type: application/json"
```

## 🎯 Kết luận

File test này cung cấp coverage toàn diện cho:

1. **API Endpoints** - Tất cả routes và methods
2. **Query Processing** - Follow-up, rewriting, routing
3. **RAG System** - Vector search, hybrid search, reranking
4. **Agent Tools** - Legal calculations and validations
5. **Web Integration** - External search capabilities
6. **Performance** - Load testing và edge cases
7. **Integration** - End-to-end conversation flows

Sử dụng các test cases này để validate functionality và performance của Vietnamese Legal Chatbot backend system.
