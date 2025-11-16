# Vietnamese Legal Chatbot Backend - Comprehensive Test Queries

## 📋 Tổng quan

File này chứa các câu test query toàn diện cho hệ thống Vietnamese Legal Chatbot RAG System, bao gồm các tính năng:

- **Follow-up Questions & Query Rewriting** - Xử lý câu hỏi tiếp theo và viết lại truy vấn
- **Route Detection** - Phân loại và định tuyến truy vấn (legal_rag, agent_tools, web_search, general_chat)
- **Legal RAG** - Tìm kiếm và trả lời dựa trên cơ sở dữ liệu pháp luật
- **Agent Tools** - Các công cụ tính toán và validation pháp lý
- **Web Search** - Tìm kiếm thông tin mới trên web
- **Multi-Query & Hybrid Search** - Tìm kiếm lai kết hợp semantic + keyword

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
