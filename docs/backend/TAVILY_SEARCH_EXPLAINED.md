# Giải Thích Chi Tiết Tavily Search Tool

## Tổng Quan
**Tavily** là một **AI-powered search engine** được tối ưu hóa cho LLM (Large Language Models). Khác với Google Search thông thường, Tavily:
- ✅ Tự động **tóm tắt kết quả** bằng AI
- ✅ **Lọc nguồn đáng tin cậy**
- ✅ Trả về kết quả **có cấu trúc** (structured data)
- ✅ Tối ưu cho **RAG systems** và **AI agents**

---

## 1. Tavily vs Google Search - So Sánh

### Google Search (Traditional)

```python

→ LLM phải tự đọc và tóm tắt
→ Có thể có nhiều nguồn không đáng tin
→ Khó parse và extract info
```

### Tavily Search (AI-Powered)

```python
}

→ Có sẵn AI summary
→ Nguồn được ranked theo relevance
→ Structured data dễ sử dụng
```

### Bảng So Sánh

| Tính năng | Google Search | Tavily Search |
|-----------|---------------|---------------|
| **AI Summary** | ❌ Không | ✅ Có (tự động tóm tắt) |
| **Relevance Score** | ❌ Không | ✅ Có (0.0-1.0) |
| **Source Quality** | ⚠️ Tất cả nguồn | ✅ Lọc nguồn uy tín |
| **Output Format** | HTML snippets | ✅ JSON structured |
| **Optimized for AI** | ❌ Không | ✅ Có |
| **Response Time** | ~500ms | ~1-2s (do AI processing) |
| **Cost** | Free (hạn chế API) | Paid ($0.005/request) |
| **Use Case** | General search | ✅ **RAG/Agent systems** |

---

## 3. Các Hàm Trong tavily_tool.py

### 3.1. `tavily_search()` - Hàm Cơ Bản

**Mục đích**: Tìm kiếm cơ bản với Tavily API

```python
def tavily_search(
    query: str,                    # Câu hỏi tìm kiếm
    max_results: int = 5,          # Số kết quả
    search_depth: str = "basic"    # "basic" hoặc "advanced"
) -> Dict:
```

#### Search Depth - Sự Khác Biệt

| Depth | Thời gian | Chi phí | Độ chính xác | Khi nào dùng? |
|-------|-----------|---------|--------------|---------------|
| **basic** | ~1s | $0.003/req | 85-90% | Câu hỏi đơn giản, nhanh |
| **advanced** | ~2-3s | $0.005/req | 95-98% | Câu hỏi phức tạp, cần chính xác |
