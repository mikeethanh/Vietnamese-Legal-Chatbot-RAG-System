# Quick Start Guide - Hướng Dẫn Triển Khai

## 📦 Cài Đặt & Khởi Động

### 1. Chuẩn Bị Environment Variables

Tạo file `.env` trong thư mục `backend/`:

```bash
# OpenAI API (required)
OPENAI_API_KEY=sk-your-key-here

# Cohere API for reranking (required)
COHERE_API_KEY=your-cohere-key

# Google Custom Search (optional, for web_search route)
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-cse-id

# Database
MYSQL_USER=root
MYSQL_ROOT_PASSWORD=your-password
MYSQL_HOST=mariadb-db
MYSQL_PORT=3306

# Redis/Celery
CELERY_BROKER_URL=redis://valkey-db:6379
CELERY_RESULT_BACKEND=redis://valkey-db:6379
```

### 2. Khởi Động Services

```bash
cd backend
docker-compose up -d --build
```

### 3. Kiểm Tra Services

```bash
# Check containers
docker ps

# Check API logs
docker logs -f chatbot-api

# Check worker logs
docker logs -f chatbot-worker

# Check Qdrant
docker logs -f qdrant-db
```

### 4. Test API

#### Test 1: Simple Legal Query
```bash
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "bot_id": "botLawyer",
    "user_id": "test_user",
    "user_message": "Thủ tục ly hôn như thế nào?",
    "sync_request": true
  }'
```

#### Test 2: Follow-up Question
```bash
# First question
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "bot_id": "botLawyer",
    "user_id": "test_user2",
    "user_message": "Thủ tục thành lập công ty TNHH như thế nào?",
    "sync_request": true
  }'

# Follow-up
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "bot_id": "botLawyer",
    "user_id": "test_user2",
    "user_message": "Vậy chi phí là bao nhiêu?",
    "sync_request": true
  }'
```

#### Test 3: Async Request
```bash
# Send request
curl -X POST http://localhost:8002/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "bot_id": "botLawyer",
    "user_id": "test_user3",
    "user_message": "Quyền lợi của người lao động khi bị sa thải?",
    "sync_request": false
  }'

# Get result (replace {task_id} with returned task_id)
curl http://localhost:8002/chat/complete/{task_id}
```

---

## 🗄️ Database Setup

### Create Database & Tables

```bash
# Connect to database
docker exec -it mariadb-tiny bash
mysql -u root -p

# Create database
CREATE DATABASE demo_bot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE demo_bot;

# Create tables (already in init.sql)
```

### Create Vector Collection

```bash
curl -X POST http://localhost:8002/collection/create \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "llm"
  }'
```

### Index Document

```bash
curl -X POST http://localhost:8002/document/create \
  -H "Content-Type: application/json" \
  -d '{
    "id": "doc1",
    "title": "Luật Doanh Nghiệp 2020",
    "content": "Nội dung văn bản pháp luật..."
  }'
```

---

## 🔍 Monitoring & Logs

### View Logs

```bash
# API logs
docker logs -f chatbot-api

# Worker logs (Celery)
docker logs -f chatbot-worker

# Database logs
docker logs -f mariadb-tiny

# Vector DB logs
docker logs -f qdrant-db

# Redis logs
docker logs -f valkey-db
```

### Check Qdrant Dashboard

Open browser: http://localhost:6333/dashboard

### Check Redis

```bash
docker exec -it valkey-db redis-cli
> KEYS *
> GET botLawyer.test_user
```

---

## 🧪 Testing Features

### Test 1: Semantic Chunking

```python
from splitter import split_document

text = """
Điều 1. Phạm vi điều chỉnh
Luật này quy định về...

Điều 2. Đối tượng áp dụng
Luật này áp dụng cho...
"""

# Semantic splitting (recommended)
nodes = split_document(text, use_semantic=True)
print(f"Generated {len(nodes)} semantic chunks")

# Token-based splitting (fallback)
nodes = split_document(text, use_semantic=False)
print(f"Generated {len(nodes)} token chunks")
```

### Test 2: Query Rewriting

```python
from query_rewriter import rewrite_query_to_multi_queries

query = "Thủ tục ly hôn như thế nào?"
queries = rewrite_query_to_multi_queries(query, num_queries=3)

print("Original:", query)
print("Variations:")
for i, q in enumerate(queries, 1):
    print(f"  {i}. {q}")
```

### Test 3: Routing

```python
from brain import detect_route

# Legal query
route = detect_route([], "Thủ tục thành lập công ty TNHH?")
print(f"Route: {route}")  # Should be: legal_rag

# Web search query
route = detect_route([], "Luật giao thông mới nhất 2025?")
print(f"Route: {route}")  # Should be: web_search

# General chat
route = detect_route([], "Xin chào, bạn là ai?")
print(f"Route: {route}")  # Should be: general_chat
```

---

## 🐛 Troubleshooting

### Issue: Import errors in editor

**Symptom:**
```
Import "llama_index.core" could not be resolved
Import "openai" could not be resolved
```

**Solution:**
- These are editor warnings only
- Code runs fine in Docker container
- To fix in editor:
  ```bash
  pip install -r requirements.txt
  ```

### Issue: API not responding

**Check:**
```bash
# Check if container is running
docker ps | grep chatbot-api

# Check logs
docker logs chatbot-api

# Restart
docker-compose restart chatbot-api
```

### Issue: Celery worker not processing

**Check:**
```bash
# Check worker logs
docker logs chatbot-worker

# Check Redis connection
docker exec -it valkey-db redis-cli ping

# Restart worker
docker-compose restart chatbot-worker
```

### Issue: Vector search returns empty

**Check:**
```bash
# Check Qdrant
curl http://localhost:6333/collections

# Check if collection exists
curl http://localhost:6333/collections/llm

# Recreate collection
curl -X POST http://localhost:8002/collection/create \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "llm"}'
```

### Issue: Reranking fails

**Check:**
- COHERE_API_KEY is set
- Cohere API has credits
- Check logs for error message

---

## 📊 Performance Tuning

### 1. Adjust Multi-Query Count

In `tasks.py`:
```python
# More queries = better coverage, slower
query_variations = rewrite_query_to_multi_queries(standalone_question, num_queries=5)

# Fewer queries = faster, less coverage
query_variations = rewrite_query_to_multi_queries(standalone_question, num_queries=2)
```

### 2. Adjust Retrieval Count

```python
# More docs = better coverage, slower reranking
retrieved_docs = retrieve_with_multi_query(query_variations, top_k=6)

# Fewer docs = faster
retrieved_docs = retrieve_with_multi_query(query_variations, top_k=3)
```

### 3. Adjust Reranking Count

```python
# More docs = better context, larger prompt
ranked_docs = rerank_documents(retrieved_docs, standalone_question, top_n=7)

# Fewer docs = faster, cheaper
ranked_docs = rerank_documents(retrieved_docs, standalone_question, top_n=3)
```

### 4. Enable Caching

Add Redis caching for queries:
```python
import hashlib

def get_cache_key(query):
    return f"cache:{hashlib.md5(query.encode()).hexdigest()}"

# Check cache before retrieval
cache_key = get_cache_key(query)
cached = redis_client.get(cache_key)
if cached:
    return json.loads(cached)

# ... do retrieval ...

# Cache result
redis_client.setex(cache_key, 3600, json.dumps(result))
```

---

## 🔐 Security Notes

1. **API Keys:** Never commit to git
2. **Database:** Use strong passwords
3. **Redis:** Enable authentication in production
4. **API:** Add rate limiting
5. **CORS:** Configure allowed origins

---

## 🚀 Production Checklist

- [ ] Set strong passwords
- [ ] Enable Redis authentication
- [ ] Configure CORS properly
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Enable API rate limiting
- [ ] Set up log aggregation
- [ ] Configure backup for database
- [ ] Configure backup for Qdrant
- [ ] Set up health checks
- [ ] Configure autoscaling for workers

---

## 📞 Support

For issues or questions:
1. Check logs first
2. Review documentation in IMPROVEMENTS.md
3. Check troubleshooting section
4. Contact development team

---

**Happy Coding! 🎉**
