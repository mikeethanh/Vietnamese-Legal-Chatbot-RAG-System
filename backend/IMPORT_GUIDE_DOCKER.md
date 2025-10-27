# Hướng dẫn Import Dữ liệu Finetune vào Qdrant

Script này giúp bạn import dữ liệu từ file `combined_finetune_data.jsonl` vào Qdrant vector database để phục vụ cho hệ thống RAG.

## 📊 Cấu trúc dữ liệu

File JSONL có cấu trúc như sau:
```json
{"question": "Ban chấp hành Hội người cao tuổi Việt Nam là cơ quan như thế nào theo quy định của pháp luật?", "context": "Ban Chấp hành Hội\n1. Ban Chấp hành Hội Người cao tuổi Việt Nam do Đại hội hiệp thương dân chủ bầu ra...", "source": "valid", "id": "valid_17916"}
```

Script sẽ:
- Lấy trường `question` làm **title**
- Lấy trường `context` làm **content**  
- Sử dụng `id` làm document ID trong Qdrant
- Bỏ qua các trường khác (`source`, ...)

---

## 🐳 Cách 1: Sử dụng với Docker (Khuyên dùng)

### Điều kiện tiên quyết

```bash
# Đảm bảo các containers đang chạy
cd backend
docker-compose up -d

# Kiểm tra
docker ps | grep chatbot
docker ps | grep qdrant
```

### Quick Start

```bash
# Di chuyển vào thư mục backend
cd backend

# Import toàn bộ dữ liệu
./import_data.sh

# Hoặc test với 1000 records đầu tiên
./import_data.sh --max-records 1000

# Import theo batch
./import_data.sh --start-from 0 --max-records 50000
./import_data.sh --start-from 50000 --max-records 50000
```

### Các tham số của import_data.sh

```bash
./import_data.sh [options]

Options:
  --file <path>          Đường dẫn file JSONL trong container
                         (mặc định: ../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl)
  --collection <name>    Tên collection Qdrant (mặc định: legal_knowledge)
  --batch-size <num>     Kích thước batch (mặc định: 100)
  --start-from <num>     Bắt đầu từ record thứ N (mặc định: 0)
  --max-records <num>    Số records tối đa (mặc định: tất cả)
  --help                 Hiển thị trợ giúp
```

### Ví dụ thực tế với Docker

```bash
# Test với 100 records trước khi import hết
./import_data.sh --max-records 100 --collection test_collection

# Import batch 50k records đầu tiên
./import_data.sh --start-from 0 --max-records 50000

# Import batch tiếp theo
./import_data.sh --start-from 50000 --max-records 50000

# Import với collection name khác
./import_data.sh --collection legal_finetune_data --max-records 10000
```

### Chạy import trong nền (background)

```bash
# Chạy import trong background với nohup
nohup ./import_data.sh --max-records 100000 > import.log 2>&1 &

# Xem progress
tail -f import.log

# Hoặc dùng docker logs
docker logs -f chatbot-worker
```

### Cách thay thế: Chạy trực tiếp docker exec

```bash
# Vào container
docker exec -it chatbot-worker bash

# Trong container, chạy:
cd /usr/src/app/src
python import_finetune_data.py \
  --file /usr/src/app/../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl \
  --max-records 1000

# Exit container
exit
```

---

## 💻 Cách 2: Chạy trực tiếp (Không dùng Docker)

Nếu bạn **KHÔNG** dùng Docker và chạy Python trực tiếp trên máy:

### Cài đặt

```bash
# Cài dependencies
cd backend
pip install -r requirements.txt

# Optional: Cài tqdm để có progress bar đẹp hơn
pip install tqdm
```

### Đảm bảo services đang chạy

```bash
# Qdrant phải đang chạy (port 6333)
# Redis/Valkey phải đang chạy (port 6379)

# Nếu dùng Docker cho database:
cd database
docker-compose up -d

cd ../backend
docker-compose up -d qdrant-db valkey-db
```

### Chạy script

```bash
cd backend/src

# Import toàn bộ
python import_finetune_data.py \
  --file ../../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl

# Import với collection cụ thể
python import_finetune_data.py \
  --file ../../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl \
  --collection legal_finetune_data

# Test với 1000 records
python import_finetune_data.py \
  --file ../../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl \
  --max-records 1000

# Import theo batch
python import_finetune_data.py \
  --file ../../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl \
  --start-from 0 \
  --max-records 50000
```

---

## 📝 Các tham số chi tiết

| Tham số | Mô tả | Mặc định | Bắt buộc |
|---------|-------|----------|----------|
| `--file` | Đường dẫn đến file JSONL | - | ✅ |
| `--collection` | Tên collection trong Qdrant | `legal_knowledge` | ❌ |
| `--batch-size` | Số records xử lý trước khi nghỉ | 100 | ❌ |
| `--start-from` | Bỏ qua N records đầu tiên | 0 | ❌ |
| `--max-records` | Số records tối đa muốn import | Tất cả | ❌ |

---

## ⚠️ Lưu ý quan trọng

### 1. Thời gian xử lý
Với **608,689 records**:
- Mỗi record cần ~0.1-0.5 giây (tùy thuộc embedding model và tốc độ)
- **Ước tính: 17-84 giờ** để import toàn bộ
- **Khuyến nghị**: 
  - Test với 100-1000 records trước
  - Chia batch 50k-100k records
  - Chạy qua đêm hoặc trong background

### 2. Chiến lược import cho dữ liệu lớn

#### Chiến lược A: Import từng batch tuần tự
```bash
# Batch 1: Records 0-100000
./import_data.sh --start-from 0 --max-records 100000

# Batch 2: Records 100000-200000  
./import_data.sh --start-from 100000 --max-records 100000

# Batch 3: Records 200000-300000
./import_data.sh --start-from 200000 --max-records 100000

# ... tiếp tục
```

#### Chiến lược B: Script tự động chia batch
```bash
# Tạo script tự động
cat > auto_import.sh << 'EOF'
#!/bin/bash
TOTAL_RECORDS=608689
BATCH_SIZE=50000
START=0

while [ $START -lt $TOTAL_RECORDS ]; do
  echo "Importing batch starting at $START..."
  ./import_data.sh --start-from $START --max-records $BATCH_SIZE
  START=$((START + BATCH_SIZE))
  echo "Sleeping 10 seconds before next batch..."
  sleep 10
done
EOF

chmod +x auto_import.sh
./auto_import.sh
```

### 3. Monitoring

#### Xem logs realtime (Docker)
```bash
# Theo dõi logs của worker
docker logs -f chatbot-worker

# Hoặc nếu chạy qua import_data.sh
tail -f import.log
```

#### Kiểm tra số lượng trong Qdrant
```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
collection_info = client.get_collection("legal_knowledge")
print(f"Total points: {collection_info.points_count}")
```

### 4. Recovery khi bị gián đoạn

Nếu script dừng giữa chừng:
1. Kiểm tra log để xem đã import đến record nào
2. Sử dụng `--start-from` để tiếp tục

```bash
# Ví dụ: Script dừng ở record 50000
./import_data.sh --start-from 50000
```

### 5. Tối ưu hiệu năng

#### Trong Docker
- Tăng resources cho container trong `docker-compose.yml`:
```yaml
chatbot-worker:
  # ... existing config
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 4G
```

#### Tăng tốc embedding
- Nếu dùng local embedding model → Sử dụng GPU
- Nếu dùng API embedding (OpenAI, Cohere) → Tăng rate limit

---

## 🧪 Testing & Validation

### Test trước khi import hết

```bash
# Test với 100 records vào collection test
./import_data.sh --max-records 100 --collection test_legal

# Kiểm tra kết quả
python << EOF
from vectorize import search_vector
from brain import get_embedding

query = "Quy định về người cao tuổi"
vector = get_embedding(query)
results = search_vector("test_legal", vector, limit=5)
print("Test results:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.get('title', 'N/A')[:100]}...")
EOF
```

### Validate sau khi import

```bash
# Vào container Python
docker exec -it chatbot-worker python

# Trong Python shell:
from vectorize import search_vector
from brain import get_embedding
from qdrant_client import QdrantClient

# Kiểm tra số lượng
client = QdrantClient(url="http://qdrant-db:6333")
info = client.get_collection("legal_knowledge")
print(f"Total documents: {info.points_count}")

# Test search
query = "Thừa kế tài sản"
vector = get_embedding(query)
results = search_vector("legal_knowledge", vector, limit=3)

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Title: {doc.get('title', 'N/A')}")
    print(f"Content preview: {doc.get('content', 'N/A')[:200]}...")
```

---

## 📊 Output mẫu

### Khi chạy với Docker

```
Checking Docker containers...
✓ All containers are running

Import Configuration:
  Container: chatbot-worker
  File: /usr/src/app/../data_pipeline/data/process_data/finetune_data/combined_finetune_data.jsonl
  Collection: legal_knowledge
  Batch size: 100
  Start from: 0
  Max records: 1000

Continue? (y/n) y
Starting import...

2025-10-27 10:00:00 - INFO - Starting import process...
2025-10-27 10:00:00 - INFO - Counting total records...
2025-10-27 10:00:05 - INFO - Total records in file: 608689
2025-10-27 10:00:05 - INFO - Starting import from record 0, will process 1000 records
Importing: 100%|████████████████| 1000/1000 [05:23<00:00, success=998, errors=0, skipped=2]
============================================================
Import Summary:
  Total processed: 1000
  Successfully imported: 998
  Errors: 0
  Skipped (missing data): 2
============================================================

✓ Import completed!
```

---

## 🔧 Troubleshooting

### Lỗi: "chatbot-worker container is not running"
```bash
cd backend
docker-compose up -d
```

### Lỗi: "qdrant-db container is not running"
```bash
cd database
docker-compose up -d
# hoặc
cd backend
docker-compose up -d qdrant-db
```

### Lỗi: "Collection already exists"
✅ Không sao, script sẽ tự động sử dụng collection hiện có.

### Lỗi: "Connection refused to Qdrant"
Kiểm tra Qdrant có đang chạy không:
```bash
curl http://localhost:6333/collections
# Hoặc từ trong container:
docker exec chatbot-worker curl http://qdrant-db:6333/collections
```

### Lỗi: "No embedding model configured"
Kiểm tra file `.env` trong `backend/`:
```bash
# Cần có các biến này
OPENAI_API_KEY=your_key
# Hoặc
EMBEDDING_API_URL=your_custom_embedding_url
```

### Script chạy quá chậm
- Kiểm tra embedding model (local vs API)
- Tăng resources cho Docker
- Giảm `--batch-size` hoặc chia nhỏ hơn

---

## ✅ Kết quả sau khi import thành công

Dữ liệu sẽ có trong Qdrant và có thể được sử dụng cho:

1. ✅ **RAG queries** - Trả lời câu hỏi pháp luật qua `bot_rag_answer_message()`
2. ✅ **Vector search** - Tìm kiếm ngữ nghĩa qua `search_vector()`
3. ✅ **Multi-query retrieval** - Tìm kiếm với nhiều query variations
4. ✅ **Reranking** - Sắp xếp lại kết quả theo độ liên quan
5. ✅ **Conversational AI** - Chat với context từ corpus pháp luật

---

**Ghi chú**: Script này sử dụng hàm `index_document_v2()` có sẵn trong hệ thống, đảm bảo tính nhất quán với RAG pipeline đang chạy. Mỗi document sẽ được:
1. Split thành các chunks nhỏ hơn (nếu quá dài)
2. Embedding bằng model hiện tại
3. Lưu vào Qdrant với metadata (title, content)
