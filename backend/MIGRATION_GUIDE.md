# Hướng dẫn Migration và Import Data

## Tổng quan thay đổi

Đã thực hiện các thay đổi sau:

1. **Database Schema**: Đổi column `title` → `question` trong table `document`
2. **Code Changes**: Cập nhật tất cả references từ `title` → `question`
3. **Import Script**: Tạo script để import dữ liệu từ `train_qa_format.jsonl` vào Qdrant

## Bước 1: Migration Database

### Cách 1: Sử dụng SQL script

```bash
# Truy cập vào MariaDB container
docker exec -it mariadb-tiny bash

# Chạy MySQL client
mysql -u root -p

# Chạy migration script
source /usr/src/app/migration_title_to_question.sql
```

### Cách 2: Chạy trực tiếp

```bash
docker exec -it mariadb-tiny mysql -u root -p -e "
USE demo_bot;
ALTER TABLE document CHANGE COLUMN title question VARCHAR(2000) NOT NULL DEFAULT '';
DESCRIBE document;
"
```

## Bước 2: Rebuild và Deploy Backend

```bash
cd backend

# Stop các services hiện tại
docker compose down

# Rebuild với code mới
docker compose up -d --build

# Check logs
docker logs -f chatbot-api
docker logs -f chatbot-worker
```

## Bước 3: Import dữ liệu từ train_qa_format.jsonl vào Qdrant

### Cách 1: Chạy script trong container

```bash
# Copy dữ liệu vào container (nếu chưa mount)
docker cp /home/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/data_pipeline/data/finetune_data/train_qa_format.jsonl chatbot-api:/usr/src/app/data/

# Chạy import script
docker exec -it chatbot-api bash /usr/src/app/import_data.sh
```

### Cách 2: Chạy trực tiếp Python script

```bash
docker exec -it chatbot-api python /usr/src/app/src/import_data.py --data-file /usr/src/app/data/train_qa_format.jsonl --collection llm --batch-size 100
```

### Cách 3: Sử dụng API endpoint

Sau khi có script import, bạn cũng có thể tạo endpoint API:

```python
# Thêm vào app.py
@app.post("/data/import")
async def import_qa_data_endpoint():
    from import_data import import_qa_data
    success = import_qa_data()
    return {"success": success}
```

Sau đó gọi:
```bash
curl -X POST http://localhost:8000/data/import
```

## Bước 4: Kiểm tra kết quả

### Kiểm tra Database

```bash
docker exec -it mariadb-tiny mysql -u root -p -e "
USE demo_bot;
DESCRIBE document;
SELECT COUNT(*) FROM document;
SELECT id, question, LEFT(content, 100) FROM document LIMIT 5;
"
```

### Kiểm tra Qdrant

Truy cập Qdrant Dashboard:
- URL: http://localhost:6333/dashboard
- Kiểm tra collection `llm`
- Xem số lượng vectors đã được index

### Test API

```bash
# Test chatbot với câu hỏi pháp luật
curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "bot_id": "botLawyer",
    "user_id": "test_user",
    "user_message": "Trong Bộ luật Hình sự thì bao nhiêu tuổi được xem là người già?",
    "sync_request": true
  }'
```

## Cấu trúc dữ liệu mới

### Database Table: document

```sql
CREATE TABLE document (
    id INT NOT NULL AUTO_INCREMENT,
    question VARCHAR(2000) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);
```

### Qdrant Vector Payload

```json
{
    "question": "Trong Bộ luật Hình sự thì bao nhiêu tuổi...",
    "content": "Người cao tuổi, người già...",
    "source": "train_qa_format",
    "doc_id": 0
}
```

## Lưu ý quan trọng

1. **Backup dữ liệu**: Trước khi chạy migration, hãy backup database:
   ```bash
   docker exec mariadb-tiny mysqldump -u root -p demo_bot > backup.sql
   ```

2. **Mount data_pipeline**: Đảm bảo folder `data_pipeline` được mount vào container hoặc copy file vào:
   ```yaml
   # Trong docker-compose.yml
   volumes:
     - .:/usr/src/app/
     - ../data_pipeline:/usr/src/app/data_pipeline
   ```

3. **Memory & Performance**: File `train_qa_format.jsonl` có ~19,537 dòng. Import sẽ mất thời gian. Monitor:
   ```bash
   docker stats chatbot-api
   docker logs -f chatbot-api
   ```

4. **Qdrant Collection**: Nếu collection `llm` đã tồn tại với schema cũ, cân nhắc xóa và tạo lại:
   ```bash
   curl -X DELETE http://localhost:6333/collections/llm
   ```

## Troubleshooting

### Lỗi: File not found

```bash
# Check file path
docker exec -it chatbot-api ls -la /usr/src/app/../data_pipeline/data/finetune_data/

# Adjust path trong import_data.py nếu cần
```

### Lỗi: Connection refused to Qdrant

```bash
# Check Qdrant service
docker ps | grep qdrant
docker logs qdrant-db

# Check network
docker exec -it chatbot-api ping qdrant-db
```

### Import quá chậm

```bash
# Giảm batch size hoặc limit số records để test
docker exec -it chatbot-api python /usr/src/app/src/import_data.py \
    --batch-size 50 \
    --limit 1000  # Nếu thêm tham số này
```

## Files đã thay đổi

1. `backend/src/models.py` - Document model
2. `backend/src/brain.py` - gen_doc_prompt function
3. `backend/src/tasks.py` - index_document_v2 function
4. `backend/src/app.py` - create_document endpoint
5. `backend/README.md` - SQL schema documentation
6. `backend/src/import_data.py` - **New** Import script
7. `backend/import_data.sh` - **New** Shell script wrapper
8. `backend/migration_title_to_question.sql` - **New** Migration SQL

## Tài liệu tham khảo

- Qdrant API: https://qdrant.tech/documentation/
- SQLAlchemy: https://docs.sqlalchemy.org/
- FastAPI: https://fastapi.tiangolo.com/
