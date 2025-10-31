# Tóm tắt thay đổi: Migration từ Title sang Question

## Ngày thực hiện: 31/10/2025

## Mục tiêu
Thay đổi cấu trúc database và vector store từ `title/content` sang `question/content` để phù hợp với dữ liệu Q&A trong file `train_qa_format.jsonl`.

## Các file đã thay đổi

### 1. **backend/src/models.py**
- Đổi column `title` → `question` (VARCHAR 2000)
- Cập nhật function `insert_document(question, content)`

### 2. **backend/src/brain.py**
- Cập nhật function `gen_doc_prompt()` để sử dụng `question` thay vì `title`
- Format prompt theo dạng Q&A tiếng Việt

### 3. **backend/src/tasks.py**
- Cập nhật function `index_document_v2()` để lưu payload với key `question`
- Thay đổi tham số từ `title` → `question`

### 4. **backend/src/app.py**
- Cập nhật endpoint `/document/create` để nhận `question` thay vì `title`

### 5. **backend/README.md**
- Cập nhật SQL schema cho table `document`

## Files mới tạo

### 6. **backend/src/import_data.py** ⭐
Script Python để import dữ liệu từ `train_qa_format.jsonl` vào Qdrant:
- Đọc file JSONL
- Tạo embeddings cho từng Q&A pair
- Split documents thành chunks nếu cần
- Upload vào Qdrant collection
- Logging và error handling

### 7. **backend/import_data.sh** ⭐
Shell script wrapper để chạy import trong Docker container

### 8. **backend/migration_title_to_question.sql** ⭐
SQL migration script để update database schema

### 9. **backend/MIGRATION_GUIDE.md** ⭐
Hướng dẫn chi tiết cách thực hiện migration và import dữ liệu

## Cấu trúc dữ liệu mới

### Database (MariaDB)
```sql
CREATE TABLE document (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question VARCHAR(2000) NOT NULL,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### Qdrant Vector Payload
```json
{
    "question": "Câu hỏi pháp luật...",
    "content": "Nội dung trả lời...",
    "source": "train_qa_format",
    "doc_id": 0
}
```

## Các bước thực hiện

### Bước 1: Migration Database
```bash
docker exec -it mariadb-tiny mysql -u root -p
USE demo_bot;
ALTER TABLE document CHANGE COLUMN title question VARCHAR(2000) NOT NULL DEFAULT '';
```

### Bước 2: Rebuild Backend
```bash
cd backend
docker compose down
docker compose up -d --build
```

### Bước 3: Import dữ liệu
```bash
# Option 1: Chạy script
docker exec -it chatbot-api bash /usr/src/app/import_data.sh

# Option 2: Chạy trực tiếp Python
docker exec -it chatbot-api python /usr/src/app/src/import_data.py
```

## Kiểm tra kết quả

### Database
```bash
docker exec -it mariadb-tiny mysql -u root -p -e "
USE demo_bot;
DESCRIBE document;
SELECT COUNT(*) FROM document;
"
```

### Qdrant Dashboard
- URL: http://localhost:6333/dashboard
- Check collection: `llm`

### Test API
```bash
curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "user_message": "Hợp đồng lao động là gì?"
  }'
```

## Lợi ích của thay đổi

1. ✅ **Phù hợp với dữ liệu**: Cấu trúc Q&A rõ ràng hơn
2. ✅ **Tăng độ chính xác**: Prompt được tối ưu cho tiếng Việt
3. ✅ **Dữ liệu lớn**: Có thể import ~19K Q&A pairs từ file training
4. ✅ **Dễ maintain**: Code rõ ràng hơn với naming convention phù hợp

## Lưu ý quan trọng

⚠️ **Backup trước khi migration**:
```bash
docker exec mariadb-tiny mysqldump -u root -p demo_bot > backup_$(date +%Y%m%d).sql
```

⚠️ **Mount data_pipeline folder** vào Docker container hoặc copy file:
```yaml
volumes:
  - ../data_pipeline:/usr/src/app/data_pipeline
```

⚠️ **Import có thể mất thời gian**: File có ~19,537 dòng, monitor progress:
```bash
docker logs -f chatbot-api
docker stats chatbot-api
```

## Rollback (nếu cần)

```sql
-- Rollback database
ALTER TABLE document CHANGE COLUMN question title VARCHAR(100) NOT NULL DEFAULT '';

-- Restore từ backup
mysql -u root -p demo_bot < backup_YYYYMMDD.sql
```

## Next Steps (Tùy chọn)

1. [ ] Thêm index cho column `question` để tăng tốc query
2. [ ] Tạo API endpoint để trigger import on-demand
3. [ ] Thêm tính năng update/delete documents
4. [ ] Implement incremental import (chỉ import data mới)
5. [ ] Add monitoring và metrics cho import process

## Contact

Nếu có vấn đề, check:
- MIGRATION_GUIDE.md - Hướng dẫn chi tiết
- docker logs - Logs từ containers
- Qdrant dashboard - Vector database status
