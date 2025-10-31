# Data Pipeline cho Vietnamese Legal Chatbot

Pipeline xử lý dữ liệu cho hệ thống chatbot tư vấn pháp luật Việt Nam, bao gồm các công cụ xử lý dữ liệu RAG và chuẩn bị dữ liệu fine-tuning.

## 🎯 Mục tiêu

Data pipeline này phục vụ cho:
- **Xử lý dữ liệu RAG**: Chuẩn bị corpus pháp luật cho việc tìm kiếm ngữ nghĩa
- **Chuẩn bị dữ liệu fine-tuning**: Tạo datasets cho việc fine-tune mô hình ngôn ngữ
- **Tải xuống và xử lý dữ liệu**: Tự động hóa quá trình thu thập và làm sạch dữ liệu

## 📁 Cấu trúc thư mục

```
data_pipeline/
├── data/                           # Dữ liệu đầu vào và đầu ra
│   ├── embed/                      # Dữ liệu cho embedding và RAG
│   │   └── law_vi.jsonl           # Corpus pháp luật Việt Nam
│   ├── finetune_data/             # Dữ liệu fine-tuning tập 1
│   │   ├── metadata.json          # Metadata của dataset
│   │   ├── train_qa_format.jsonl  # Dữ liệu train định dạng Q&A
│   │   ├── test_qa_format.jsonl   # Dữ liệu test định dạng Q&A
│   │   ├── train_conversation_format.jsonl  # Định dạng hội thoại
│   │   └── train_instruction_format.jsonl   # Định dạng instruction
│   ├── finetune_data2/            # Dữ liệu fine-tuning tập 2 (ViLQA)
│   │   ├── vilqa_metadata.json
│   │   ├── vilqa_qa_format.jsonl
│   │   ├── vilqa_conversation_format.jsonl
│   │   └── vilqa_instruction_format.jsonl
│   ├── finetune_data3/            # Dữ liệu fine-tuning tập 3
│   └── finetune_rag/              # Dữ liệu fine-tuning cho RAG
├── utils/                          # Công cụ xử lý dữ liệu
│   ├── download_embed_data.ipynb   # Tải dữ liệu embedding
│   ├── process_finetune_data.ipynb # Xử lý dữ liệu fine-tuning
│   ├── process_finetune_data_2.ipynb
│   └── process_finetune_data_3.ipynb
├── requirements.txt               # Dependencies Python
└── README.md                     # Tài liệu này
```

## 🛠️ Công nghệ sử dụng

- **Apache Spark**: Xử lý dữ liệu quy mô lớn
- **Pandas**: Thao tác và phân tích dữ liệu
- **MinIO/S3**: Lưu trữ đám mây
- **Jupyter Notebooks**: Môi trường phát triển tương tác
- **PyDeequ**: Đảm bảo chất lượng dữ liệu

## 🚀 Cài đặt và sử dụng

### 1. Chuẩn bị môi trường

```bash
cd data_pipeline

# Cài đặt dependencies
pip install -r requirements.txt

# Tạo thư mục dữ liệu (nếu chưa có)
mkdir -p data/{embed,finetune_data,finetune_data2,finetune_data3,finetune_rag}
```

### 2. Tải dữ liệu embedding

```bash
# Mở Jupyter notebook để tải dữ liệu
jupyter notebook utils/download_embed_data.ipynb
```

Notebook này sẽ:
- Tải corpus pháp luật Việt Nam từ Hugging Face
- Lưu dữ liệu vào `data/embed/law_vi.jsonl`
- Thống kê số lượng và chất lượng dữ liệu

### 3. Xử lý dữ liệu fine-tuning

#### Tập dữ liệu 1 (Cơ bản)
```bash
jupyter notebook utils/process_finetune_data.ipynb
```

#### Tập dữ liệu 2 (ViLQA)
```bash
jupyter notebook utils/process_finetune_data_2.ipynb
```

#### Tập dữ liệu 3 (Mở rộng)
```bash
jupyter notebook utils/process_finetune_data_3.ipynb
```

## 📊 Định dạng dữ liệu

### 1. Dữ liệu RAG (law_vi.jsonl)
```json
{
  "text": "Điều 1. Phạm vi điều chỉnh...",
  "metadata": {
    "source": "Luật Dân sự 2015",
    "article": "Điều 1",
    "chapter": "Chương I"
  }
}
```

### 2. Dữ liệu Fine-tuning Q&A
```json
{
  "question": "Luật Dân sự quy định gì về quyền sở hữu?",
  "answer": "Theo Luật Dân sự 2015, quyền sở hữu là...",
  "context": "Điều 123. Quyền sở hữu..."
}
```

### 3. Dữ liệu Conversation Format
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Luật Dân sự quy định gì về quyền sở hữu?"
    },
    {
      "from": "assistant",
      "value": "Theo Luật Dân sự 2015, quyền sở hữu là..."
    }
  ]
}
```

### 4. Dữ liệu Instruction Format
```json
{
  "instruction": "Hãy giải thích quy định của Luật Dân sự về quyền sở hữu.",
  "input": "",
  "output": "Theo Luật Dân sự 2015, quyền sở hữu là..."
}
```

## 🔧 Cấu hình

### Biến môi trường (.env)
```bash
# AWS S3 Configuration (nếu sử dụng)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1
S3_BUCKET=your-legal-data-bucket

# MinIO Configuration (nếu sử dụng)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=legal-data

# Spark Configuration
SPARK_MASTER=local[*]
SPARK_EXECUTOR_MEMORY=4g
SPARK_DRIVER_MEMORY=2g
```

## 📈 Thống kê dữ liệu

### Dữ liệu RAG
- **Tổng số documents**: ~1.9M văn bản pháp luật
- **Kích thước**: ~2.5GB
- **Nguồn**: Corpus pháp luật Việt Nam từ Zalo AI Challenge

### Dữ liệu Fine-tuning
- **Tập 1**: ~50K cặp câu hỏi-trả lời cơ bản
- **Tập 2 (ViLQA)**: ~100K cặp Q&A chuyên sâu
- **Tập 3**: ~75K cặp conversation format

## 🧪 Đảm bảo chất lượng dữ liệu

### Validation checks
- Kiểm tra định dạng JSON
- Validation độ dài text
- Loại bỏ duplicates
- Kiểm tra encoding UTF-8
- Validation metadata

### Data quality metrics
```python
# Ví dụ sử dụng PyDeequ cho quality checks
from pydeequ import Check, VerificationSuite
from pydeequ.analyzers import Size, Completeness

check = Check(spark, CheckLevel.Warning, "Dataset Quality Check") \
    .hasSize(lambda x: x >= 1000000) \
    .isComplete("text") \
    .containsURL("text", lambda x: x <= 0.1)
```

## 🔄 Pipeline tự động

### Chạy pipeline hoàn chỉnh
```bash
# Script tự động xử lý tất cả dữ liệu
python scripts/full_pipeline.py

# Hoặc chạy từng bước
python scripts/download_data.py
python scripts/process_embed_data.py
python scripts/process_finetune_data.py
```

### Lập lịch xử lý (Cron job)
```bash
# Cập nhật dữ liệu hàng tuần
0 2 * * 0 cd /path/to/data_pipeline && python scripts/weekly_update.py
```

## 🐳 Docker Support

### Chạy pipeline trong container
```bash
# Build image
docker build -t legal-data-pipeline .

# Chạy xử lý dữ liệu
docker run -v $(pwd)/data:/app/data legal-data-pipeline python scripts/process_data.py
```

### Docker Compose
```yaml
version: '3.8'
services:
  data-pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - SPARK_MASTER=local[*]
```

## 📝 Logging và Monitoring

### Cấu hình logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics thu thập
- Thời gian xử lý từng bước
- Số lượng records được xử lý
- Tỷ lệ lỗi và warnings
- Sử dụng bộ nhớ và CPU

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/new-data-source`)
3. Commit changes (`git commit -am 'Add new data source'`)
4. Push to branch (`git push origin feature/new-data-source`)
5. Tạo Pull Request

## 📄 License

Project này được phân phối dưới MIT License - xem file [LICENSE](../LICENSE) để biết thêm chi tiết.

## 🆘 Hỗ trợ

- **Issues**: [GitHub Issues](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/issues)
- **Email**: mikeethanh@example.com
- **Documentation**: [Wiki](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/wiki)

---

**Lưu ý**: Pipeline này được thiết kế cho mục đích nghiên cứu và giáo dục. Dữ liệu pháp luật cần được xác minh với các chuyên gia pháp lý.