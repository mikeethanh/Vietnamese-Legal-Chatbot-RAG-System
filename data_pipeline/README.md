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

