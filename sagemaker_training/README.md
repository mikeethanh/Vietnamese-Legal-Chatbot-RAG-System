# Vietnamese Legal Embedding Model - SageMaker Training

Hướng dẫn training model embedding cho Vietnamese Legal documents sử dụng AWS SageMaker và deploying embeddings vào Qdrant vector database.

## 🎯 Mục tiêu

1. Training custom embedding model trên Vietnamese legal corpus
2. Deploy model thành SageMaker endpoint
3. Generate embeddings cho toàn bộ corpus data  
4. Lưu trữ embeddings vào Qdrant collection

## 🏗️ Cấu trúc Project

```
sagemaker_training/
├── scripts/
│   ├── train_embedding.py      # Script training chính
│   ├── inference.py           # Script inference cho endpoint
│   └── upload_to_s3.py       # Upload data lên S3
├── notebooks/
│   └── train_legal_embedding_model.ipynb  # Notebook chính
├── requirements.txt           # Dependencies
└── README.md                 # File này
```

## 🚀 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu

```bash
# Upload corpus data lên S3
python scripts/upload_to_s3.py \
  --local-file ../data_pipeline/data/process_data/rag_corpus/merged_corpus.jsonl \
  --bucket legal-datalake \
  --s3-key processed/rag_corpus/merged_corpus.jsonl
```

### Bước 2: Chạy Jupyter Notebook

```bash
# Khởi động Jupyter
jupyter notebook notebooks/train_legal_embedding_model.ipynb
```

### Bước 3: Thực hiện theo notebook

1. **Setup Environment**: Import libraries và configure AWS
2. **Prepare Data**: Verify dữ liệu trên S3
3. **Configure Training**: Setup SageMaker estimator
4. **Launch Training**: Bắt đầu training job
5. **Deploy Model**: Tạo SageMaker endpoint
6. **Setup Qdrant**: Kết nối với Qdrant database
7. **Generate Embeddings**: Sử dụng endpoint để tạo embeddings
8. **Insert to Qdrant**: Lưu embeddings vào collection

## ⚙️ Configuration

### AWS Requirements

```bash
# AWS CLI configured với credentials
aws configure

# Hoặc set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=ap-southeast-1
```

### SageMaker IAM Role

IAM role cần có permissions:
- `SageMakerFullAccess`
- `S3FullAccess` (hoặc specific bucket access)
- `IAMPassRole`

### Qdrant Setup

```bash
# Chạy Qdrant với Docker
docker run -p 6333:6333 qdrant/qdrant
```

## 💰 Chi phí ước tính

### SageMaker Training
- **Instance**: `ml.g4dn.xlarge` (~$1.5/hour)
- **Training time**: 2-4 hours
- **Estimated cost**: $3-6

### SageMaker Inference
- **Instance**: `ml.m5.large` (~$0.1/hour)  
- **Usage time**: Tùy theo nhu cầu
- **⚠️ Nhớ xóa endpoint sau khi sử dụng!**

## 🔧 Troubleshooting

### Lỗi thường gặp

1. **IAM Permission Denied**
   ```
   Solution: Kiểm tra IAM role có đủ quyền SageMaker và S3
   ```

2. **Training Job Failed**
   ```
   Solution: Kiểm tra CloudWatch logs để xem error details
   ```

3. **Qdrant Connection Failed**
   ```
   Solution: Đảm bảo Qdrant server đang chạy trên đúng host:port
   ```

4. **Out of Memory during Training**
   ```
   Solution: Giảm batch_size hoặc dùng instance type lớn hơn
   ```

### Debug Commands

```bash
# Kiểm tra S3 data
aws s3 ls s3://legal-datalake/processed/rag_corpus/

# Kiểm tra SageMaker training jobs
aws sagemaker list-training-jobs --status-equals InProgress

# Kiểm tra SageMaker endpoints
aws sagemaker list-endpoints
```

## 📊 Model Performance

### Base Model
- **Architecture**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Vector dimension**: 384
- **Languages**: Multilingual (including Vietnamese)

### Training Configuration
- **Epochs**: 3
- **Batch size**: 16
- **Learning rate**: 2e-5
- **Max sequence length**: 512

### Expected Results
- Better performance on Vietnamese legal domain
- Improved semantic similarity for legal concepts
- Domain-specific embeddings for RAG system

## 🔄 Next Steps sau khi hoàn thành

1. **Evaluate model performance**: So sánh với base model
2. **Integrate with RAG system**: Update backend để sử dụng new embeddings
3. **Monitor performance**: Track search quality và response time
4. **Fine-tune further**: Nếu cần thiết, train thêm với more data

## 📞 Support

Nếu gặp vấn đề, kiểm tra:
1. CloudWatch logs cho SageMaker training jobs
2. SageMaker Console cho job status
3. Qdrant logs nếu có connection issues

## ⚠️ Lưu ý quan trọng

- **Xóa SageMaker endpoint** sau khi sử dụng để tránh phí
- **Monitor training cost** trong AWS Cost Explorer
- **Backup model artifacts** trong S3 bucket
- **Test thoroughly** trước khi deploy production