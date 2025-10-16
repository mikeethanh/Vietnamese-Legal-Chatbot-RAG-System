# Vietnamese Legal Corpus Processing with Apache Spark

Công cụ xử lý dữ liệu corpus pháp luật Việt Nam sử dụng Apache Spark để tạo dataset huấn luyện cho RAG chatbot.

## Cấu trúc dữ liệu trên S3

```
s3://legal-datalake/
├── raw/
│   ├── rag_corpus/
│   │   ├── corpus.csv
│   │   ├── data (1).csv
│   │   ├── updated_legal_corpus.csv
│   │   ├── legal_corpus.json
│   │   ├── zalo_corpus.json
│   │   └── vbpl_crawl.json
│   └── finetune_data/
│       └── ...
└── processed/
    └── rag_corpus/
        └── combined.jsonl
```

## 1. Setup environment
```shell
conda create -n dl python=3.9
conda activate dl
pip install -r requirements.txt

# Cài đặt Apache Spark
wget https://archive.apache.org/dist/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz
tar -xzf spark-3.3.2-bin-hadoop3.tgz
export SPARK_HOME=/path/to/spark-3.3.2-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH
```

## 2. Configure AWS S3 credentials
```shell
# Configure AWS credentials
aws configure
# OR set environment variables
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="ap-southeast-1"
```

## 3. Sử dụng

### Phương án 1: Sử dụng shell script (đơn giản nhất)
```bash
cd data_pipeline
chmod +x run_spark_process.sh
./run_spark_process.sh
```

### Phương án 2: Chạy trực tiếp với spark-submit
```bash
# Cho AWS S3 (bucket: legal-datalake)
spark-submit \
  --master local[*] \
  --driver-memory 12g \
  --packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.11.901 \
  utils/spark_process_rag_corpus.py \
    --bucket legal-datalake \
    --raw-prefix raw/rag_corpus \
    --out-prefix processed/rag_corpus \
    --coalesce
```

## 4. Kết quả

Sau khi chạy thành công, dữ liệu được xử lý sẽ có tại:
- `s3://legal-datalake/processed/rag_corpus/combined.jsonl`

Định dạng JSON Lines:
```json
{"id": "unique_hash", "text": "nội dung văn bản pháp luật..."}
{"id": "unique_hash", "text": "nội dung văn bản pháp luật..."}
```
```shell
aws s3 ls
```
## 3. Upload data to S3
We will upload our datasets (finetune_data and rag_corpus) directly to an existing S3 bucket.
```shell
python utils/upload_to_s3.py 
```
