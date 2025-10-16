# Data Pipeline for Vietnamese Legal Chatbot RAG System

This data pipeline processes Vietnamese legal documents and prepares them for the RAG (Retrieval-Augmented Generation) system using Apache Spark.

## Overview

The pipeline processes various legal document formats (CSV, JSON, JSONL) and converts them into a unified format suitable for embedding generation and vector database storage.

## Features

- **Multi-format Support**: Processes CSV, JSON, and JSONL files containing legal documents
- **Data Cleaning**: Removes duplicates, empty content, and invalid entries
- **Text Standardization**: Normalizes Vietnamese text for consistent processing
- **Scalable Processing**: Uses Apache Spark for handling large datasets
- **Flexible Output**: Generates JSONL format optimized for RAG systems

## Prerequisites

### System Requirements
- Python 3.8+
- Apache Spark 3.4+
- Java 8 or 11
- Minimum 8GB RAM (12GB recommended)

### Python Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n legal-rag python=3.9
conda activate legal-rag
pip install -r requirements.txt

# Install Apache Spark
wget https://archive.apache.org/dist/spark/spark-3.4.0/spark-3.4.0-bin-hadoop3.tgz
tar -xzf spark-3.4.0-bin-hadoop3.tgz
export SPARK_HOME=/path/to/spark-3.4.0-bin-hadoop3
export PATH=$SPARK_HOME/bin:$PATH

# Copy environment template (optional)
cp utils/.env.example utils/.env
```

### 2. Prepare Your Data
Place your legal documents in the following structure:
```
data_pipeline/
├── data/
│   └── rag_corpus/
│       ├── corpus.csv
│       ├── data (1).csv
│       ├── updated_legal_corpus.csv
│       ├── legal_corpus.json
│       ├── zalo_corpus.json
│       └── vbpl_crawl.json
└── processed/
    └── rag_corpus/
        └── combined.jsonl
```

### 3. Run Processing Pipeline

#### Option 1: Using automation script (recommended)
```bash
cd data_pipeline
chmod +x run_spark_process.sh
./run_spark_process.sh
```

#### Option 2: Manual execution with spark-submit
```bash
spark-submit \
  --master local[*] \
  --driver-memory 12g \
  utils/spark_process_rag_corpus.py \
  --raw-prefix data/rag_corpus \
  --out-prefix processed/rag_corpus \
  --coalesce
```

## File Formats and Structure

### Input Files

#### CSV Files
- `corpus.csv`: Contains legal documents with `text` column
- `data (1).csv`: Legal content with `full_text` column  
- `updated_legal_corpus.csv`: Legal corpus with `content` column

#### JSON Files
- `legal_corpus.json`: Legal documents in JSON format
- `zalo_corpus.json`: Zalo-sourced legal content
- `vbpl_crawl.json`: Crawled legal documents from VBPL

### Output Format
The pipeline generates a unified JSONL file where each line contains:
```json
{"id": "unique_document_id", "text": "processed_legal_text"}
{"id": "unique_document_id", "text": "processed_legal_text"}
```

## Configuration Options

### Command Line Arguments
- `--raw-prefix`: Path to raw data directory (default: `data/rag_corpus`)
- `--out-prefix`: Path to output directory (default: `processed/rag_corpus`)  
- `--coalesce`: Combine output into single file (recommended for final processing)

### Spark Configuration
The pipeline automatically configures Spark with:
- Adaptive query execution enabled
- Dynamic partition coalescing
- Optimized memory settings for text processing

## Results

After successful execution, processed data will be available at:
- `processed/rag_corpus/combined.jsonl`

The output format follows JSON Lines specification with each document containing a unique ID and processed text content suitable for RAG system integration.

## Cloud Storage Integration

### Upload to S3 (Optional)
For cloud deployment, use the S3 upload utility:
```bash
# Configure AWS credentials first
aws configure

# Upload processed data
python utils/upload_to_s3.py
```

### S3 Data Structure
```
s3://legal-datalake/
├── raw/
│   └── rag_corpus/
│       └── [source files]
└── processed/
    └── rag_corpus/
        └── combined.jsonl
```
