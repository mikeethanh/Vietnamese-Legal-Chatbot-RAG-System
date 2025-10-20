# Data Pipeline for Vietnamese Legal Chatbot RAG System

This comprehensive data pipeline processes Vietnamese legal documents for both RAG (Retrieval-Augmented Generation) corpus and fine-tuning datasets using Apache Spark and pandas.

## Overview

The pipeline handles two main data processing workflows:
1. **RAG Corpus Processing**: Converts various legal document formats into unified JSONL for vector database
2. **Fine-tuning Data Processing**: Prepares question-context pairs for LLaMA model training

## Features

### RAG Corpus Processing
- **Multi-format Support**: Processes CSV, JSON, and JSONL files containing legal documents
- **Data Cleaning**: Removes duplicates, empty content, and invalid entries
- **Text Standardization**: Normalizes Vietnamese text for consistent processing
- **Scalable Processing**: Uses Apache Spark for handling large datasets
- **Gzip Handling**: Automatically processes Spark's compressed output files

### Fine-tuning Data Processing
- **Question-Context Extraction**: Standardizes QA datasets for model training
- **Multiple Sources**: Handles large_vi_legal_queries, train, and validation datasets
- **Instruction Format**: Creates Alpaca-style instruction following format
- **Data Validation**: Ensures clean, properly formatted training data

## Prerequisites

### System Requirements
- Python 3.8+
- Apache Spark 3.4+
- Java 8 or 11
- Minimum 8GB RAM (12GB recommended)


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
```

### 2. Data Structure
```
data_pipeline/
├── data/
│   ├── raw/
│   │   ├── rag_corpus/
│   │   │   ├── corpus.csv
│   │   │   ├── data (1).csv
│   │   │   ├── updated_legal_corpus.csv
│   │   │   ├── legal_corpus.json
│   │   │   ├── zalo_corpus.json
│   │   │   └── vbpl_crawl.json
│   │   └── finetune_data/
│   │       ├── large_vi_legal_queries.csv
│   │       ├── train.csv
│   │       └── valid.csv
│   └── process_data/
│       ├── rag_corpus/
│       │   └── merged_corpus.jsonl
│       └── finetune_data/
│           ├── combined_finetune_data.jsonl
│           ├── train_instruction.jsonl
│           └── valid_instruction.jsonl
```

## Processing Workflows

### RAG Corpus Processing

#### Step 1: Run Spark Processing
```bash
# Using automation script (recommended)
cd data_pipeline
chmod +x run_spark_process.sh
./run_spark_process.sh

# Or manually with spark-submit
spark-submit \
  --master local[*] \
  --driver-memory 12g \
  utils/spark_process_rag_corpus.py \
  --raw-prefix data/raw/rag_corpus \
  --out-prefix data/process_data/rag_corpus \
  --coalesce
```

#### Step 2: Convert Spark Output to JSONL
Spark creates compressed part files that need to be merged:

```bash
# Find the generated UUID directory
ls data/process_data/rag_corpus/combined.jsonl/

# Convert part files to single JSONL (replace UUID with actual directory)
python utils/merge_part_files_v2.py \
  data/process_data/rag_corpus/combined.jsonl/[UUID] \
  data/process_data/rag_corpus/merged_corpus.jsonl
```

#### Step 3: Clean Invalid JSON Lines (if needed)
```bash
# Clean any invalid JSON lines
python utils/clean_jsonl.py data/process_data/rag_corpus/merged_corpus.jsonl
```

### Fine-tuning Data Processing

#### Step 1: Process CSV Files
```bash
# Convert CSV files to standardized JSONL format
python utils/process_finetune_data.py \
  --input-dir data/raw/finetuning \
  --output-dir data/process_data/finetune_data
```

#### Step 2: Create Instruction Format
```bash
# Generate Alpaca-style instruction format for training
python utils/create_instruction_format.py \
  data/process_data/finetune_data/train.jsonl \
  data/process_data/finetune_data/train_instruction.jsonl \
  --format alpaca

python utils/create_instruction_format.py \
  data/process_data/finetune_data/valid.jsonl \
  data/process_data/finetune_data/valid_instruction.jsonl \
  --format alpaca
```

