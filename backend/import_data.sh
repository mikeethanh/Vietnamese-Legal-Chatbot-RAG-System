#!/bin/bash

# Script to import Q&A data into Qdrant vector database
# This script runs inside the Docker container

echo "=============================================="
echo "Starting Q&A Data Import to Qdrant"
echo "=============================================="

# Wait for services to be ready
echo "Waiting for Qdrant to be ready..."
sleep 10

# Run the import script
cd /usr/src/app/src

python import_data.py \
    --data-file /usr/src/app/../data_pipeline/data/finetune_data/train_qa_format.jsonl \
    --collection llm \
    --batch-size 100

echo "=============================================="
echo "Import completed!"
echo "=============================================="
