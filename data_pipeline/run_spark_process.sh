#!/bin/bash

# Vietnamese Legal Chatbot - Spark Processing Script
# Usage: ./run_spark_process.sh

set -e

# Default configuration
BUCKET="legal-datalake"
RAW_PREFIX="raw/rag_corpus"
OUT_PREFIX="processed/rag_corpus"

# S3 Configuration (set these environment variables or modify here)
S3_ENDPOINT="${S3_ENDPOINT:-}"  # Leave empty for AWS S3, set for custom endpoint
ACCESS_KEY="${AWS_ACCESS_KEY_ID:-}"
SECRET_KEY="${AWS_SECRET_ACCESS_KEY:-}"

# Spark configuration
SPARK_MASTER="local[*]"
DRIVER_MEMORY="12g"
PACKAGES="org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.11.901"

echo "🏛️  === Vietnamese Legal Chatbot - Spark Processing ==="
echo "📦 Bucket: $BUCKET"
echo "📂 Raw prefix: $RAW_PREFIX"
echo "📁 Output prefix: $OUT_PREFIX"
if [ -n "$S3_ENDPOINT" ]; then
    echo "🌐 S3 endpoint: $S3_ENDPOINT"
else
    echo "🌐 Using AWS S3 (default endpoint)"
fi
echo "=================================================="

# Check if spark-submit is available
if ! command -v spark-submit &> /dev/null; then
    echo "❌ spark-submit command not found. Please install Apache Spark."
    echo "   Download from: https://spark.apache.org/downloads.html"
    exit 1
fi

# Build spark-submit command
SPARK_CMD="spark-submit \
  --master $SPARK_MASTER \
  --driver-memory $DRIVER_MEMORY \
  --packages $PACKAGES \
  utils/spark_process_rag_corpus.py \
    --bucket $BUCKET \
    --raw-prefix $RAW_PREFIX \
    --out-prefix $OUT_PREFIX \
    --coalesce"

# Add S3 endpoint if specified
if [ -n "$S3_ENDPOINT" ]; then
    SPARK_CMD="$SPARK_CMD --s3-endpoint $S3_ENDPOINT"
fi

# Add credentials if specified
if [ -n "$ACCESS_KEY" ]; then
    SPARK_CMD="$SPARK_CMD --access-key $ACCESS_KEY"
fi

if [ -n "$SECRET_KEY" ]; then
    SPARK_CMD="$SPARK_CMD --secret-key $SECRET_KEY"
fi

echo "🚀 Running Spark job..."
echo "Command: $SPARK_CMD"
echo ""

# Execute the command
eval $SPARK_CMD

echo ""
echo "✅ Spark processing completed successfully!"
echo "📁 Processed data available at: s3://$BUCKET/$OUT_PREFIX/combined.jsonl"
