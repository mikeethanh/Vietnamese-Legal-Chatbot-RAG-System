#!/bin/bash

# Quick data preprocessing and upload script
# Usage: ./prepare_data.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if raw data exists
RAW_DATA="../data_pipeline/data/finetune_llm/finetune_llm_data.jsonl"
LOCAL_RAW_DATA="data_processing/raw_data/finetune_llm_data.jsonl"

if [ -f "$RAW_DATA" ]; then
    print_status "Found raw data at $RAW_DATA"
    
    # Create directory and copy
    mkdir -p data_processing/raw_data
    cp "$RAW_DATA" "$LOCAL_RAW_DATA"
    print_success "Raw data copied to local processing directory"
    
elif [ -f "$LOCAL_RAW_DATA" ]; then
    print_status "Found raw data in local processing directory"
    
else
    print_error "Raw data not found!"
    print_error "Please place your finetune_llm_data.jsonl in one of these locations:"
    print_error "  1. $RAW_DATA"
    print_error "  2. $LOCAL_RAW_DATA"
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Copying from template..."
    cp .env.template .env
    print_warning "Please edit .env file with your Digital Ocean Spaces credentials before continuing"
    exit 1
fi

# Source environment variables
source .env

# Check DO Spaces credentials
if [ -z "$DO_SPACES_KEY" ] || [ "$DO_SPACES_KEY" = "your_do_spaces_key_here" ]; then
    print_error "Digital Ocean Spaces credentials not configured!"
    print_error "Please edit .env file with your actual credentials"
    exit 1
fi

print_status "Starting data preprocessing pipeline..."

# 1. Analyze data
print_status "Step 1: Analyzing raw data structure..."
python data_processing/analyze_data.py

# 2. Process data for Llama format
print_status "Step 2: Converting to Llama-3.1 format..."
python data_processing/process_llama_data.py

# 3. Split data into train/val/test
print_status "Step 3: Creating train/val/test splits..."
python data_processing/split_data.py

# 4. Upload to Digital Ocean Spaces
print_status "Step 4: Uploading to Digital Ocean Spaces..."
python do_spaces_manager.py upload-data

# 5. Verify upload
print_status "Step 5: Verifying upload..."
python do_spaces_manager.py list process_data/finetune_data/

print_success "Data preprocessing completed successfully!"
print_status "Your data is now ready for training on GPU droplets"

echo ""
echo "📊 Next Steps:"
echo "1. Create a GPU droplet for training"
echo "2. Clone this repository on the droplet" 
echo "3. Run: ./run_pipeline.sh train"
echo ""
echo "📂 Data uploaded to Digital Ocean Spaces:"
echo "   - Bucket: $DO_SPACES_BUCKET"
echo "   - Path: process_data/finetune_data/"
echo ""
echo "🚀 Ready for GPU training!"