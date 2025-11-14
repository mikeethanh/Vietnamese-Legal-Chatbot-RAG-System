#!/bin/bash

# Vietnamese Legal LLM Training and Deployment Script
# Usage: ./run_pipeline.sh [command]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to load environment variables
load_env() {
    if [ -f ".env" ]; then
        print_status "Loading environment variables from .env..."
        # Export variables, filtering out comments and empty lines
        set -a  # automatically export all variables
        source .env
        set +a  # turn off automatic export
        print_success "Environment variables loaded"
        
        # Verify key variables are loaded
        if [ -z "$DO_SPACES_KEY" ] || [ -z "$DO_SPACES_SECRET" ]; then
            print_error "Digital Ocean Spaces credentials not found in .env file"
            exit 1
        fi
    else
        print_warning ".env file not found. Please copy .env.template to .env and configure it."
        exit 1
    fi
}

# Function to check if GPU is available
check_gpu() {
    print_status "Checking GPU availability..."
    if nvidia-smi > /dev/null 2>&1; then
        print_success "GPU found: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | head -1
    else
        print_error "No GPU found. This pipeline requires NVIDIA GPU."
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    print_status "Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Copy environment template if .env doesn't exist
    if [ ! -f ".env" ]; then
        print_status "Creating .env file from template..."
        cp .env.template .env
        print_warning "Please edit .env file with your API keys and configuration"
    fi
    
    print_success "Environment setup completed"
}

# Function to upload data to Digital Ocean Spaces
upload_data() {
    print_status "Uploading processed data to Digital Ocean Spaces..."
    
    # Load environment variables
    load_env
    
    # Process data first if not done
    if [ ! -f "data_processing/splits/train.jsonl" ]; then
        print_warning "Data not processed yet. Processing first..."
        process_data
        split_data
    fi
    
    # Upload data
    python3 do_spaces_manager.py upload-data
    
    print_success "Data uploaded to Digital Ocean Spaces"
}

# Function to download data from Digital Ocean Spaces
download_data() {
    print_status "Downloading data from Digital Ocean Spaces..."
    
    # Load environment variables
    load_env
    
    # Download training data
    python3 do_spaces_manager.py download-data
    
    print_success "Data downloaded from Digital Ocean Spaces"
}

# Function to analyze data
analyze_data() {
    print_status "Analyzing data structure..."
    python3 data_processing/analyze_data.py
    print_success "Data analysis completed"
}

# Function to process data
process_data() {
    print_status "Processing data for Llama format..."
    python3 data_processing/process_llama_data.py
    print_success "Data processing completed"
}

# Function to split data
split_data() {
    print_status "Splitting data into train/val/test..."
    python3 data_processing/split_data.py
    print_success "Data splitting completed"
}

# Function to run training
train_model() {
    print_status "Starting model training..."
    check_gpu
    
    # Load environment variables
    load_env
    
    # Download data from DO Spaces if not available locally
    if [ ! -f "data_processing/splits/train.jsonl" ]; then
        print_warning "Training data not found locally. Downloading from Digital Ocean Spaces..."
        download_data
    fi
    
    # Start training
    print_status "Training Llama-3.1-8B with LoRA..."
    cd finetune
    python3 train_llama.py
    cd ..
    
    print_success "Training completed! Model saved locally and uploaded to Digital Ocean Spaces"
}

# Function to evaluate model
evaluate_model() {
    print_status "Evaluating trained model..."
    
    if [ ! -d "finetune/outputs/final_model" ]; then
        print_error "Trained model not found. Please run training first."
        exit 1
    fi
    
    python3 evaluation/evaluate_model.py
    print_success "Evaluation completed! Results saved to evaluation/results"
}

# Function to serve model
serve_model() {
    print_status "Starting model serving..."
    
    # Load environment variables
    load_env
    
    if [ ! -d "finetune/outputs/final_model" ]; then
        print_error "Trained model not found. Please run training first."
        exit 1
    fi
    
    # Copy model to serving directory
    if [ ! -d "serving/model" ]; then
        mkdir -p serving/model
        cp -r finetune/outputs/final_model/* serving/model/
    fi
    
    # Start server
    export MODEL_PATH="$(pwd)/serving/model"
    cd serving
    python3 serve_model.py
}

# Function to build Docker image
build_docker() {
    print_status "Building Docker image..."
    
    # Ensure model exists
    if [ ! -d "finetune/outputs/final_model" ]; then
        print_error "Trained model not found. Please run training first."
        exit 1
    fi
    
    # Copy model for Docker
    mkdir -p docker/model
    cp -r finetune/outputs/final_model/* docker/model/
    
    # Build image
    cd docker
    docker build -t vietnamese-legal-llm:latest .
    
    print_success "Docker image built successfully"
}

# Function to deploy with Docker Compose
deploy_docker() {
    print_status "Deploying with Docker Compose..."
    
    # Ensure model exists
    if [ ! -d "docker/model" ]; then
        print_error "Model not found in docker directory. Run build_docker first."
        exit 1
    fi
    
    cd docker
    docker-compose up -d
    
    print_success "Service deployed! API available at http://localhost:6000"
    print_status "Check health: curl http://localhost:6000/health"
}

# Function to run complete pipeline
run_full_pipeline() {
    print_status "Running complete training and deployment pipeline..."
    
    setup_environment
    
    # Copy data from local if available, otherwise prompt user
    if [ -f "../data_pipeline/data/finetune_llm/finetune_llm_data.jsonl" ]; then
        print_status "Found local data, copying and processing..."
        mkdir -p data_processing/raw_data
        cp ../data_pipeline/data/finetune_llm/finetune_llm_data.jsonl data_processing/raw_data/
        analyze_data
        process_data
        split_data
        upload_data
    else
        print_warning "Local data not found. Please:"
        print_warning "1. Place your data in data_processing/raw_data/finetune_llm_data.jsonl"
        print_warning "2. Or run data processing steps manually"
        return 1
    fi
    
    train_model
    evaluate_model
    
    print_success "Complete pipeline finished successfully!"
    print_status "Your Vietnamese Legal LLM model is trained and uploaded to Digital Ocean Spaces!"
}

# Function to upload model to Digital Ocean Spaces
upload_model() {
    print_status "Uploading model to Digital Ocean Spaces..."
    
    # Load environment variables
    load_env
    
    model_dir="finetune/outputs/final_model"
    if [ ! -d "$model_dir" ]; then
        print_error "Model not found. Please run training first."
        exit 1
    fi
    
    python3 do_spaces_manager.py upload-model "$model_dir"
    print_success "Model uploaded to Digital Ocean Spaces"
}

# Function to download model from Digital Ocean Spaces
download_model() {
    print_status "Downloading model from Digital Ocean Spaces..."
    
    # Load environment variables
    load_env
    
    model_name=${1:-"latest"}
    local_dir=${2:-"./model"}
    
    python3 do_spaces_manager.py download-model "$model_name" "$local_dir"
    print_success "Model downloaded from Digital Ocean Spaces"
}

# Function to list models in Digital Ocean Spaces
list_models() {
    print_status "Listing models in Digital Ocean Spaces..."
    
    # Load environment variables
    load_env
    
    python3 do_spaces_manager.py list "models/"
}

# Function to show usage
show_usage() {
    echo "Vietnamese Legal LLM Training and Deployment Pipeline"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup           Setup environment and dependencies"
    echo "  upload          Upload processed data to Digital Ocean Spaces"
    echo "  download        Download training data from Digital Ocean Spaces"
    echo "  analyze         Analyze data structure"
    echo "  process         Process data for training"
    echo "  split           Split data into train/val/test"
    echo "  train           Train the model"
    echo "  evaluate        Evaluate trained model"
    echo "  serve           Serve model locally"
    echo "  build-docker    Build Docker image"
    echo "  deploy          Deploy with Docker Compose"
    echo "  pipeline        Run complete pipeline"
    echo "  check-gpu       Check GPU availability"
    echo ""
    echo "Digital Ocean Spaces Commands:"
    echo "  upload-model    Upload model to DO Spaces"
    echo "  download-model  Download model from DO Spaces"
    echo "  list-models     List available models in DO Spaces"
    echo ""
    echo "Examples:"
    echo "  $0 pipeline              # Run complete pipeline"
    echo "  $0 upload               # Upload data to DO Spaces"
    echo "  $0 train                # Train model (auto-download data)"
    echo "  $0 serve                # Serve model (auto-download model)"
    echo "  $0 deploy               # Deploy with Docker"
}

# Main execution
case "$1" in
    setup)
        setup_environment
        ;;
    upload)
        upload_data
        ;;
    download)
        download_data
        ;;
    analyze)
        analyze_data
        ;;
    process)
        process_data
        ;;
    split)
        split_data
        ;;
    train)
        train_model
        ;;
    evaluate)
        evaluate_model
        ;;
    serve)
        serve_model
        ;;
    build-docker)
        build_docker
        ;;
    deploy)
        deploy_docker
        ;;
    pipeline)
        run_full_pipeline
        ;;
    check-gpu)
        check_gpu
        ;;
    upload-model)
        upload_model
        ;;
    download-model)
        download_model "$2" "$3"
        ;;
    list-models)
        list_models
        ;;
    *)
        show_usage
        exit 1
        ;;
esac