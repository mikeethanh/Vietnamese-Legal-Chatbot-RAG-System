#!/bin/bash

# Deployment script for Vietnamese Legal Embedding Server
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env exists
check_env() {
    if [ ! -f .env ]; then
        log_error ".env file not found!"
        log_info "Please copy .env.template to .env and configure it"
        exit 1
    fi
    
    # Load environment variables
    export $(cat .env | xargs)
    
    # Check required variables
    if [ -z "$SPACES_ACCESS_KEY" ] || [ -z "$SPACES_SECRET_KEY" ]; then
        log_error "SPACES_ACCESS_KEY and SPACES_SECRET_KEY are required!"
        exit 1
    fi
}

# Download latest model from Spaces
download_model() {
    log_info "Downloading latest model from Spaces..."
    
    # Create models directory
    mkdir -p models
    
    # Use Python script to download model (simple approach)
    python3 << 'EOF'
import boto3
import os
import sys

# Initialize Spaces client
endpoint_url = os.getenv('SPACES_ENDPOINT', 'https://sgp1.digitaloceanspaces.com')
region = 'sgp1' if 'sgp1' in endpoint_url else 'sfo3'

client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('SPACES_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SPACES_SECRET_KEY'),
    endpoint_url=endpoint_url,
    region_name=region
)

bucket = os.getenv('SPACES_BUCKET', 'legal-datalake')

try:
    # List all model folders
    response = client.list_objects_v2(
        Bucket=bucket,
        Prefix='models/embedding_model_gpu_'
    )
    
    if 'Contents' not in response:
        print("No models found in Spaces!")
        sys.exit(1)
    
    # Get latest model folder
    model_folders = {}
    for obj in response['Contents']:
        if obj['Key'].endswith('/'):
            folder_name = obj['Key'].rstrip('/')
            timestamp = folder_name.split('_')[-2] + '_' + folder_name.split('_')[-1]
            model_folders[timestamp] = folder_name
    
    if not model_folders:
        print("No valid model folders found!")
        sys.exit(1)
    
    # Get latest timestamp
    latest_timestamp = sorted(model_folders.keys())[-1]
    latest_model_path = model_folders[latest_timestamp]
    
    print(f"Latest model: {latest_model_path}")
    
    # Save model path to file for docker-compose
    with open('latest_model_path.txt', 'w') as f:
        f.write(latest_model_path)
    
    print("Model path saved successfully!")
    
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        log_info "Model path identified successfully"
        MODEL_PATH=$(cat latest_model_path.txt)
        log_info "Using model: $MODEL_PATH"
        
        # Update .env file with latest model path
        sed -i "s|MODEL_PATH=.*|MODEL_PATH=$MODEL_PATH|g" .env
    else
        log_error "Failed to identify latest model"
        exit 1
    fi
}

# Deploy services
deploy() {
    log_info "Deploying embedding server..."
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Build and start services
    docker-compose build --no-cache
    docker-compose up -d
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 10
    
    # Check health
    health
}

# Check service health
health() {
    log_info "Checking service health..."
    
    # Check if container is running
    if ! docker-compose ps | grep -q "Up"; then
        log_error "Services are not running!"
        docker-compose logs
        exit 1
    fi
    
    # Check health endpoint
    for i in {1..10}; do
        if curl -s -f http://localhost:5000/health > /dev/null 2>&1; then
            log_info "✅ Service is healthy!"
            return 0
        fi
        log_warn "Waiting for health check... ($i/10)"
        sleep 5
    done
    
    log_error "Health check failed!"
    docker-compose logs
    exit 1
}

# Test embedding API
test() {
    log_info "Testing embedding API..."
    
    response=$(curl -s -X POST http://localhost:5000/embed \
        -H "Content-Type: application/json" \
        -d '{"texts": ["Luật doanh nghiệp"], "normalize": true}')
    
    if echo "$response" | grep -q "embeddings"; then
        log_info "✅ API test successful!"
        echo "Response: $response"
    else
        log_error "❌ API test failed!"
        echo "Response: $response"
        exit 1
    fi
}

# Show logs
logs() {
    docker-compose logs -f
}

# Stop services
stop() {
    log_info "Stopping services..."
    docker-compose down
}

# Main script
case "$1" in
    "download")
        check_env
        download_model
        ;;
    "deploy")
        check_env
        download_model
        deploy
        ;;
    "health")
        health
        ;;
    "test")
        test
        ;;
    "logs")
        logs
        ;;
    "stop")
        stop
        ;;
    *)
        echo "Usage: $0 {download|deploy|health|test|logs|stop}"
        echo ""
        echo "Commands:"
        echo "  download  - Download latest model from Spaces"
        echo "  deploy    - Deploy embedding server"
        echo "  health    - Check service health"
        echo "  test      - Test embedding API"
        echo "  logs      - Show service logs"
        echo "  stop      - Stop services"
        exit 1
        ;;
esac