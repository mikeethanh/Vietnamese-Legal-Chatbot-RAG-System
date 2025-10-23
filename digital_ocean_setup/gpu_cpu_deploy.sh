#!/bin/bash
#
# GPU/CPU Deployment Script cho Vietnamese Legal AI
# Quản lý workflow: GPU training → CPU serving
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Unicode symbols
CHECK="✅"
CROSS="❌"
GPU="🚀"
CPU="💻"
UPLOAD="📤"
DOWNLOAD="📥"

# Configuration
COMPOSE_FILE="docker-compose.gpu.yml"
ENV_FILE=".env"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_gpu() {
    echo -e "${PURPLE}[GPU]${NC} $1"
}

log_cpu() {
    echo -e "${BLUE}[CPU]${NC} $1"
}

# Detect environment type
detect_environment() {
    if nvidia-smi >/dev/null 2>&1; then
        echo "gpu"
    else
        echo "cpu"
    fi
}

# Check environment file
check_env_file() {
    if [ ! -f "$ENV_FILE" ]; then
        log_error ".env file not found!"
        log_info "Please create .env file. Use .env.template as reference."
        exit 1
    fi
    source "$ENV_FILE"
    log_info "Environment file loaded"
}

# GPU specific functions
setup_gpu() {
    log_gpu "Setting up GPU environment..."
    
    # Check NVIDIA drivers
    if ! nvidia-smi >/dev/null 2>&1; then
        log_error "NVIDIA drivers not found!"
        log_info "Installing NVIDIA drivers..."
        
        apt update
        apt install -y ubuntu-drivers-common
        ubuntu-drivers autoinstall
        
        log_warn "Please reboot the system and run this script again."
        exit 1
    fi
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        log_error "NVIDIA Docker not configured!"
        log_info "Installing NVIDIA Container Toolkit..."
        
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
        
        apt update && apt install -y nvidia-docker2
        systemctl restart docker
        
        log_info "NVIDIA Docker installed successfully"
    fi
    
    log_gpu "GPU environment ready!"
}

# GPU training function
gpu_train() {
    log_gpu "Starting GPU training..."
    
    check_env_file
    setup_gpu
    
    # Create necessary directories
    mkdir -p data models logs
    
    # Build GPU training image
    log_gpu "Building GPU training image..."
    docker build -f Dockerfile.gpu-training -t legal-embedding-gpu:latest .
    
    # Run GPU training
    log_gpu "Starting GPU training container..."
    
    docker run --gpus all \
        --name legal-gpu-training \
        --rm \
        -v $(pwd)/data:/tmp/data \
        -v $(pwd)/models:/tmp/model \
        -v $(pwd)/logs:/tmp/logs \
        -e SPACES_ACCESS_KEY="$SPACES_ACCESS_KEY" \
        -e SPACES_SECRET_KEY="$SPACES_SECRET_KEY" \
        -e SPACES_ENDPOINT="$SPACES_ENDPOINT" \
        -e SPACES_BUCKET="$SPACES_BUCKET" \
        legal-embedding-gpu:latest \
        python train_embedding_gpu.py \
        --spaces-access-key "$SPACES_ACCESS_KEY" \
        --spaces-secret-key "$SPACES_SECRET_KEY" \
        --spaces-endpoint "$SPACES_ENDPOINT" \
        --spaces-bucket "$SPACES_BUCKET" \
        --base-model "${BASE_MODEL:-VietAI/viet-electra-base}" \
        --epochs ${EPOCHS:-5} \
        --batch-size ${GPU_BATCH_SIZE:-32} \
        --max-samples ${MAX_SAMPLES:-}
    
    if [ $? -eq 0 ]; then
        log_gpu "GPU training completed successfully!"
        
        # Extract model path
        MODEL_PATH=$(ls -1t models/ | head -1)
        log_gpu "Model saved: $MODEL_PATH"
        
        # Create deployment info
        cat > deployment_info.json << EOF
{
    "training_completed": "$(date -Iseconds)",
    "model_path": "$MODEL_PATH",
    "gpu_used": "$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)",
    "training_method": "GPU"
}
EOF
        
    else
        log_error "GPU training failed!"
        exit 1
    fi
}

# CPU serving functions
cpu_setup() {
    log_cpu "Setting up CPU serving environment..."
    
    check_env_file
    
    # Build CPU serving image
    log_cpu "Building CPU serving image..."
    docker-compose build embedding-server
    
    log_cpu "CPU environment ready!"
}

cpu_deploy() {
    log_cpu "Deploying CPU serving services..."
    
    cpu_setup
    
    # Stop existing services
    docker-compose down 2>/dev/null || true
    
    # Start serving services
    docker-compose up -d embedding-server nginx
    
    # Wait for services
    log_cpu "Waiting for services to start..."
    sleep 15
    
    # Check health
    if check_cpu_health; then
        log_cpu "CPU serving deployed successfully!"
    else
        log_error "CPU serving deployment failed!"
        exit 1
    fi
}

check_cpu_health() {
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:5000/health >/dev/null 2>&1; then
            log_cpu "Health check passed (attempt $attempt)"
            return 0
        fi
        
        log_cpu "Health check failed, attempt $attempt/$max_attempts"
        sleep 10
        ((attempt++))
    done
    
    return 1
}

# Download model from GPU training
download_model() {
    local model_path="$1"
    
    if [ -z "$model_path" ]; then
        log_error "Model path not specified"
        exit 1
    fi
    
    log_cpu "Downloading model: $model_path"
    
    # Use AWS CLI với Spaces endpoint
    aws s3 sync "s3://$SPACES_BUCKET/models/$model_path" "./models/$model_path" \
        --endpoint-url="$SPACES_ENDPOINT"
    
    if [ $? -eq 0 ]; then
        log_cpu "Model downloaded successfully"
        
        # Update environment để point to new model
        sed -i "s|MODEL_PATH=.*|MODEL_PATH=models/$model_path|" "$ENV_FILE"
        
    else
        log_error "Model download failed"
        exit 1
    fi
}

# Monitor functions
monitor_gpu() {
    log_gpu "GPU Monitoring (Ctrl+C to stop):"
    
    while true; do
        clear
        echo -e "${PURPLE}=== GPU Status ===${NC}"
        nvidia-smi
        echo -e "\n${PURPLE}=== Docker Containers ===${NC}"
        docker ps --filter "name=legal"
        echo -e "\n${PURPLE}=== System Resources ===${NC}"
        free -h
        echo -e "\n${PURPLE}=== Disk Usage ===${NC}"
        df -h
        sleep 10
    done
}

monitor_cpu() {
    log_cpu "CPU Monitoring (Ctrl+C to stop):"
    
    while true; do
        clear
        echo -e "${BLUE}=== Service Status ===${NC}"
        docker-compose ps
        echo -e "\n${BLUE}=== System Resources ===${NC}"
        free -h
        echo -e "\n${BLUE}=== API Health ===${NC}"
        curl -s http://localhost:5000/health | jq . 2>/dev/null || echo "API not responding"
        echo -e "\n${BLUE}=== Recent Logs ===${NC}"
        docker-compose logs --tail=5 embedding-server
        sleep 15
    done
}

# Backup and cleanup
backup_gpu_artifacts() {
    log_gpu "Creating backup of GPU artifacts..."
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="gpu_training_backup_$timestamp"
    
    tar -czf "${backup_name}.tar.gz" \
        models/ \
        logs/ \
        .env \
        deployment_info.json 2>/dev/null
    
    # Upload backup to Spaces
    if [ ! -z "$SPACES_BUCKET" ]; then
        aws s3 cp "${backup_name}.tar.gz" "s3://$SPACES_BUCKET/backups/" \
            --endpoint-url="$SPACES_ENDPOINT"
        log_gpu "Backup uploaded to Spaces: backups/${backup_name}.tar.gz"
    fi
    
    log_gpu "Backup created: ${backup_name}.tar.gz"
}

# Cost optimization
estimate_costs() {
    local hours="$1"
    local gpu_cost_per_hour=3.0  # $72/month ÷ 24 hours
    local cpu_cost_per_hour=1.0  # $24/month ÷ 24 hours
    
    if [ -z "$hours" ]; then
        hours=2  # Default estimate
    fi
    
    local gpu_cost=$(echo "$gpu_cost_per_hour * $hours" | awk '{print $1 * $3}')
    local cpu_monthly_cost=24  # Fixed monthly cost
    
    echo -e "${CYAN}💰 Cost Estimate:${NC}"
    echo -e "  GPU Training ($hours hours): \$${gpu_cost}"
    echo -e "  CPU Serving (monthly): \$${cpu_monthly_cost}"
    echo -e "  Total for this training: \$${gpu_cost}"
}

# Main menu
show_help() {
    cat << EOF
Usage: $0 [COMMAND]

GPU Commands (chạy trên GPU Droplet):
    gpu-setup      - Setup GPU environment
    gpu-train      - Train model trên GPU
    gpu-monitor    - Monitor GPU training
    gpu-backup     - Backup GPU artifacts
    gpu-cleanup    - Clean GPU environment

CPU Commands (chạy trên CPU Droplet):
    cpu-setup      - Setup CPU serving environment
    cpu-deploy     - Deploy serving services
    cpu-monitor    - Monitor CPU serving
    cpu-health     - Check serving health
    download-model MODEL_PATH - Download model từ Spaces

General Commands:
    auto-train     - Auto GPU training workflow
    auto-deploy    - Auto CPU deployment workflow
    select-model   - Interactive model selection
    estimate-cost  - Estimate costs
    help           - Show this help

Environment Detection:
    Current: $(detect_environment)

Examples:
    # Trên GPU Droplet - Choose model interactively
    $0 select-model && $0 gpu-train
    
    # Trên GPU Droplet - Direct training
    $0 gpu-setup && $0 gpu-train
    
    # Trên CPU Droplet  
    $0 download-model embedding_model_20231023_120000
    $0 cpu-deploy
    
    # Check costs
    $0 estimate-cost

Model Options:
    VietAI/viet-electra-base (recommended for Vietnamese legal)
    intfloat/multilingual-e5-large (best quality, needs more GPU)
    vinai/phobert-base-v2 (stable Vietnamese option)
EOF
}

# Model selection function
select_model() {
    log_info "🤖 Interactive Model Selection"
    echo
    echo "Available Vietnamese embedding models:"
    echo
    echo "1. VietAI/viet-electra-base (RECOMMENDED)"
    echo "   ⭐ Best for Vietnamese legal documents"
    echo "   💾 Memory: ~1GB, Speed: Fast"
    echo "   🎯 Optimized for Vietnamese context"
    echo
    echo "2. intfloat/multilingual-e5-large"
    echo "   🏆 Highest quality multilingual model"
    echo "   💾 Memory: ~2GB, Speed: Slower"
    echo "   🌐 Best for mixed Vietnamese/English"
    echo
    echo "3. vinai/phobert-base-v2"
    echo "   🇻🇳 Proven Vietnamese model"
    echo "   💾 Memory: ~1.5GB, Speed: Medium"
    echo "   📚 Academic research proven"
    echo
    echo "4. sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    echo "   🔄 Stable fallback option"
    echo "   💾 Memory: ~1.2GB, Speed: Medium"
    echo "   ⚖️  Balanced performance"
    echo
    
    read -p "Select model (1-4) [1]: " choice
    
    case $choice in
        2)
            MODEL="intfloat/multilingual-e5-large"
            EPOCHS=3
            BATCH_SIZE=16
            ;;
        3)
            MODEL="vinai/phobert-base-v2"
            EPOCHS=4
            BATCH_SIZE=28
            ;;
        4)
            MODEL="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            EPOCHS=4
            BATCH_SIZE=24
            ;;
        1|"")
            MODEL="VietAI/viet-electra-base"
            EPOCHS=5
            BATCH_SIZE=32
            ;;
        *)
            log_error "Invalid choice. Using default VietAI/viet-electra-base"
            MODEL="VietAI/viet-electra-base"
            EPOCHS=5
            BATCH_SIZE=32
            ;;
    esac
    
    # Update .env file
    if [ -f ".env" ]; then
        sed -i "s|BASE_MODEL=.*|BASE_MODEL=$MODEL|" .env
        sed -i "s|EPOCHS=.*|EPOCHS=$EPOCHS|" .env
        sed -i "s|GPU_BATCH_SIZE=.*|GPU_BATCH_SIZE=$BATCH_SIZE|" .env
        
        log_info "✅ Updated .env with:"
        log_info "   BASE_MODEL=$MODEL"
        log_info "   EPOCHS=$EPOCHS"
        log_info "   GPU_BATCH_SIZE=$BATCH_SIZE"
    else
        log_warn "⚠️ .env file not found. Please create it from .env.template"
    fi
}

# Auto workflows
auto_train_workflow() {
    log_info "🤖 Starting automatic GPU training workflow..."
    
    if [ "$(detect_environment)" != "gpu" ]; then
        log_error "This command requires GPU environment"
        exit 1
    fi
    
    estimate_costs 2
    
    read -p "Continue with GPU training? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Training cancelled"
        exit 0
    fi
    
    gpu_train
    backup_gpu_artifacts
    
    log_info "🎉 GPU training workflow completed!"
    log_info "💡 Next: Copy model path to CPU Droplet và run 'cpu-deploy'"
}

auto_deploy_workflow() {
    log_info "🤖 Starting automatic CPU deployment workflow..."
    
    if [ "$(detect_environment)" != "cpu" ]; then
        log_warn "This command is designed for CPU environment"
    fi
    
    cpu_deploy
    
    log_info "🎉 CPU deployment workflow completed!"
    log_info "💡 API available at: http://$(curl -s ifconfig.me):5000"
}

# Main script
case "$1" in
    # GPU commands
    gpu-setup)
        setup_gpu
        ;;
    gpu-train)
        gpu_train
        ;;
    gpu-monitor)
        monitor_gpu
        ;;
    gpu-backup)
        backup_gpu_artifacts
        ;;
    gpu-cleanup)
        docker system prune -af
        rm -rf models/* logs/* data/*
        ;;
        
    # CPU commands
    cpu-setup)
        cpu_setup
        ;;
    cpu-deploy)
        cpu_deploy
        ;;
    cpu-monitor)
        monitor_cpu
        ;;
    cpu-health)
        check_cpu_health && echo "✅ Healthy" || echo "❌ Unhealthy"
        ;;
    download-model)
        download_model "$2"
        ;;
        
    # Auto workflows
    auto-train)
        auto_train_workflow
        ;;
    auto-deploy)
        auto_deploy_workflow
        ;;
        
    # General commands
    select-model)
        select_model
        ;;
    estimate-cost)
        estimate_costs "$2"
        ;;
    help|*)
        show_help
        ;;
esac