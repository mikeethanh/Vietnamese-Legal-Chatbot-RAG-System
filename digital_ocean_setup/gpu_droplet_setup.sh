#!/bin/bash
# GPU Droplet Quick Setup for Vietnamese Legal BGE-M3 Training
# Usage: ./gpu_droplet_setup.sh

set -e

echo "🚀 Setting up DigitalOcean GPU Droplet for AI/ML Training..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install additional dependencies
echo "🔧 Installing additional dependencies..."
apt install -y htop nvtop tree curl wget git

# Verify GPU and CUDA
echo "🔥 Checking GPU and CUDA installation..."
nvidia-smi
nvcc --version

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip
pip install -r requirements_gpu.txt

# Setup environment file
echo "⚙️ Configuring environment..."
if [ ! -f .env ]; then
    cp .env.gpu-droplet .env
    echo "📝 Created .env file - please edit with your credentials"
fi

# Create necessary directories
mkdir -p ./models
mkdir -p ./data
mkdir -p ./logs

# Setup monitoring
echo "📊 Setting up system monitoring..."
cat > monitor_training.sh << 'EOF'
#!/bin/bash
# Monitor GPU and system during training
while true; do
    clear
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv
    echo
    echo "=== System Status ==="
    htop -n 1 | head -20
    echo
    echo "=== Disk Usage ==="
    df -h
    echo
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
EOF
chmod +x monitor_training.sh

# Setup training script
echo "🏃 Setting up training launcher..."
cat > start_training.sh << 'EOF'
#!/bin/bash
# Start BGE-M3 training with monitoring

echo "🚀 Starting Vietnamese Legal BGE-M3 Training..."

# Load environment
source .env

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Start training with logging
nohup python3 train_embedding_gpu.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
TRAIN_PID=$!

echo "📝 Training started with PID: $TRAIN_PID"
echo "📊 Logs: training_$(date +%Y%m%d_%H%M%S).log"
echo "🔍 Monitor with: tail -f training_$(date +%Y%m%d_%H%M%S).log"
echo "📈 GPU monitoring: ./monitor_training.sh"

# Save PID for easy stopping
echo $TRAIN_PID > .training_pid
EOF
chmod +x start_training.sh

# Setup stop script
cat > stop_training.sh << 'EOF'
#!/bin/bash
if [ -f .training_pid ]; then
    PID=$(cat .training_pid)
    echo "🛑 Stopping training process $PID..."
    kill $PID 2>/dev/null && echo "✅ Training stopped" || echo "⚠️ Process not found"
    rm .training_pid
else
    echo "⚠️ No training PID found"
fi
EOF
chmod +x stop_training.sh

echo "✅ GPU Droplet setup completed!"
echo
echo "📋 Next Steps:"
echo "1. Edit .env file with your DigitalOcean Spaces credentials"
echo "2. Start training: ./start_training.sh"
echo "3. Monitor progress: ./monitor_training.sh"
echo "4. Stop training: ./stop_training.sh"
echo
echo "🔍 Useful Commands:"
echo "   nvidia-smi          # Check GPU status"
echo "   htop               # System monitoring"
echo "   df -h              # Disk usage"
echo "   tail -f *.log      # View training logs"