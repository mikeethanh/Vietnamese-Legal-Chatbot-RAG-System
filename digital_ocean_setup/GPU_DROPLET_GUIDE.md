# 🚀 DigitalOcean GPU Droplet Training Guide
## Vietnamese Legal BGE-M3 Embedding Model

### 📋 **Table of Contents**
1. [Tạo GPU Droplet](#1-tạo-gpu-droplet)
2. [Initial Setup](#2-initial-setup)
3. [Environment Configuration](#3-environment-configuration)
4. [Training Deployment](#4-training-deployment)
5. [Monitoring & Management](#5-monitoring--management)
6. [Cost Management](#6-cost-management)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. 🖥️ **Tạo GPU Droplet**

### **Step 1.1: Truy cập DigitalOcean Dashboard**
```bash
# Login vào: https://cloud.digitalocean.com
# Navigate: Create → Droplets
```

### **Step 1.2: Chọn Image**
- **Choose an image**: Marketplace
- **Search**: "AI/ML Ready"
- **Select**: `PyTorch 2.1 + CUDA 12.1 on Ubuntu 22.04`
- **Benefits**: Pre-installed CUDA, PyTorch, drivers

### **Step 1.3: Chọn GPU Plan**
**Recommended cho BGE-M3:**

| GPU Type | VRAM | Cost/hour | Training Time | Total Cost |
|----------|------|-----------|---------------|------------|
| **H100** | 80GB | $4.89 | 2-3 hours | $10-15 ✅ |
| H200 | 141GB | $6.50 | 1.5-2 hours | $10-13 |
| Basic GPU | 12GB | $0.75 | ❌ OOM Error | N/A |

**💡 Tip**: Chọn H100 cho balance tốt nhất giữa cost và performance

### **Step 1.4: Configuration**
- **Region**: NYC3, SFO3, AMS3 (có GPU availability)
- **VPC**: Default
- **Authentication**: SSH Keys (secure hơn password)
- **Hostname**: `vietnamese-legal-gpu-training`
- **Tags**: `ai-ml`, `bge-m3`, `training`

### **Step 1.5: Create Droplet**
- Click "Create Droplet"
- Wait 2-3 minutes for provisioning
- Note down IP address

---

## 2. 🔧 **Initial Setup**

### **Step 2.1: SSH Connection**
```bash
# SSH vào droplet (thay YOUR_DROPLET_IP)
ssh root@YOUR_DROPLET_IP

# Verify GPU
nvidia-smi
# Expected: NVIDIA H100 80GB HBM3
```

### **Step 2.2: System Update**
```bash
# Update packages
apt update && apt upgrade -y

# Install monitoring tools
apt install -y htop nvtop tree curl wget git unzip

# Verify CUDA installation
nvcc --version
# Expected: CUDA 12.1
```

### **Step 2.3: Clone Repository**
```bash
# Clone project
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git

# Navigate to setup directory
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# List files
ls -la
```

---

## 3. ⚙️ **Environment Configuration**

### **Step 3.1: Run Setup Script**
```bash
# Make executable and run
chmod +x gpu_droplet_setup.sh
./gpu_droplet_setup.sh

# Expected output: Setup completion message
```

### **Step 3.2: Configure Spaces Credentials**
```bash
# Edit environment file
nano .env

# Add your DigitalOcean Spaces credentials:
SPACES_KEY=your_spaces_access_key_here
SPACES_SECRET=your_spaces_secret_key_here
SPACES_BUCKET=your-bucket-name-here
```

**🔑 How to get Spaces credentials:**
1. DigitalOcean Dashboard → API → Spaces Keys
2. Generate New Key
3. Save Access Key và Secret Key

### **Step 3.3: Verify Environment**
```bash
# Check environment variables
cat .env | grep -E "SPACES_|GPU_|MODEL_"

# Test Python dependencies
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Expected output:
# PyTorch: 2.1.0+cu121
# CUDA: True
```

### **Step 3.4: Create Data Directory**
```bash
# Create necessary directories
mkdir -p data/legal_corpus
mkdir -p models/checkpoints
mkdir -p logs

# Download sample data (if needed)
# curl -O https://your-data-source.com/legal_corpus.jsonl
```

---

## 4. 🚀 **Training Deployment**

### **Step 4.1: Pre-flight Check**
```bash
# Memory check
free -h
# Expected: ~125GB total memory

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# Expected: H100, 81920 MiB, 0 MiB

# Disk space
df -h
# Expected: >50GB available
```

### **Step 4.2: Start Training**
```bash
# Option A: Direct training
python3 train_embedding_gpu.py

# Option B: Background training (recommended)
./start_training.sh

# Expected output:
# 🚀 Starting Vietnamese Legal BGE-M3 Training...
# 📝 Training started with PID: 12345
# 📊 Logs: training_20251024_143022.log
```

### **Step 4.3: Verify Training Started**
```bash
# Check process
ps aux | grep python3

# Check GPU usage (should show utilization)
nvidia-smi

# Check logs
tail -f training_*.log

# Expected in logs:
# 🔥 GPU: NVIDIA H100 80GB HBM3
# 💾 Total VRAM: 80.0GB
# ✅ Model loaded successfully
```

---

## 5. 📊 **Monitoring & Management**

### **Step 5.1: Real-time Monitoring**
```bash
# Terminal 1: GPU monitoring
./monitor_training.sh

# Terminal 2: Log monitoring
tail -f training_*.log

# Terminal 3: System monitoring
htop
```

### **Step 5.2: Training Progress Indicators**
```bash
# Look for these in logs:
grep -E "Epoch|Loss|Memory" training_*.log

# Expected progression:
# Epoch 1/3: Loss decreasing from ~0.5 to ~0.1
# Memory usage stable at ~20-30GB
```

### **Step 5.3: Performance Metrics**
```bash
# GPU utilization (target: 85-95%)
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# Memory usage (target: <70GB on H100)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Training speed (target: ~100 samples/sec)
grep "samples/sec" training_*.log
```

### **Step 5.4: Checkpoints & Model Saving**
```bash
# Check model checkpoints
ls -la models/

# Monitor model upload progress
tail -f training_*.log | grep -E "Uploading|Progress"
```

---

## 6. 💰 **Cost Management**

### **Step 6.1: Training Duration Estimates**
```bash
# Based on configuration:
# Max samples: 50,000
# Batch size: 2
# Epochs: 3
# Estimated time: 2-3 hours on H100
```

### **Step 6.2: Auto-shutdown Setup**
```bash
# Set automatic shutdown after 4 hours (safety)
echo "shutdown -h +240" | at now
at -l  # Verify scheduled shutdown

# Cancel if needed
at -r [job_number]
```

### **Step 6.3: Cost Monitoring**
```bash
# Check current usage (if doctl installed)
doctl account get

# Estimated costs:
# H100: $4.89/hour × 3 hours = ~$15
# Storage: $0.02/GB/month
# Bandwidth: Free egress up to 1TB
```

### **Step 6.4: Snapshot for Backup**
```bash
# Create snapshot before training (via dashboard)
# Name: "vietnamese-legal-bge-m3-pre-training"
# Size: ~25GB
# Cost: ~$1.25/month
```

---

## 7. 🔧 **Troubleshooting**

### **Step 7.1: Common Issues & Solutions**

#### **Issue: CUDA Out of Memory**
```bash
# Check memory usage
nvidia-smi

# Solutions:
# 1. Reduce batch size in .env
echo "GPU_BATCH_SIZE=1" >> .env

# 2. Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"

# 3. Restart training
./stop_training.sh
./start_training.sh
```

#### **Issue: Training Stuck/Slow**
```bash
# Check CPU usage
htop

# Check I/O wait
iostat 1

# Solutions:
# 1. Increase dataloader workers
echo "DATALOADER_NUM_WORKERS=8" >> .env

# 2. Check disk space
df -h
```

#### **Issue: Connection Lost**
```bash
# Use screen for persistent session
screen -S training

# Run training inside screen
./start_training.sh

# Detach: Ctrl+A, D
# Reattach: screen -r training
```

### **Step 7.2: Log Analysis**
```bash
# Check for errors
grep -E "ERROR|FAILED|Exception" training_*.log

# Check memory patterns
grep -E "Memory|GPU memory" training_*.log

# Check training progress
grep -E "Epoch|Loss|samples/sec" training_*.log
```

### **Step 7.3: Recovery Procedures**
```bash
# If training fails:
# 1. Check last checkpoint
ls -la models/checkpoints/

# 2. Resume from checkpoint (if available)
python3 train_embedding_gpu.py --resume-from-checkpoint models/checkpoints/latest

# 3. Restart with reduced resources
# Edit .env: GPU_BATCH_SIZE=1, MAX_SAMPLES=10000
```

---

## 8. ✅ **Success Indicators**

### **Step 8.1: Training Completion**
Look for these in logs:
```bash
✅ Training completed successfully!
📤 Uploading model to Digital Ocean Spaces...
📍 Model available at: https://your-bucket.nyc3.digitaloceanspaces.com/models/vietnamese-legal-bge-m3
🎉 Training completed successfully!
```

### **Step 8.2: Model Validation**
```bash
# Check model files
ls -la vietnamese-legal-bge-m3/

# Expected files:
# - config.json
# - pytorch_model.bin
# - tokenizer.json
# - README.md
```

### **Step 8.3: Cleanup**
```bash
# Stop monitoring
./stop_training.sh

# Archive logs
tar -czf training_logs_$(date +%Y%m%d).tar.gz *.log

# Upload final model (if not auto-uploaded)
python3 -c "
from utils import upload_model_to_spaces
# Upload model manually if needed
"
```

---

## 9. 🎯 **Quick Commands Reference**

```bash
# Setup
./gpu_droplet_setup.sh

# Start training  
./start_training.sh

# Monitor
./monitor_training.sh
tail -f training_*.log

# Stop
./stop_training.sh

# Check status
nvidia-smi
htop
df -h

# Cleanup
shutdown -h now  # Destroy droplet to stop charges
```

---

## 10. 📞 **Support & Resources**

### **Documentation:**
- [DigitalOcean GPU Droplets](https://docs.digitalocean.com/products/droplets/how-to/create/)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)

### **Monitoring URLs:**
- Droplet Dashboard: `https://cloud.digitalocean.com/droplets`
- Spaces Browser: `https://cloud.digitalocean.com/spaces`
- Billing: `https://cloud.digitalocean.com/billing`

### **Emergency Commands:**
```bash
# Force stop all training
pkill -f python3

# Emergency GPU reset
nvidia-smi --gpu-reset

# Emergency reboot
reboot
```

---

**🎉 That's it! Follow this guide step by step for successful BGE-M3 training on DigitalOcean GPU Droplet.**