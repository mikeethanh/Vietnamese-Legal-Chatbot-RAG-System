# 🚀 Vietnamese Legal LLM - GPU Droplet Deployment Guide

Hướng dẫn chi tiết để finetune và serving Vietnamese Legal LLM trên Digital Ocean GPU droplets với data/model lưu trên Spaces.

## 🎯 Tổng quan Architecture

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   Local Dev     │    │  Digital Ocean      │    │ Digital Ocean   │
│   Environment   │───▶│   GPU Droplet       │◄──▶│    Spaces       │
│                 │    │   (Training)        │    │ (Data/Models)   │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
                                 │
                                 ▼
                       ┌─────────────────────┐
                       │ Digital Ocean       │
                       │ GPU Droplet         │
                       │ (Serving)           │
                       └─────────────────────┘
```

### Workflow:
1. **Local**: Process data → Upload to Spaces
2. **Training GPU Droplet**: Download data → Train → Upload model
3. **Serving GPU Droplet**: Download model → Serve API

---

## 📋 Prerequisites

### Digital Ocean Setup
- [ ] Digital Ocean account
- [ ] Spaces bucket: `legal-datalake`
- [ ] Spaces API keys (Key + Secret)
- [ ] GPU droplet quota (H200 recommended)

### Required API Keys
- [ ] Digital Ocean Spaces keys
- [ ] Weights & Biases (optional, for monitoring)
- [ ] OpenAI API key (optional, for evaluation)
- [ ] HuggingFace token (optional)

---

## 🔧 Part 1: Local Data Preparation

### 1.1 Setup Local Environment

```bash
# Clone repository
git clone <your-repo>
cd Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving

# Setup environment
./run_pipeline.sh setup

# Configure environment variables
cp .env.template .env
# Edit .env with your Digital Ocean Spaces credentials
```

### 1.2 Environment Configuration (.env)

```bash
# Digital Ocean Spaces (Required)
DO_SPACES_KEY=your_actual_spaces_key
DO_SPACES_SECRET=your_actual_spaces_secret  
DO_SPACES_ENDPOINT=https://sgp1.digitaloceanspaces.com
DO_SPACES_BUCKET=legal-datalake

# Optional API Keys
WANDB_API_KEY=your_wandb_key
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_hf_token
```

### 1.3 Process and Upload Data

```bash
# Copy your raw data
mkdir -p data_processing/raw_data
cp /path/to/finetune_llm_data.jsonl data_processing/raw_data/

# Process data pipeline
./run_pipeline.sh analyze    # Analyze raw data
./run_pipeline.sh process    # Convert to Llama format  
./run_pipeline.sh split      # Create train/val/test splits

# Upload to Digital Ocean Spaces
./run_pipeline.sh upload

# Verify upload
python do_spaces_manager.py list process_data/
```

**Expected Spaces Structure:**
```
legal-datalake/
└── process_data/
    └── finetune_data/
        ├── raw/finetune_llm_data.jsonl
        ├── processed/processed_llama_data.jsonl
        ├── splits/train.jsonl
        ├── splits/val.jsonl
        ├── splits/test.jsonl
        └── analysis/data_analysis.json
```

---

## 🏋️ Part 2: Training on GPU Droplet

### 2.1 Create Training GPU Droplet

**Recommended Configuration:**
- **Image**: Ubuntu 22.04 LTS
- **Size**: GPU-H100-80GB or GPU-H200-141GB
- **Region**: Choose based on your location
- **Add your SSH key**

### 2.2 Setup Training Environment

```bash
# SSH into droplet
ssh root@your-training-droplet-ip

# Update system
apt update && apt upgrade -y

# Install NVIDIA drivers and Docker
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt update
apt install -y cuda-toolkit-12-3 nvidia-driver-545

# Install Python and dependencies
apt install -y python3 python3-pip python3-venv git

# Reboot to load drivers
reboot
```

### 2.3 Clone and Setup Repository

```bash
# SSH back in after reboot
ssh root@your-training-droplet-ip

# Verify GPU
nvidia-smi

# Clone repository
git clone <your-repo>
cd Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment variables
cp .env.template .env
# Edit .env with your credentials
```

### 2.4 Download Data and Train

```bash
# Download training data from Spaces
./run_pipeline.sh download

# Verify data
ls -la data_processing/splits/

# Start training
./run_pipeline.sh train

# Monitor training (if using WandB)
# Check: https://wandb.ai/your-username/vietnamese-legal-llm
```

**Training Process:**
- Downloads data from Spaces automatically
- Trains with Unsloth LoRA optimization
- Saves model locally and uploads to Spaces
- Takes ~4-6 hours on H200 GPU

**Expected Training Output:**
```
🚀 Training completed!
📁 Model saved locally: finetune/outputs/final_model  
☁️  Model uploaded to Spaces: models/vietnamese-legal-llama-20241103_142000
```

### 2.5 Verify Model Upload

```bash
# List models in Spaces
./run_pipeline.sh list-models

# Expected output:
# models/vietnamese-legal-llama-20241103_142000/config.json
# models/vietnamese-legal-llama-20241103_142000/pytorch_model.bin
# models/vietnamese-legal-llama-20241103_142000/tokenizer.json
# models/vietnamese-legal-llama-20241103_142000/model_info.json
```

---

## 🌐 Part 3: Serving on GPU Droplet

### 3.1 Create Serving GPU Droplet

**Recommended Configuration:**
- **Image**: Ubuntu 22.04 LTS  
- **Size**: GPU-H100-80GB (can be smaller than training)
- **Region**: Same as training droplet
- **Add your SSH key**

### 3.2 Setup Serving Environment

```bash
# SSH into serving droplet
ssh root@your-serving-droplet-ip

# Install NVIDIA drivers and Docker (same as training setup)
# ... (repeat steps from 2.2)

# Clone repository  
git clone <your-repo>
cd Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure for serving
cp .env.template .env
# Edit .env with Spaces credentials and model name
```

### 3.3 Configure Serving Environment

```bash
# .env for serving
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_ENDPOINT=https://sgp1.digitaloceanspaces.com
DO_SPACES_BUCKET=legal-datalake

# Model configuration
MODEL_PATH=/app/model
MODEL_NAME=vietnamese-legal-llama-20241103_142000  # or 'latest'
HOST=0.0.0.0
PORT=8000
```

### 3.4 Download Model and Start Serving

```bash
# Download specific model
./run_pipeline.sh download-model vietnamese-legal-llama-20241103_142000 ./model

# Or download latest model
./run_pipeline.sh download-model latest ./model

# Verify model download
ls -la model/

# Start serving
cd serving
export MODEL_PATH="../model"
python serve_model.py

# Should see:
# Model loaded successfully
# Server running on http://0.0.0.0:8000
```

### 3.5 Test API

```bash
# From another terminal or local machine
curl -X POST http://your-serving-droplet-ip:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Quy định về thời hiệu khởi kiện là gì?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'

# Test health
curl http://your-serving-droplet-ip:8000/health
```

---

## 🐋 Part 4: Docker Deployment (Optional)

### 4.1 Build and Deploy with Docker

```bash
# On serving droplet
cd Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving

# Build Docker image
docker build -f docker/Dockerfile -t vietnamese-legal-llm .

# Run with Docker Compose
cd docker
docker-compose up -d

# Check logs
docker-compose logs -f vietnamese-legal-llm
```

### 4.2 Production Configuration

```yaml
# docker-compose.yml environment
environment:
  - MODEL_NAME=vietnamese-legal-llama-20241103_142000
  - DO_SPACES_KEY=${DO_SPACES_KEY}
  - DO_SPACES_SECRET=${DO_SPACES_SECRET}
  - DO_SPACES_ENDPOINT=https://sgp1.digitaloceanspaces.com
  - DO_SPACES_BUCKET=legal-datalake
```

---

## 📊 Part 5: Monitoring and Maintenance

### 5.1 Model Performance Monitoring

```bash
# Run evaluation
./run_pipeline.sh evaluate

# Monitor GPU usage
nvidia-smi -l 1

# Monitor API metrics
curl http://your-droplet-ip:8000/health
```

### 5.2 Model Updates

```bash
# Retrain with new data
# On training droplet:
./run_pipeline.sh train

# Update serving droplet
# On serving droplet:
./run_pipeline.sh download-model latest ./new_model
# Update MODEL_PATH and restart service
```

### 5.3 Scaling Multiple Serving Instances

```bash
# Load balancer configuration
# Use nginx or cloud load balancer
# Point to multiple serving droplets
```

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in training config
export CUDA_VISIBLE_DEVICES=0
# Edit train_llama.py: per_device_train_batch_size=1
```

**2. Model Download Failed**
```bash
# Check Spaces credentials
python do_spaces_manager.py list models/
# Verify network connectivity
```

**3. Slow Serving Response**
```bash
# Check GPU utilization
nvidia-smi
# Consider 4-bit quantization
# load_in_4bit=True in serving config
```

**4. API Connection Issues**
```bash
# Check firewall
ufw allow 8000
# Check service status
netstat -tlnp | grep 8000
```

---

## 💰 Cost Optimization

### GPU Droplet Management

**Training Strategy:**
- Create H200 droplet for training only
- Destroy after model upload
- Estimated cost: $10-20 for 6-hour training session

**Serving Strategy:**
- Use smaller GPU (H100-80GB) for serving
- Keep running 24/7 for production
- Estimated cost: $200-400/month

**Cost-Saving Tips:**
- Use Droplet snapshots for quick setup
- Schedule training during off-peak hours
- Use preemptible instances if available
- Monitor and scale based on usage

---

## 🎯 Success Checklist

### Training Completion ✅
- [ ] Data uploaded to Spaces
- [ ] GPU droplet created and configured
- [ ] Model trained successfully (4-6 hours)
- [ ] Model uploaded to Spaces
- [ ] Training droplet destroyed (cost saving)

### Serving Setup ✅
- [ ] Serving GPU droplet created
- [ ] Model downloaded from Spaces
- [ ] API server running
- [ ] Health checks passing
- [ ] API responses working

### Production Ready ✅
- [ ] Domain name configured
- [ ] SSL certificate installed
- [ ] Load balancer setup (if needed)
- [ ] Monitoring and alerts configured
- [ ] Backup and disaster recovery plan

---

## 📚 Additional Resources

- [Digital Ocean GPU Droplets Documentation](https://docs.digitalocean.com/products/droplets/how-to/create-gpu-droplets/)
- [Digital Ocean Spaces Documentation](https://docs.digitalocean.com/products/spaces/)
- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Llama-3.1 Model Documentation](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

---

## 📞 Support

For issues specific to this deployment:
- Check GPU droplet logs: `journalctl -u your-service`
- Monitor Spaces usage: Digital Ocean console
- GPU debugging: `nvidia-smi`, `nvidia-debugdump`

**🎉 Congratulations! You now have a fully deployed Vietnamese Legal LLM system on Digital Ocean! 🇻🇳⚖️🤖**