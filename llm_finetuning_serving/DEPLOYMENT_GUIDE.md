# 🚀 Vietnamese Legal LLM - Hybrid Cloud Deployment Guide

Hướng dẫn chi tiết để finetune Vietnamese Legal LLM trên Digital Ocean và serving trên AWS EC2 với data/model lưu trên Spaces.

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
                       │    AWS EC2          │
                       │ GPU Instance        │
                       │ (Serving)           │
                       └─────────────────────┘
```

### Workflow:

1. **Local**: Process data → Upload to Digital Ocean Spaces
2. **Training GPU Droplet (DO)**: Download data → Train → Upload model
3. **Serving GPU Instance (AWS EC2)**: Download model → Serve API

## 🏋️ Part 2: Training on GPU Droplet

### 2.1 Create Training GPU Droplet

**Recommended Configuration:**

- **Image**: Ubuntu 22.04 LTS
- **Size**: GPU-H100-80GB or GPU-H200-141GB
- **Region**: Choose based on your location
- **Add your SSH key**

### 2.3 Clone and Setup Repository

```bash
# SSH back in after reboot
ssh root@your-training-droplet-ip

# Verify GPU
nvidia-smi

# Clone repository
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving

# Setup environment
apt install python3-pip


python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment variables
cp .env.template .env
nano .env
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
# Check: https://wandb.ai/mikeethanh04/vietnamese-legal-llm
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

## 🌐 Part 3: Serving on AWS EC2

### 3.1 Create AWS EC2 Instance

**Recommended Instance Types:**

**GPU Instances (High Performance):**

- **For Production**: `g5.xlarge` (1 GPU, 4 vCPUs, 16GB RAM) - ~$1.006/hour
- **For High Performance**: `g5.2xlarge` (1 GPU, 8 vCPUs, 32GB RAM) - ~$2.013/hour
- **For Cost-Effective GPU**: `g4dn.xlarge` (1 GPU, 4 vCPUs, 16GB RAM) - ~$0.526/hour
- **For Heavy Workloads**: `p3.2xlarge` (1 V100 GPU, 8 vCPUs, 61GB RAM) - ~$3.06/hour

**CPU Instances (Cost-Effective):**

- **For Development**: `c5.xlarge` (4 vCPUs, 8GB RAM) - ~$0.17/hour
- **For Production**: `c5.2xlarge` (8 vCPUs, 16GB RAM) - ~$0.34/hour
- **For Heavy CPU**: `c5.4xlarge` (16 vCPUs, 32GB RAM) - ~$0.68/hour
- **Memory Optimized**: `r5.xlarge` (4 vCPUs, 32GB RAM) - ~$0.252/hour

**Launch Configuration:**

1. **AMI**:
   - **For GPU**: Deep Learning AMI GPU PyTorch 2.1.0 (Ubuntu 22.04)
   - **For CPU**: Ubuntu Server 22.04 LTS
2. **Instance Type**: Choose based on your performance/cost needs
3. **Key Pair**: Select your existing key pair or create new one
4. **Security Group**: Allow SSH (22), HTTP (80), HTTPS (443), Custom (7000)
5. **Storage**: 50GB gp3 SSD minimum (CPU), 100GB gp3 SSD (GPU)
6. **Region**: Choose based on your target audience (us-east-1 recommended)

### 3.2 Setup EC2 Instance

```bash
# SSH into EC2 instance
ssh -i "minh.pem" ubuntu@3.89.75.45

# Update system
sudo apt update && sudo apt upgrade -y

# Install additional dependencies
sudo apt install -y python3-pip python3-venv git htop tree

# Clone repository
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/llm_finetuning_serving

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements_serving_cpu.txt

# Install additional AWS-specific packages
pip install boto3 awscli
```

### 3.3 Configure AWS Credentials & Environment

```bash
# Configure Digital Ocean Spaces for model download
cp .env.template .env
nano .env
```

**Edit `.env` file:**

```bash
# Digital Ocean Spaces configuration
DO_SPACES_KEY=your_spaces_key
DO_SPACES_SECRET=your_spaces_secret
DO_SPACES_ENDPOINT=https://sfo3.digitaloceanspaces.com
DO_SPACES_BUCKET=legal-datalake

# Model configuration
MODEL_PATH=/home/ubuntu/model
MODEL_NAME=vietnamese-legal-llama-20251111_115138
HOST=0.0.0.0
PORT=6000

```

### 3.4 Download Model and Start Serving

```bash
# Create model directory
mkdir -p /home/ubuntu/model

# Make script executable
chmod +x ./run_pipeline.sh

# Download specific model from Digital Ocean Spaces
./run_pipeline.sh download-model vietnamese-legal-llama-20251111_115138 /home/ubuntu/model

# Verify model download
ls -la /home/ubuntu/model/

# Set up serving environment
cd serving
export MODEL_PATH="/home/ubuntu/model"

# For CPU serving:
export USE_GPU=false
export DEVICE=cpu

# For GPU serving (uncomment if using GPU instance):
# export USE_GPU=true
# export DEVICE=cuda
# export CUDA_VISIBLE_DEVICES=0

# Test device availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CPU cores available: {torch.get_num_threads()}')
print('Using CPU for inference - No GPU required!')
"

# Start serving on CPU (recommended for cost-effective deployment)
# First, install missing packages
pip install python-dotenv accelerate

python3 serve_model_cpu.py

python.exe do_spaces_manager.py download-model vietnamese-legal-llama-20251111_115138
# Expected output:
# Loading model from /home/ubuntu/model for CPU inference...
# Model loaded successfully on CPU!
# Model parameters: 8,030,261,248
# Starting server on 0.0.0.0:6000

# Alternative: For GPU serving (if using GPU instance)
# export USE_GPU=true
# export DEVICE=cuda
# export CUDA_VISIBLE_DEVICES=0
# python3 serve_model.py
```

**Hoặc dùng AWS CLI:**

```bash
# Get Security Group ID
aws ec2 describe-instances --instance-ids i-04bfa9643df64c40d \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId'

# Add rules (thay YOUR_SG_ID)
aws ec2 authorize-security-group-ingress \
  --group-id sg-066b6004908ea401a \
  --protocol tcp \
  --port 6000 \
  --cidr 0.0.0.0/0
```

### 3.7 Test API

**From Linux/Mac/WSL:**

```bash
# From another terminal or local machine
curl -X POST http://3.90.175.145:6000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Quy định về thời hiệu khởi kiện là gì?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

**From Windows Command Prompt:**

```cmd
curl -X POST http://3.89.75.45:6000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Quy định về thời hiệu khởi kiện là gì?\"}], \"temperature\": 0.7, \"max_tokens\": 512}"
```

**From PowerShell:**

```powershell
$body = @{
    messages = @(
        @{
            role = "user"
            content = "Quy định về thời hiệu khởi kiện là gì?"
        }
    )
    temperature = 0.7
    max_tokens = 512
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://3.90.175.145:6000/v1/chat/completions" -Method POST -Body $body -ContentType "application/json"
```

**Test health endpoint:**

```bash
# Linux/Mac/WSL
curl http://3.89.75.45:6000/health

# Windows Command Prompt
curl http://localhost:6000/health

# PowerShell
Invoke-RestMethod -Uri "http://3.90.175.145:6000/health"
```

curl http://localhost:6000/health

# Test from local Python

python3 -c "
import requests
response = requests.post('http://your-ec2-public-ip:7000/v1/chat/completions',
headers={'Content-Type': 'application/json'},
json={'messages': [{'role': 'user', 'content': 'Quyền và nghĩa vụ của công dân là gì?'}],
'temperature': 0.7, 'max_tokens': 512})
print(response.json())
"

````

### 3.7 Production Setup & Security

```bash
# Configure firewall
sudo ufw allow ssh
sudo ufw allow 7000
sudo ufw enable

# Set up reverse proxy with Nginx (optional)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/vietnamese-legal-llm
````

**Nginx configuration:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:7000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/vietnamese-legal-llm /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# Optional: Install SSL certificate with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
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

**1. CUDA Out of Memory (EC2)**

```bash
# Check GPU memory usage
nvidia-smi
# Reduce batch size or model precision
export CUDA_VISIBLE_DEVICES=0
# Consider 4-bit quantization
# load_in_4bit=True in serving config
# Restart instance if needed
sudo reboot
```

**2. Model Download Failed**

```bash
# Check Digital Ocean Spaces credentials
python do_spaces_manager.py list models/
# Verify network connectivity from EC2
ping sfo3.digitaloceanspaces.com
# Check IAM permissions if using S3
aws s3 ls
```

**3. EC2 Instance Issues**

```bash
# Check instance status
aws ec2 describe-instance-status --instance-ids i-1234567890abcdef0
# Check system logs
sudo dmesg | tail -50
# Check disk space
df -h
# Check memory usage
free -h
```

**4. GPU Driver Issues**

```bash
# Verify GPU is detected
lspci | grep -i nvidia
# Check driver version
nvidia-smi
# Reinstall drivers if needed (if not using Deep Learning AMI)
sudo apt purge nvidia-*
sudo apt autoremove
```

**5. API Connection Issues**

```bash
# Check EC2 security groups
aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
# Check service status
sudo systemctl status vietnamese-legal-llm
# Check port binding
sudo netstat -tlnp | grep 7000
# Test local connectivity
curl localhost:7000/health
```

**6. CPU Serving Specific Issues**

```bash
# Memory issues with CPU serving
free -h
# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Slow CPU performance
htop  # Check CPU utilization
# Consider upgrading to higher CPU instance
# Or add more concurrent workers
```

**7. Performance Optimization**

```bash
# Monitor GPU utilization
watch -n1 nvidia-smi
# Monitor system resources
htop
# Scale up instance type if needed
# Use multiple instances with load balancer
```

---

## 💰 Cost Optimization

### Hybrid Cloud Cost Management

**Training Strategy (Digital Ocean):**

- Create H200 droplet for training only
- Destroy after model upload
- Estimated cost: $10-20 for 6-hour training session

**Serving Strategy (AWS EC2):**

**CPU Instances (Most Cost-Effective):**

- **c5.xlarge**: ~$122/month (development)
- **c5.2xlarge**: ~$245/month (production)
- **c5.4xlarge**: ~$490/month (high-performance CPU)
- **r5.xlarge**: ~$181/month (memory optimized)

**GPU Instances (High Performance):**

- **g4dn.xlarge**: ~$380/month (cost-effective GPU)
- **g5.xlarge**: ~$720/month (balanced performance)
- **g5.2xlarge**: ~$1,440/month (high performance)

**Cost Comparison:**

- **CPU Serving**: 70-90% cost savings vs GPU, slower inference (10-30s)
- **GPU Serving**: Higher cost, fast inference (1-5s)
- Use Spot Instances for 60-90% additional cost savings

**AWS Cost-Saving Tips:**

- Use Spot Instances for development/testing
- Schedule instances with Lambda/CloudWatch for auto start/stop
- Use Reserved Instances for 1-year+ commitments (save 30-60%)
- Monitor GPU utilization with CloudWatch
- Use Auto Scaling Groups for dynamic scaling
- Store models in S3 with Intelligent Tiering
- Use Application Load Balancer for multiple instances

**Estimated Monthly Costs:**

- **Development**: g4dn.xlarge Spot ~ $80-150/month
- **Production**: g5.xlarge Reserved ~ $400-500/month
- **Enterprise**: g5.2xlarge with Load Balancer ~ $1,500-2,000/month

---

## 🎯 Success Checklist

### Training Completion ✅

- [ ] Data uploaded to Digital Ocean Spaces
- [ ] GPU droplet created and configured
- [ ] Model trained successfully (4-6 hours)
- [ ] Model uploaded to Spaces
- [ ] Training droplet destroyed (cost saving)

### AWS EC2 Serving Setup ✅

- [ ] EC2 GPU instance launched (g5.xlarge recommended)
- [ ] Deep Learning AMI configured
- [ ] Security groups configured (SSH, port 7000)
- [ ] Model downloaded from Spaces to EC2
- [ ] GPU drivers and CUDA working
- [ ] API server running on EC2
- [ ] Health checks passing
- [ ] API responses working from public IP

### Production Ready ✅

- [ ] Systemd service configured for auto-start
- [ ] Nginx reverse proxy setup (optional)
- [ ] Domain name configured
- [ ] SSL certificate installed (Let's Encrypt)
- [ ] CloudWatch monitoring enabled
- [ ] Auto Scaling Group configured (if needed)
- [ ] Backup and disaster recovery plan
- [ ] Cost monitoring and alerts configured

---

## 📚 Additional Resources

**Digital Ocean (Training):**

- [Digital Ocean GPU Droplets Documentation](https://docs.digitalocean.com/products/droplets/how-to/create-gpu-droplets/)
- [Digital Ocean Spaces Documentation](https://docs.digitalocean.com/products/spaces/)

**AWS EC2 (Serving):**

- [AWS EC2 GPU Instances Documentation](https://docs.aws.amazon.com/ec2/latest/userguide/accelerated-computing-instances.html)
- [AWS Deep Learning AMI Documentation](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
- [AWS Auto Scaling Documentation](https://docs.aws.amazon.com/autoscaling/ec2/userguide/what-is-amazon-ec2-auto-scaling.html)
- [AWS Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)
- [AWS CloudWatch Monitoring](https://docs.aws.amazon.com/cloudwatch/latest/monitoring/)

**Model & Frameworks:**

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Llama-3.1 Model Documentation](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [PyTorch GPU Documentation](https://pytorch.org/get-started/locally/)

---

## 📞 Support

For issues specific to this hybrid deployment:

**Training Issues (Digital Ocean):**

- Check GPU droplet logs: `journalctl -u your-service`
- Monitor Spaces usage: Digital Ocean console

**Serving Issues (AWS EC2):**

- Check service logs: `sudo journalctl -u vietnamese-legal-llm -f`
- Monitor EC2 metrics: AWS CloudWatch console
- GPU debugging: `nvidia-smi`, `watch -n1 nvidia-smi`
- Instance debugging: `htop`, `df -h`, `free -h`

**Common AWS EC2 Troubleshooting:**

- Security group configuration
- EC2 instance status checks
- EBS volume space issues
- GPU driver compatibility

**🎉 Congratulations! You now have a fully deployed Vietnamese Legal LLM system with hybrid cloud architecture - Training on Digital Ocean and Serving on AWS EC2! 🇻🇳⚖️🤖**
