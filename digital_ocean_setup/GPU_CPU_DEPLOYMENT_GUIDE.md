# 🚀 Hướng dẫn sử dụng GPU Droplet cho Training + CPU Droplet cho Serving

## 🎯 Chiến lược

**Training**: GPU Droplet (tạm thời, xóa sau khi train xong) → Tiết kiệm chi phí
**Serving**: CPU Droplet (chạy lâu dài) → Ổn định và rẻ

## Bước 1: Tạo GPU Droplet cho Training

### 1.1. Vào Digital Ocean Dashboard
1. Login: https://cloud.digitalocean.com/
2. Click **"Create"** → **"Droplets"**

### 1.2. Cấu hình GPU Droplet
1. **Operating System**: Ubuntu 22.04 (LTS) x64
2. **Plan**: Click tab **"Premium Intel with GPU"**
3. **GPU Options**:
   - **Basic GPU**: $72/month - 1 vCPU, 8GB RAM, 1 GPU (NVIDIA V100) ← Khuyến nghị
   - **Professional GPU**: $144/month - 2 vCPUs, 16GB RAM, 1 GPU (NVIDIA V100) ← Nếu cần performance cao
4. **Datacenter**: Singapore (SGP1)
5. **Authentication**: SSH Key (khuyến nghị)
6. **Hostname**: `legal-ai-gpu-training`
7. **Tags**: `gpu`, `training`, `temporary`
8. Click **"Create Droplet"**

### 1.3. Ghi lại thông tin
- **GPU Droplet IP**: `_______________`

---

## Bước 2: Tạo CPU Droplet cho Serving

### 2.1. Tạo CPU Droplet thứ 2
1. Click **"Create"** → **"Droplets"** (lần 2)
2. **Operating System**: Ubuntu 22.04 (LTS) x64
3. **Plan**: Basic - Regular with SSD
4. **Size**: 
   - **$24/month**: 4GB RAM, 2 vCPUs, 80GB SSD ← Cho serving cơ bản
   - **$48/month**: 8GB RAM, 4 vCPUs, 160GB SSD ← Khuyến nghị cho performance tốt
5. **Datacenter**: Singapore (SGP1) (cùng region)
6. **Authentication**: SSH Key (dùng cùng key)
7. **Hostname**: `legal-ai-cpu-serving`
8. **Tags**: `cpu`, `serving`, `production`
9. Click **"Create Droplet"**

### 2.2. Ghi lại thông tin
- **CPU Droplet IP**: `_______________`

---

## Bước 3: Setup GPU Droplet cho Training

### 3.1. Kết nối GPU Droplet
```bash
ssh root@GPU_DROPLET_IP
```

### 3.2. Cài đặt NVIDIA Drivers và CUDA
```bash
# Update system
apt update && apt upgrade -y

# Install NVIDIA drivers
apt install -y ubuntu-drivers-common
ubuntu-drivers autoinstall

# Reboot để load drivers
reboot
```

**Đợi 2-3 phút và kết nối lại:**
```bash
ssh root@GPU_DROPLET_IP
```

### 3.3. Verify GPU
```bash
# Check GPU
nvidia-smi
```

### 3.4. Cài đặt Docker với GPU support
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt update && apt install -y nvidia-docker2
systemctl restart docker

# Test GPU in Docker
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu22.04 nvidia-smi
```

---

## Bước 4: Setup Repository trên GPU Droplet

### 4.1. Clone repository
```bash
cd /root
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
```

### 4.2. Cấu hình environment
```bash
cp .env.template .env
nano .env
```

### 4.3. Tạo thư mục cần thiết
```bash
mkdir -p data models logs
```

---

## Bước 5: Setup GPU Training Environment

### 5.1. Kiểm tra file training script
```bash
# Kiểm tra file training script có tồn tại
ls -la train_embedding_gpu.py
```

### 5.2. Kiểm tra requirements file
```bash
# Sử dụng requirements_gpu.txt cho GPU training
ls -la requirements_gpu.txt
```
## Bước 6: Training trên GPU Droplet

### 6.1. Load environment variables
```bash
# Load environment variables
source .env

# Verify environment variables
echo "🔍 Checking environment variables..."
echo "SPACES_ACCESS_KEY: ${SPACES_ACCESS_KEY:0:10}..." 
echo "SPACES_SECRET_KEY: ${SPACES_SECRET_KEY:0:10}..."
echo "SPACES_BUCKET: $SPACES_BUCKET"
echo "BASE_MODEL: $BASE_MODEL"
```

### 6.2. Build GPU image
```bash
# Build với verbose output để debug
docker build -f Dockerfile.gpu-training -t legal-embedding-gpu:latest . --no-cache

# Kiểm tra image đã build thành công
docker images | grep legal-embedding-gpu
```

### 6.3. Test GPU access trong Docker
```bash
# Test GPU trước khi training
docker run --rm --gpus all legal-embedding-gpu:latest nvidia-smi

# Test PyTorch GPU access
docker run --rm --gpus all legal-embedding-gpu:latest python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"
```

### 6.4. Run training với GPU (method 1: Direct run)
```bash
# Đảm bảo đang ở đúng thư mục
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Load environment variables
source .env

# Run training với full config từ .env
docker run --gpus all \
  --name legal-gpu-training \
  -v $(pwd)/data:/tmp/data \
  -v $(pwd)/models:/tmp/model \
  -v $(pwd)/logs:/tmp/logs \
  -v $(pwd)/.env:/app/.env:ro \
  -e SPACES_ACCESS_KEY="$SPACES_ACCESS_KEY" \
  -e SPACES_SECRET_KEY="$SPACES_SECRET_KEY" \
  -e SPACES_ENDPOINT="$SPACES_ENDPOINT" \
  -e SPACES_BUCKET="$SPACES_BUCKET" \
  -e BASE_MODEL="$BASE_MODEL" \
  -e MODEL_PATH="$MODEL_PATH" \
  -e EPOCHS="$EPOCHS" \
  -e GPU_BATCH_SIZE="$GPU_BATCH_SIZE" \
  -e LEARNING_RATE="$LEARNING_RATE" \
  -e WARMUP_STEPS="$WARMUP_STEPS" \
  -e MAX_SEQ_LENGTH="$MAX_SEQ_LENGTH" \
  -e GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS" \
  -e USE_FP16="$USE_FP16" \
  -e USE_GPU="$USE_GPU" \
  -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  -e MAX_SAMPLES="$MAX_SAMPLES" \
  --rm \
  legal-embedding-gpu:latest
```


### 6.6. Monitor training (trong terminal khác)
```bash
# Terminal 1: Monitor GPU usage
watch -n 5 nvidia-smi

# Terminal 2: Monitor Docker logs
docker logs -f legal-gpu-training

# Terminal 3: Monitor system resources
watch -n 10 'echo "=== CPU/Memory ===" && top -n 1 | head -5 && echo "=== Disk ===" && df -h'
```

### 6.7. Training progress tracking
```bash
# Kiểm tra training progress
ls -la models/
tail -f logs/training.log

# Kiểm tra model đã upload lên Spaces
# (Nếu training script có auto-upload)
```

**⏰ Thời gian training:**
- **GPU V100**: 15-30 phút
- **GPU H100**: 8-15 phút  
- **Batch size**: Có thể tăng lên 128+ với GPU H100

**🚨 Troubleshooting:**
```bash
# Nếu training fail, check logs
docker logs legal-gpu-training

# Kiểm tra GPU memory
nvidia-smi

# Restart nếu cần
docker stop legal-gpu-training
docker rm legal-gpu-training
# Rồi chạy lại command ở bước 6.4
```

---

## Bước 7: Setup CPU Droplet và Download Model từ Spaces

### 7.1. Kết nối CPU Droplet
```bash
ssh root@CPU_DROPLET_IP
```

### 7.2. Cài đặt Docker
```bash
# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Verify Docker installation
docker --version
docker run hello-world
```

### 7.3. Clone repository
```bash
cd /root
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
```

### 7.4. Cấu hình environment cho serving
```bash
# Copy template
cp .env.serving.template .env.serving

# Edit với thông tin của bạn
nano .env.serving
```

**Cần điền các thông tin sau trong `.env.serving`:**
```bash
SPACES_ACCESS_KEY=your_access_key_here
SPACES_SECRET_KEY=your_secret_key_here
SPACES_ENDPOINT=https://sgp1.digitaloceanspaces.com
SPACES_BUCKET=legal-datalake

# Để trống để tự động lấy model mới nhất
MODEL_PATH=

# API config
API_HOST=0.0.0.0
API_PORT=5000
MAX_BATCH_SIZE=32
```

### 7.5. Download model từ Spaces
```bash
# Load environment variables
source .env.serving

# Tạo thư mục models
mkdir -p models logs

# Download model (sẽ tự động lấy model mới nhất)
python3 download_model_from_spaces.py

# Verify model đã download
ls -la models/
```

**💡 Lưu ý:**
- Script sẽ tự động list tất cả models có sẵn trong Spaces
- Nếu không chỉ định `MODEL_PATH`, nó sẽ chọn model mới nhất
- Model sẽ được download vào thư mục `./models/`

---

## Bước 8: Deploy Serving API trên CPU Droplet

### 8.1. Build Docker image
```bash
# Đảm bảo đang ở đúng thư mục
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Build image
docker build -f Dockerfile.cpu-serving -t legal-embedding-serving:latest .

# Verify image đã build
docker images | grep legal-embedding-serving
```

### 8.2. Chạy API với Docker Compose (Recommended)
```bash
# Start service với docker-compose
docker-compose -f docker-compose.serving.yml up -d

# Check logs
docker-compose -f docker-compose.serving.yml logs -f

# Check status
docker-compose -f docker-compose.serving.yml ps
```

### 8.3. Hoặc chạy trực tiếp với Docker (Alternative)
```bash
# Run container
docker run -d \
  --name legal-embedding-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  --env-file .env.serving \
  --restart unless-stopped \
  legal-embedding-serving:latest

# Check logs
docker logs -f legal-embedding-api

# Check container status
docker ps | grep legal-embedding-api
```

### 8.4. Verify API is running
```bash
# Test health endpoint
curl http://localhost:5000/health

# Expected output:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cpu",
#   "embedding_dim": 1024
# }
```

**⏰ Thời gian khởi động:**
- **Model loading**: 30-60 giây
- **API ready**: 1-2 phút

---

## Bước 9: Test và Monitor Serving API

### 9.1. Chạy test suite
```bash
# Test từ local (trên CPU droplet)
python3 test_api.py http://localhost:5000

# Test từ máy khác (thay YOUR_CPU_DROPLET_IP)
python3 test_api.py http://YOUR_CPU_DROPLET_IP:5000
```

### 9.2. Test thủ công với curl

**Test embedding:**
```bash
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Luật Dân sự năm 2015",
      "Bộ luật Hình sự năm 2017"
    ]
  }'
```

**Test similarity:**
```bash
curl -X POST http://localhost:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts1": ["Luật Dân sự về quyền sở hữu"],
    "texts2": ["Quy định về tài sản", "Luật Hình sự"]
  }'
```

### 9.3. Monitor API logs
```bash
# Với docker-compose
docker-compose -f docker-compose.serving.yml logs -f

# Với docker run
docker logs -f legal-embedding-api

# Hoặc check file logs
tail -f logs/serve_model.log
```

### 9.4. Monitor system resources
```bash
# Monitor CPU, Memory
watch -n 5 'top -n 1 | head -20'

# Monitor Docker stats
docker stats legal-embedding-api

# Check disk usage
df -h
```

### 9.5. Cấu hình Firewall (Optional nhưng recommended)
```bash
# Allow SSH
ufw allow OpenSSH

# Allow API port
ufw allow 5000/tcp

# Enable firewall
ufw enable

# Check status
ufw status
```

---

## Bước 10: Cleanup và Best Practices

### 10.1. Xóa GPU Droplet sau khi training xong
```bash
# Sau khi model đã upload lên Spaces và verify thành công
# Vào Digital Ocean Dashboard:
# 1. Chọn GPU Droplet
# 2. Click "Destroy"
# 3. Confirm deletion
# 
# Lý do: GPU Droplet rất đắt ($72-144/month)
# Chỉ cần trong quá trình training
```

### 10.2. Backup và Update Model

**Khi có model mới:**
```bash
# SSH vào CPU droplet
ssh root@CPU_DROPLET_IP
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Backup model cũ (optional)
mv models models_backup_$(date +%Y%m%d)

# Download model mới
mkdir -p models
python3 download_model_from_spaces.py

# Restart service
docker-compose -f docker-compose.serving.yml restart

# Verify
curl http://localhost:5000/health
```

### 10.3. Auto-restart policy
```bash
# Docker Compose đã config restart: unless-stopped
# Container sẽ tự động restart nếu:
# - Droplet reboot
# - Container crash
# - Docker daemon restart
```

### 10.4. Best Practices

**✅ Security:**
- Sử dụng SSH key thay vì password
- Enable firewall với `ufw`
- Giới hạn access API bằng IP whitelist hoặc API key
- Định kỳ update security patches: `apt update && apt upgrade`

**✅ Performance:**
- Monitor CPU/Memory usage định kỳ
- Adjust `MAX_BATCH_SIZE` dựa trên RAM available
- Consider upgrade droplet nếu performance không đủ

**✅ Cost Optimization:**
- **Xóa GPU droplet ngay** sau training
- CPU droplet: $24-48/month (rẻ hơn nhiều)
- Backup models lên Spaces (cheap storage)

**✅ Monitoring:**
```bash
# Setup simple monitoring script
cat > /root/monitor.sh << 'EOF'
#!/bin/bash
while true; do
  echo "=== $(date) ==="
  curl -s http://localhost:5000/health || echo "API DOWN!"
  docker stats --no-stream legal-embedding-api
  echo ""
  sleep 300  # Check every 5 minutes
done
EOF

chmod +x /root/monitor.sh

# Run in background
nohup /root/monitor.sh > /root/monitor.log 2>&1 &
```

### 10.5. Troubleshooting Common Issues

**API không start:**
```bash
# Check logs
docker logs legal-embedding-api

# Common issues:
# 1. Model không tồn tại -> Download lại
# 2. Out of memory -> Reduce MAX_BATCH_SIZE hoặc upgrade droplet
# 3. Port conflict -> Change API_PORT trong .env.serving
```

**Performance chậm:**
```bash
# Check system resources
htop
docker stats

# Solutions:
# 1. Upgrade to bigger droplet (4GB -> 8GB RAM)
# 2. Reduce MAX_BATCH_SIZE
# 3. Optimize model (quantization - advanced)
```

**Model outdated:**
```bash
# Download và deploy model mới
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
python3 download_model_from_spaces.py
docker-compose -f docker-compose.serving.yml restart
```

---

## 📊 Chi phí dự kiến

| Service | Size | Cost/month | Usage |
|---------|------|------------|-------|
| **GPU Droplet** | V100, 8GB RAM | $72 | **Tạm thời** (1-2 giờ) ≈ $0.1 |
| **CPU Droplet** | 4GB RAM, 2 vCPUs | $24 | **Lâu dài** |
| **CPU Droplet** | 8GB RAM, 4 vCPUs | $48 | **Recommended** |
| **Spaces Storage** | 250GB | $5 | Models + Data |

**💰 Total Cost: ~$29-53/month** (chỉ trả CPU serving + storage)

---

## 🎉 Hoàn thành!

Bạn đã có:
- ✅ GPU Droplet để training (xóa sau khi xong)
- ✅ Model được lưu an toàn trên Spaces
- ✅ CPU Droplet serving API 24/7
- ✅ Chi phí tối ưu (~$29-53/month)

**Next steps:**
- Integrate API vào backend của bạn
- Setup monitoring & alerting
- Consider load balancer nếu traffic cao

