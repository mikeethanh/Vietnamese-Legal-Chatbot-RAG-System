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

## Bước 7: Setup CPU Droplet

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
# Copy template (nếu có) hoặc tạo mới
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

### 7.5. Tạo thư mục cần thiết
```bash
# Tạo thư mục models và logs
mkdir -p models logs
```

---

## Bước 8: Build Docker Image và Download Model

### 8.1. Build Docker image trước
```bash
# Đảm bảo đang ở đúng thư mục
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Build image (image này đã có Python và tất cả dependencies cần thiết)
docker build -f Dockerfile.cpu-serving -t legal-embedding-serving:latest .

# Verify image đã build
docker images | grep legal-embedding-serving
```

**💡 Lưu ý:** Image này đã bao gồm:
- Python 3.10
- Tất cả dependencies trong `requirements_serving.txt`
- Script `download_model_from_spaces.py`
- Script `serve_model.py`

### 8.2. Download model từ Spaces bằng Docker container
```bash
# Load environment variables
source .env.serving

# Chạy container để download model (dùng image vừa build)
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e SPACES_ACCESS_KEY="$SPACES_ACCESS_KEY" \
  -e SPACES_SECRET_KEY="$SPACES_SECRET_KEY" \
  -e SPACES_ENDPOINT="$SPACES_ENDPOINT" \
  -e SPACES_BUCKET="$SPACES_BUCKET" \
  -e MODEL_PATH="$MODEL_PATH" \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# Verify model đã download
ls -la models/
```

**💡 Giải thích:**
- `--rm`: Tự động xóa container sau khi chạy xong
- `-v $(pwd)/models:/app/models`: Mount thư mục models để lưu file download
- Các `-e`: Pass environment variables vào container
- `python download_model_from_spaces.py`: Override CMD để chạy script download thay vì serve

**📋 Output mong đợi:**
```
✅ Đã kết nối với Spaces: https://sgp1.digitaloceanspaces.com
📋 Liệt kê models có sẵn trong bucket 'legal-datalake'...
✅ Tìm thấy X model(s):
   1. models/legal-embedding-v1
   2. models/legal-embedding-v2
📥 Đang download model từ 'models/legal-embedding-v2'...
...
✅ Download hoàn tất!
```

### 8.3. Deploy Serving API với Docker Compose (Recommended)
```bash
# Start service với docker-compose
docker-compose -f docker-compose.serving.yml up -d

# Check logs
docker-compose -f docker-compose.serving.yml logs -f

# Check status
docker-compose -f docker-compose.serving.yml ps
```

### 8.4. Hoặc chạy trực tiếp với Docker (Alternative)
```bash
# Run container để serving API
docker run -d \
  --name legal-embedding-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_PATH=/app/models \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=5000 \
  -e MAX_BATCH_SIZE=32 \
  --restart unless-stopped \
  legal-embedding-serving:latest

# Check logs
docker logs -f legal-embedding-api

# Check container status
docker ps | grep legal-embedding-api
```

### 8.5. Verify API is running
```bash
# Test health endpoint từ trong droplet
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

### 8.6. 🔥 MỞ FIREWALL để Serving ra bên ngoài

**Bước này RẤT QUAN TRỌNG** - Nếu không làm thì API chỉ chạy local!

```bash
# Kiểm tra firewall status
ufw status

# Nếu firewall chưa active hoặc chưa config:
# Allow SSH (QUAN TRỌNG - làm trước khi enable ufw!)
ufw allow OpenSSH
ufw allow 22/tcp

# Allow API port
ufw allow 5000/tcp

# Enable firewall
ufw --force enable

# Verify firewall rules
ufw status verbose
```

**📋 Expected output:**
```
Status: active

To                         Action      From
--                         ------      ----
22/tcp                     ALLOW       Anywhere
5000/tcp                   ALLOW       Anywhere
OpenSSH                    ALLOW       Anywhere
```

### 8.7. 🌐 Test API từ bên ngoài

```bash
# Test từ máy local của bạn (thay YOUR_DROPLET_IP)
curl http://YOUR_DROPLET_IP:5000/health

# Test embedding endpoint
curl -X POST http://YOUR_DROPLET_IP:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Luật Dân sự năm 2015"]
  }'
```

**✅ Nếu thành công, bạn sẽ thấy:**
- `/health` trả về status healthy
- `/embed` trả về array of embeddings

**🔧 Troubleshooting:**
```bash
# Nếu không kết nối được từ bên ngoài:

# 1. Check container có đang chạy không
docker ps | grep legal-embedding-api

# 2. Check port mapping
docker port legal-embedding-api

# 3. Check firewall
ufw status verbose

# 4. Check logs
docker logs legal-embedding-api

# 5. Test từ trong droplet trước
curl http://localhost:5000/health

# 6. Nếu model không tồn tại, download lại
docker run --rm \
  -v $(pwd)/models:/app/models \
  --env-file .env.serving \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# 7. Restart API container
docker restart legal-embedding-api
```

**💡 Lưu ý bảo mật:**
- Port 5000 đang mở công khai ra internet
- Consider thêm authentication/API key nếu cần
- Hoặc chỉ allow IP cụ thể:
```bash
# Chỉ allow từ IP cụ thể
ufw delete allow 5000/tcp
ufw allow from YOUR_BACKEND_IP to any port 5000
```

---

## Bước 9: Test và Monitor Serving API

### 9.1. Test nhanh với script
```bash
# Trên máy local, clone repo nếu chưa có
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
```

### 9.2. Chạy test suite đầy đủ (Python)
```bash
# Test từ local (trên máy local của bạn, không phải droplet)
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
python3 test_api.py http://YOUR_DROPLET_IP:5000
```

### 9.3. Test thủ công với curl

**🔍 Endpoint 1: Health Check**
```bash
# Check xem API có sống không
curl http://YOUR_DROPLET_IP:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "embedding_dim": 1024,
  "timestamp": 1234567890.123
}
```

**📝 Endpoint 2: Generate Embeddings**
```bash
# Tạo embedding vectors cho text
curl -X POST http://YOUR_DROPLET_IP:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Luật Dân sự năm 2015",
      "Bộ luật Hình sự năm 2017"
    ]
  }'
```

**Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ...],  // 1024 dimensions
    [0.234, -0.567, 0.890, ...]   // 1024 dimensions
  ],
  "processing_time": 0.123,
  "count": 2
}
```

**🔢 Endpoint 3: Calculate Similarity**
```bash
# Tính độ tương đồng giữa các câu
curl -X POST http://YOUR_DROPLET_IP:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts1": ["Luật Dân sự về quyền sở hữu"],
    "texts2": ["Luật Dân  sự"]
  }'
```

**Response:**
```json
{
  "similarities": [
    [0.85, 0.23]  // similarities[i][j] = similarity(texts1[i], texts2[j])
  ],
  "processing_time": 0.089,
  "shape": [1, 2]
}
```

**💡 Cách sử dụng trong code Python:**
```python
import requests

API_URL = "http://YOUR_DROPLET_IP:5000"

# 1. Generate embeddings
response = requests.post(
    f"{API_URL}/embed",
    json={"texts": ["Luật Dân sự", "Bộ luật Hình sự"]}
)
embeddings = response.json()["embeddings"]
print(f"Got {len(embeddings)} embeddings")

# 2. Calculate similarity
response = requests.post(
    f"{API_URL}/similarity",
    json={
        "texts1": ["Quyền sở hữu tài sản"],
        "texts2": ["Tài sản riêng", "Tài sản chung", "Quyền kế thừa"]
    }
)
similarities = response.json()["similarities"]
print(f"Similarities: {similarities}")

# 3. Find most similar
query = "Luật về đất đai"
candidates = ["Quy định về nhà đất", "Bộ luật Hình sự", "Luật Đất đai 2013"]

response = requests.post(
    f"{API_URL}/similarity",
    json={"texts1": [query], "texts2": candidates}
)
scores = response.json()["similarities"][0]
best_idx = scores.index(max(scores))
print(f"Most similar: {candidates[best_idx]} (score: {scores[best_idx]:.3f})")
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

### 9.5. Cấu hình Firewall (BẮT BUỘC cho production)
```bash
# Kiểm tra firewall hiện tại
ufw status

# Allow SSH (QUAN TRỌNG - phải làm trước!)
ufw allow OpenSSH
ufw allow 22/tcp

# Allow API port
ufw allow 5000/tcp

# Enable firewall
ufw --force enable

# Verify
ufw status verbose
```

**🔒 Tùy chọn bảo mật cao hơn:**
```bash
# Chỉ allow API từ IP backend của bạn
ufw delete allow 5000/tcp
ufw allow from YOUR_BACKEND_SERVER_IP to any port 5000

# Hoặc allow từ một subnet
ufw allow from 10.0.0.0/8 to any port 5000

# Rate limiting để chống DDoS
ufw limit 5000/tcp
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
mkdir -p models

# Download model mới bằng Docker container
source .env.serving
docker run --rm \
  -v $(pwd)/models:/app/models \
  -e SPACES_ACCESS_KEY="$SPACES_ACCESS_KEY" \
  -e SPACES_SECRET_KEY="$SPACES_SECRET_KEY" \
  -e SPACES_ENDPOINT="$SPACES_ENDPOINT" \
  -e SPACES_BUCKET="$SPACES_BUCKET" \
  -e MODEL_PATH="$MODEL_PATH" \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# Restart service
docker-compose -f docker-compose.serving.yml restart
# Hoặc nếu dùng docker run:
# docker restart legal-embedding-api

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
# Download và deploy model mới bằng Docker container
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
source .env.serving

docker run --rm \
  -v $(pwd)/models:/app/models \
  --env-file .env.serving \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

docker-compose -f docker-compose.serving.yml restart
# Hoặc: docker restart legal-embedding-api
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

