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

## Bước 7: Setup CPU Droplet cho Serving

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
# Tạo file .env.serving với cấu hình đơn giản
nano .env.serving
```

**Nội dung file `.env.serving`:**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
MAX_BATCH_SIZE=32

# Model Configuration
# Sử dụng baseline model BGE-M3 (RECOMMENDED)
# Model sẽ được download tự động từ Hugging Face
MODEL_PATH=./models/bge-m3
```

**💡 Lý do sử dụng baseline model:**
- ✅ Performance tốt hơn fine-tuned model trong thử nghiệm thực tế
- ✅ Không cần training, tiết kiệm chi phí GPU
- ✅ Download nhanh từ Hugging Face (không cần Spaces)
- ✅ Model ổn định và được cộng đồng support tốt

### 7.5. Tạo thư mục cần thiết
```bash
# Tạo thư mục models và logs
mkdir -p models logs
```

---

## Bước 8: Download Baseline Model và Deploy API

### 8.1. Build Docker image
```bash
# Đảm bảo đang ở đúng thư mục
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Build image với all dependencies (bao gồm huggingface_hub)
docker build -f Dockerfile.cpu-serving -t legal-embedding-serving:latest .

# Verify image đã build thành công
docker images | grep legal-embedding-serving
```

**💡 Image này bao gồm:**
- Python 3.10
- PyTorch (CPU version)
- sentence-transformers
- transformers
- huggingface_hub (để download model)
- Flask (cho API)
- Script `download_model_from_spaces.py`
- Script `serve_model.py`

### 8.2. Download baseline model BGE-M3 từ Hugging Face

**🎯 Chiến lược mới:** Sử dụng baseline model `BAAI/bge-m3` thay vì fine-tuned model

```bash
# Download model bằng Docker container
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# Verify model đã download
ls -lah models/bge-m3/
```

**� Output mong đợi:**
```
🚀 Starting model download process...

======================================================================
🎯 DOWNLOADING BASELINE MODEL (RECOMMENDED)
======================================================================
📦 Model: BAAI/bge-m3
📁 Local directory: ./models/bge-m3
⬇️  Downloading from Hugging Face...
Fetching 15 files: 100%|██████████| 15/15 [02:30<00:00]
✅ Baseline model downloaded successfully!
📍 Model path: ./models/bge-m3

💡 Model này chưa được fine-tune, phù hợp cho serving
   vì baseline model có performance tốt hơn fine-tuned model.

======================================================================
✨ DOWNLOAD COMPLETED SUCCESSFULLY!
======================================================================
📂 Model location: ./models/bge-m3

📋 Next steps:
   1. Sử dụng model này cho serving với serve_model.py
   2. Model đã sẵn sàng để phục vụ requests
```

**⏰ Thời gian download:**
- Model size: ~2.3GB
- Download time: 2-5 phút (tùy network)

**🔍 Kiểm tra structure của model:**
```bash
# Xem các file đã download
tree -L 2 models/bge-m3/

# Expected structure:
# models/bge-m3/
# ├── config.json
# ├── config_sentence_transformers.json
# ├── model.safetensors
# ├── pytorch_model.bin
# ├── sentence_bert_config.json
# ├── tokenizer.json
# ├── tokenizer_config.json
# └── vocab.txt
```

### 8.3. Deploy Serving API với Docker Compose (Recommended)

**Method 1: Sử dụng Docker Compose (Đơn giản nhất)**
```bash
# Start service
docker-compose -f docker-compose.serving.yml up -d

# Check logs realtime
docker-compose -f docker-compose.serving.yml logs -f

# Check service status
docker-compose -f docker-compose.serving.yml ps
```

**📋 Output khi API start thành công:**
```
legal-embedding-api | 📥 Loading model from: ./models/bge-m3
legal-embedding-api | 💻 Using device: cpu
legal-embedding-api | ✅ Model loaded successfully!
legal-embedding-api | 📊 Embedding dimension: 1024
legal-embedding-api |  * Running on http://0.0.0.0:5000
```

### 8.4. Hoặc deploy bằng Docker run (Alternative)

**Method 2: Chạy trực tiếp với docker run**
```bash
# Run container serving API
docker run -d \
  --name legal-embedding-api \
  -p 5000:5000 \
  -v $(pwd)/models/bge-m3:/app/models/bge-m3 \
  -v $(pwd)/logs:/app/logs \
  -e MODEL_PATH=/app/models/bge-m3 \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=5000 \
  -e MAX_BATCH_SIZE=32 \
  --restart unless-stopped \
  legal-embedding-serving:latest

# Monitor logs
docker logs -f legal-embedding-api

# Check container status
docker ps | grep legal-embedding-api
```

**💡 Giải thích các options:**
- `-d`: Chạy container ở background
- `-p 5000:5000`: Map port 5000 ra ngoài
- `-v $(pwd)/models/bge-m3:/app/models/bge-m3`: Mount model directory
- `-e MODEL_PATH=/app/models/bge-m3`: Chỉ định path đến model
- `--restart unless-stopped`: Tự động restart khi droplet reboot

### 8.5. Verify API is running

**Test 1: Health check endpoint**
```bash
# Test từ trong droplet
curl http://localhost:5000/health
```

**Expected output:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "embedding_dim": 1024,
  "timestamp": 1730198400.123
}
```

**Test 2: Embedding endpoint**
```bash
# Test tạo embeddings
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Luật Dân sự năm 2015"]
  }'
```

**Expected output:**
```json
{
  "embeddings": [[0.123, -0.456, 0.789, ...]],  // 1024 dimensions
  "embedding_dim": 1024,
  "num_texts": 1,
  "inference_time": 0.089
}
```

**⏰ Thời gian khởi động API:**
- Model loading: 30-60 giây (lần đầu)
- API ready: 1-2 phút
- Requests tiếp theo: < 0.1 giây/câu

### 8.6. 🔥 Cấu hình Firewall (BẮT BUỘC!)

**⚠️ Bước này RẤT QUAN TRỌNG** - Nếu không làm thì API chỉ chạy local!

```bash
# Kiểm tra firewall status
ufw status

# QUAN TRỌNG: Allow SSH trước khi enable firewall (tránh bị lock out!)
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
22/tcp (v6)               ALLOW       Anywhere (v6)
5000/tcp (v6)             ALLOW       Anywhere (v6)
OpenSSH (v6)              ALLOW       Anywhere (v6)
```

### 8.7. 🌐 Test API từ bên ngoài internet

**Từ máy local của bạn (không phải trong droplet):**

```bash
# Thay YOUR_DROPLET_IP bằng IP thực của droplet
export DROPLET_IP="YOUR_DROPLET_IP"

# Test 1: Health check
curl http://$DROPLET_IP:5000/health

# Test 2: Generate embeddings
curl -X POST http://$DROPLET_IP:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Luật Dân sự năm 2015", "Bộ luật Hình sự năm 2017"]
  }'

# Test 3: Calculate similarity
curl -X POST http://$DROPLET_IP:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts1": ["Luật về quyền sở hữu tài sản"],
    "texts2": ["Tài sản riêng", "Tài sản chung", "Quyền kế thừa"]
  }'
```

**✅ Nếu thành công:**
- `/health` trả về `"status": "healthy"`
- `/embed` trả về array of embeddings (1024 dimensions)
- `/similarity` trả về ma trận similarity scores

### 8.8. 📊 Benchmark và Performance Testing

```bash
# Test performance với multiple requests
for i in {1..10}; do
  echo "Request $i:"
  time curl -X POST http://localhost:5000/embed \
    -H "Content-Type: application/json" \
    -d '{
      "texts": ["Test sentence for benchmarking performance"]
    }' -s -o /dev/null
done

# Expected inference time:
# - Single sentence: 50-100ms
# - Batch 10 sentences: 200-400ms
# - Batch 32 sentences: 500-1000ms
```

**💡 Performance tips:**
- Sử dụng batch requests khi có nhiều texts
- `MAX_BATCH_SIZE=32` là optimal cho 4GB RAM
- Upgrade lên 8GB RAM nếu cần xử lý batch lớn hơn

### 8.9. � Troubleshooting Common Issues

**Issue 1: API không start được**
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
```### 8.9. 🔧 Troubleshooting Common Issues

**Issue 1: API không start được**
```bash
# Check logs để xem lỗi gì
docker logs legal-embedding-api

# Common errors:
# 1. Model không tồn tại -> Download lại model
# 2. Out of memory -> Giảm MAX_BATCH_SIZE hoặc upgrade droplet
# 3. Port conflict -> Đổi API_PORT
```

**Solution cho Issue 1:**
```bash
# Download lại model nếu bị lỗi
docker run --rm \
  -v $(pwd)/models:/app/models \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# Restart API
docker restart legal-embedding-api
```

**Issue 2: Không connect được từ bên ngoài**
```bash
# Checklist:
# 1. Container có đang chạy?
docker ps | grep legal-embedding-api

# 2. Port mapping đúng chưa?
docker port legal-embedding-api

# 3. Firewall đã mở chưa?
ufw status | grep 5000

# 4. Test từ trong droplet trước
curl http://localhost:5000/health

# 5. Nếu local OK nhưng external fail -> Check firewall
ufw allow 5000/tcp
```

**Issue 3: Model download bị lỗi**
```bash
# Lỗi: Connection timeout, HTTP errors

# Solution 1: Check network
ping huggingface.co

# Solution 2: Retry download
docker run --rm \
  -v $(pwd)/models:/app/models \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# Solution 3: Download trực tiếp bằng git (nếu cần)
cd models
git lfs install
git clone https://huggingface.co/BAAI/bge-m3
mv bge-m3 bge-m3-temp && mv bge-m3-temp/* bge-m3/ && rm -rf bge-m3-temp
```

**Issue 4: Performance chậm**
```bash
# Check system resources
docker stats legal-embedding-api
htop

# Solutions:
# 1. Giảm batch size nếu out of memory
# 2. Upgrade droplet lên 8GB RAM
# 3. Optimize concurrent requests
```

**Issue 5: Container bị crash/restart liên tục**
```bash
# Check logs để tìm root cause
docker logs --tail 100 legal-embedding-api

# Common causes:
# 1. OOM (Out of Memory) -> Giảm MAX_BATCH_SIZE
# 2. Model file corrupted -> Download lại
# 3. Disk full -> Dọn dẹp: docker system prune -a

# Check disk usage
df -h
```

---

## Bước 9: Test và Monitor Serving API

### 9.1. Integration Testing với Python

**Tạo file test script trên máy local:**
```python
# test_embedding_api.py
import requests
import time

API_URL = "http://YOUR_DROPLET_IP:5000"  # Thay YOUR_DROPLET_IP

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["model_loaded"] == True
    print("✅ Health check passed!\n")

def test_embed_single():
    """Test embedding single text"""
    print("🔍 Testing /embed with single text...")
    response = requests.post(
        f"{API_URL}/embed",
        json={"texts": ["Luật Dân sự năm 2015"]}
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Embedding dim: {data['embedding_dim']}")
    print(f"Inference time: {data['inference_time']}s")
    assert response.status_code == 200
    assert len(data['embeddings']) == 1
    assert len(data['embeddings'][0]) == 1024
    print("✅ Single text embedding passed!\n")

def test_embed_batch():
    """Test embedding batch of texts"""
    print("🔍 Testing /embed with batch texts...")
    texts = [
        "Luật Dân sự năm 2015",
        "Bộ luật Hình sự năm 2017",
        "Luật Đất đai năm 2013"
    ]
    response = requests.post(
        f"{API_URL}/embed",
        json={"texts": texts}
    )
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Processed {data['num_texts']} texts")
    print(f"Inference time: {data['inference_time']}s")
    assert response.status_code == 200
    assert len(data['embeddings']) == 3
    print("✅ Batch embedding passed!\n")

def test_similarity():
    """Test similarity calculation"""
    print("🔍 Testing /similarity endpoint...")
    response = requests.post(
        f"{API_URL}/similarity",
        json={
            "texts1": ["Quyền sở hữu tài sản"],
            "texts2": ["Tài sản riêng", "Tài sản chung", "Quyền kế thừa"]
        }
    )
    data = response.json()
    print(f"Status: {response.status_code}")
    print(f"Similarities: {data['similarities']}")
    print(f"Inference time: {data['inference_time']}s")
    assert response.status_code == 200
    assert len(data['similarities']) == 1
    assert len(data['similarities'][0]) == 3
    print("✅ Similarity calculation passed!\n")

def benchmark_performance():
    """Benchmark API performance"""
    print("🔍 Benchmarking performance...")
    texts = ["Test sentence"] * 10
    
    times = []
    for i in range(5):
        start = time.time()
        response = requests.post(
            f"{API_URL}/embed",
            json={"texts": texts}
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Request {i+1}: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Average time for 10 texts: {avg_time:.3f}s")
    print(f"📊 Throughput: {10/avg_time:.1f} texts/second")
    print("✅ Benchmark completed!\n")

if __name__ == "__main__":
    try:
        test_health()
        test_embed_single()
        test_embed_batch()
        test_similarity()
        benchmark_performance()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
```

**Chạy test:**
```bash
# Trên máy local
python test_embedding_api.py
```

### 9.2. Monitor API với Docker

```bash
# Monitor logs realtime
docker logs -f legal-embedding-api

# Monitor system resources
docker stats legal-embedding-api

# Check container info
docker inspect legal-embedding-api
```

### 9.3. Setup monitoring script trong droplet

```bash
# Tạo monitoring script
cat > /root/monitor_api.sh << 'EOF'
#!/bin/bash
LOG_FILE="/root/api_monitor.log"

while true; do
    echo "=== $(date) ===" >> $LOG_FILE
    
    # Check API health
    health=$(curl -s http://localhost:5000/health)
    if echo "$health" | grep -q "healthy"; then
        echo "✅ API is healthy" >> $LOG_FILE
    else
        echo "❌ API is DOWN!" >> $LOG_FILE
        # Optional: Restart container
        # docker restart legal-embedding-api
    fi
    
    # Log system stats
    docker stats --no-stream legal-embedding-api >> $LOG_FILE
    
    echo "" >> $LOG_FILE
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x /root/monitor_api.sh

# Run monitor trong background
nohup /root/monitor_api.sh &

# Check monitor log
tail -f /root/api_monitor.log
```

---

## Bước 10: Best Practices và Maintenance

### 10.1. Xóa GPU Droplet sau khi training xong (Tiết kiệm chi phí)

```bash
# ⚠️ LƯU Ý: Chỉ xóa GPU droplet SAU KHI đã verify model hoạt động tốt

# Vào Digital Ocean Dashboard:
# 1. Chọn GPU Droplet
# 2. Click "Destroy"
# 3. Confirm deletion

# 💰 Lý do: GPU Droplet rất đắt ($72-144/month)
# Với chiến lược baseline model, không cần training nên không cần GPU!
```

### 10.2. Update Model khi cần thiết

**Scenario 1: Có baseline model mới hơn từ Hugging Face**
```bash
# SSH vào CPU droplet
ssh root@CPU_DROPLET_IP
cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Backup model cũ (optional)
mv models/bge-m3 models/bge-m3_backup_$(date +%Y%m%d)

# Download model mới
docker run --rm \
  -v $(pwd)/models:/app/models \
  legal-embedding-serving:latest \
  python download_model_from_spaces.py

# Restart API service
docker restart legal-embedding-api
# Hoặc với docker-compose:
# docker-compose -f docker-compose.serving.yml restart

# Verify
curl http://localhost:5000/health
```

**Scenario 2: Muốn thử model khác (ví dụ: bge-large)**
```python
# Edit file download_model_from_spaces.py
# Thay đổi MODEL_NAME và LOCAL_DIR trong hàm main():
MODEL_NAME = "BAAI/bge-large-en-v1.5"  # Hoặc model khác
LOCAL_DIR = "./models/bge-large"

# Sau đó download và update MODEL_PATH trong .env.serving
```

### 10.3. Auto-restart và High Availability

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
### 10.3. Auto-restart và High Availability

**Docker đã config auto-restart policy:**
```bash
# Với docker-compose.serving.yml
restart: unless-stopped

# Container sẽ tự động restart khi:
# - Droplet reboot
# - Container crash
# - Docker daemon restart
```

**Manual restart khi cần:**
```bash
# Restart container
docker restart legal-embedding-api

# Hoặc với docker-compose
docker-compose -f docker-compose.serving.yml restart

# Check status sau restart
docker ps | grep legal-embedding-api
curl http://localhost:5000/health
```

### 10.4. Security Best Practices

**✅ Firewall Configuration:**
```bash
# Chỉ allow từ backend server cụ thể (RECOMMENDED cho production)
ufw delete allow 5000/tcp
ufw allow from YOUR_BACKEND_SERVER_IP to any port 5000 proto tcp

# Hoặc allow từ một subnet
ufw allow from 10.0.0.0/16 to any port 5000 proto tcp

# Rate limiting để chống DDoS
ufw limit 5000/tcp comment 'Rate limit API requests'
```

**✅ SSH Security:**
```bash
# Disable password authentication
nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
systemctl restart sshd

# Chỉ allow SSH từ IP cụ thể
ufw delete allow 22/tcp
ufw allow from YOUR_IP to any port 22 proto tcp
```

**✅ Regular Updates:**
```bash
# Setup auto-update cho security patches
apt install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades

# Manual update định kỳ
apt update && apt upgrade -y
apt autoremove -y
```

### 10.5. Performance Optimization

**Monitor resource usage:**
```bash
# Real-time monitoring
htop
docker stats legal-embedding-api

# Check disk usage
df -h
du -sh /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup/models/

# Check memory
free -h
```

**Optimize based on workload:**
```bash
# Nếu RAM usage cao -> Giảm batch size
# Edit .env.serving:
MAX_BATCH_SIZE=16  # Thay vì 32

# Restart API
docker restart legal-embedding-api

# Nếu CPU usage cao -> Consider upgrade droplet
# $24/month (2 vCPUs) -> $48/month (4 vCPUs)
```

### 10.6. Backup Strategy

**Backup cấu hình quan trọng:**
```bash
# Tạo backup script
cat > /root/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/root/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configs
tar -czf $BACKUP_DIR/configs_$DATE.tar.gz \
  /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup/.env.serving \
  /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup/docker-compose.serving.yml

# Backup logs (last 7 days)
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz \
  /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup/logs/

# Keep only last 7 backups
cd $BACKUP_DIR && ls -t | tail -n +8 | xargs rm -f

echo "Backup completed: $DATE"
EOF

chmod +x /root/backup.sh

# Setup cron job (daily backup at 2 AM)
crontab -e
# Add line:
# 0 2 * * * /root/backup.sh >> /var/log/backup.log 2>&1
```

---

## Bước 11: Integration với Backend của bạn

### 11.1. Python Integration Example

```python
# embedding_client.py
import requests
from typing import List, Tuple
import numpy as np

class EmbeddingClient:
    def __init__(self, api_url: str):
        """
        Client để tương tác với Embedding API
        
        Args:
            api_url: URL của API (ví dụ: "http://YOUR_DROPLET_IP:5000")
        """
        self.api_url = api_url.rstrip('/')
        
    def health_check(self) -> bool:
        """Kiểm tra API có sống không"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.json().get("model_loaded", False)
        except:
            return False
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Tạo embeddings cho list of texts
        
        Args:
            texts: List of texts cần embedding
            
        Returns:
            numpy array shape (len(texts), 1024)
        """
        response = requests.post(
            f"{self.api_url}/embed",
            json={"texts": texts},
            timeout=30
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        return np.array(embeddings)
    
    def similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Tính similarity giữa 2 lists of texts
        
        Returns:
            numpy array shape (len(texts1), len(texts2))
        """
        response = requests.post(
            f"{self.api_url}/similarity",
            json={"texts1": texts1, "texts2": texts2},
            timeout=30
        )
        response.raise_for_status()
        similarities = response.json()["similarities"]
        return np.array(similarities)
    
    def find_most_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Tìm top-k candidates giống query nhất
        
        Returns:
            List of (index, text, score)
        """
        similarities = self.similarity([query], candidates)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (idx, candidates[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return results

# Usage example
if __name__ == "__main__":
    client = EmbeddingClient("http://YOUR_DROPLET_IP:5000")
    
    # Check health
    if not client.health_check():
        print("API is not available!")
        exit(1)
    
    # Example: RAG search
    query = "Quy định về quyền sở hữu đất đai"
    documents = [
        "Luật Đất đai 2013 quy định về quyền sử dụng đất",
        "Bộ luật Hình sự về tội xâm phạm tài sản",
        "Luật Dân sự về quyền sở hữu tài sản",
        "Luật Nhà ở về quyền sở hữu nhà",
    ]
    
    results = client.find_most_similar(query, documents, top_k=3)
    
    print(f"Query: {query}\n")
    print("Top 3 most similar documents:")
    for rank, (idx, text, score) in enumerate(results, 1):
        print(f"{rank}. [{score:.3f}] {text}")
```

### 11.2. Integrate vào RAG System

```python
# rag_with_embedding_api.py
from embedding_client import EmbeddingClient
import numpy as np
from typing import List, Dict

class RAGSystem:
    def __init__(self, embedding_api_url: str, corpus: List[Dict[str, str]]):
        """
        RAG System sử dụng external embedding API
        
        Args:
            embedding_api_url: URL của embedding API
            corpus: List of documents, mỗi document có 'id' và 'text'
        """
        self.client = EmbeddingClient(embedding_api_url)
        self.corpus = corpus
        
        # Pre-compute embeddings cho corpus
        print("Computing corpus embeddings...")
        corpus_texts = [doc['text'] for doc in corpus]
        self.corpus_embeddings = self.client.embed(corpus_texts)
        print(f"Embedded {len(corpus)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search relevant documents cho query
        
        Returns:
            List of documents với similarity scores
        """
        # Get query embedding
        query_embedding = self.client.embed([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.corpus_embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            {
                **self.corpus[idx],
                'score': float(similarities[idx])
            }
            for idx in top_indices
        ]
        
        return results

# Usage
if __name__ == "__main__":
    # Sample corpus
    corpus = [
        {"id": "doc1", "text": "Luật Đất đai 2013 quy định về quyền sử dụng đất"},
        {"id": "doc2", "text": "Bộ luật Hình sự về tội xâm phạm tài sản"},
        {"id": "doc3", "text": "Luật Dân sự về quyền sở hữu tài sản"},
    ]
    
    rag = RAGSystem("http://YOUR_DROPLET_IP:5000", corpus)
    
    # Search
    results = rag.search("quyền sở hữu đất đai", top_k=2)
    
    print("\nSearch Results:")
    for result in results:
        print(f"[{result['score']:.3f}] {result['id']}: {result['text']}")
```

---

## 📊 Tổng kết Chi phí và Performance

### Chi phí hàng tháng

| Service | Configuration | Cost/month | Note |
|---------|--------------|------------|------|
| **GPU Droplet** | V100, 8GB RAM | ~~$72~~ **$0** | ❌ Không cần! (dùng baseline model) |
| **CPU Droplet** | 4GB RAM, 2 vCPUs | $24 | ✅ Cho serving cơ bản |
| **CPU Droplet** | 8GB RAM, 4 vCPUs | $48 | ✅ RECOMMENDED |
| **Storage** | N/A | $0 | Model ~2.3GB, trong droplet |

**💰 Total: $24-48/month** (không cần GPU!)

### Performance Metrics (BGE-M3 on CPU)

| Metric | 4GB Droplet | 8GB Droplet |
|--------|-------------|-------------|
| Single text (avg) | 80-100ms | 60-80ms |
| Batch 10 texts | 300-400ms | 250-350ms |
| Batch 32 texts | 800-1000ms | 600-800ms |
| Max throughput | ~25 texts/sec | ~35 texts/sec |
| Embedding dim | 1024 | 1024 |

### So sánh Baseline vs Fine-tuned

| Aspect | Baseline BGE-M3 | Fine-tuned Model |
|--------|----------------|------------------|
| **Setup cost** | $0 (no training) | $72-144 (GPU training) |
| **Setup time** | 5 phút (download) | 2-4 giờ (training) |
| **Model size** | 2.3GB | 2.3GB |
| **Performance** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good |
| **Serving cost** | $24-48/month | $24-48/month |
| **Maintenance** | ✅ Easy | ⚠️ Complex |

**🎯 Kết luận: Baseline model là lựa chọn tốt nhất!**

---

## 🎉 Hoàn thành!

Bạn đã setup thành công hệ thống Embedding API với:
- ✅ **Baseline BGE-M3 model** - Performance tốt nhất
- ✅ **Không cần GPU** - Tiết kiệm $72-144/month
- ✅ **CPU Droplet serving 24/7** - Chỉ $24-48/month
- ✅ **Auto-restart** - High availability
- ✅ **Production-ready** - Security, monitoring, backup

**Next steps:**
- Integrate API vào backend của bạn
- Setup monitoring và alerting
- Consider load balancer nếu traffic cao
- Add authentication/API key nếu cần

**Support:**
- GitHub Issues: https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/issues
- Documentation: README.md trong repo

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

