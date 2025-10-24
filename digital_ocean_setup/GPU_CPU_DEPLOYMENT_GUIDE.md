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

## Bước 7: Setup CPU Droplet cho Serving

### 7.1. Kết nối CPU Droplet
```bash
ssh root@CPU_DROPLET_IP
```

### 7.2. Setup CPU Droplet
```bash
# Update system
apt update && apt upgrade -y

# Install dependencies
apt install -y git curl wget build-essential

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Clone repository
cd /root
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
```

### 7.3. Cấu hình environment
```bash
cp .env.template .env
nano .env
```

**Cập nhật .env cho CPU serving:**
```bash
# Digital Ocean Spaces Configuration
SPACES_ACCESS_KEY=your_spaces_access_key_here
SPACES_SECRET_KEY=your_spaces_secret_key_here
SPACES_ENDPOINT=https://sfo3.digitaloceanspaces.com
SPACES_BUCKET=legal-datalake

# CPU Serving specific
USE_GPU=false
# MODEL_PATH sẽ được lấy từ kết quả training GPU ở bước 6
# Ví dụ: models/embedding_model_gpu_20241024_143022
# Bạn sẽ lấy path này từ logs của GPU training hoặc check trong Spaces
MODEL_PATH=  # Để trống, sẽ cập nhật sau khi có kết quả training
PORT=5000
BATCH_SIZE=16  # Thấp hơn cho CPU
```

**📋 Cách lấy MODEL_PATH:**
1. **Từ GPU training logs:** Khi training xong, script sẽ in ra path như:
   ```
   🎉 Model uploaded successfully to: models/embedding_model_gpu_20241024_143022
   ```
2. **Từ Digital Ocean Spaces:** Vào Spaces dashboard → legal-datalake → models → copy tên folder mới nhất
3. **Từ deploy script:** Script `deploy.sh` có thể tự động detect latest model

---

## Bước 8: Transfer Model từ GPU sang CPU Droplet

### 8.1. Sau khi training xong trên GPU - Lấy MODEL_PATH

**Trên GPU Droplet:**
```bash
# Kiểm tra model đã upload lên Spaces
ls -la /tmp/model/

# Ghi lại model path từ training logs
tail -n 20 /tmp/logs/training.log | grep "Model uploaded successfully"
# Kết quả sẽ là: 🎉 Model uploaded successfully to: models/embedding_model_gpu_20241024_143022

# Hoặc check trực tiếp trên Spaces bằng AWS CLI
aws s3 ls s3://legal-datalake/models/ --endpoint-url=https://sgp1.digitaloceanspaces.com
```

**📝 Ghi lại MODEL_PATH:** `models/embedding_model_gpu_YYYYMMDD_HHMMSS`

**Ví dụ:** `models/embedding_model_gpu_20241024_143022`

### 8.2. Cập nhật MODEL_PATH và deploy trên CPU Droplet

**Trên CPU Droplet:**
```bash
# Cập nhật MODEL_PATH trong .env file với path từ bước 8.1
nano .env

# Thêm MODEL_PATH vào file .env:
# MODEL_PATH=models/embedding_model_gpu_20241024_143022  # Thay bằng path thật từ bước 8.1

# Hoặc dùng sed để cập nhật nhanh
sed -i 's|MODEL_PATH=.*|MODEL_PATH=models/embedding_model_gpu_20241024_143022|g' .env

# Verify cấu hình
grep MODEL_PATH .env
```

### 8.3. Deploy serving services
```bash
# Build serving image
docker-compose build embedding-server

# Deploy serving services  
./deploy.sh deploy

# Hoặc dùng script tự động download latest model
./deploy.sh download  # Tự động tìm model mới nhất
./deploy.sh deploy    # Deploy với model mới
```

---

## Bước 9: Verify và Test

### 9.1. Test CPU Serving Droplet
```bash
# Trên CPU Droplet
./deploy.sh health

# Test từ local
curl http://CPU_DROPLET_IP/health
```

### 9.2. Performance comparison
```bash
# Test embedding speed
python test_api.py http://CPU_DROPLET_IP:5000
```

---

## Bước 10: Cleanup GPU Droplet

### 10.1. Backup quan trọng
**Trên GPU Droplet:**
```bash
# Backup logs và configs
tar -czf training_backup.tar.gz /tmp/model /tmp/data /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup/.env

# Upload backup lên Spaces (optional)
aws s3 cp training_backup.tar.gz s3://legal-datalake/backups/ --endpoint-url=https://sgp1.digitaloceanspaces.com
```

### 10.2. Destroy GPU Droplet
1. **Vào Digital Ocean Dashboard**
2. **Droplets** → **legal-ai-gpu-training**
3. **Settings** → **Destroy**
4. **Type droplet name** → **Destroy**

💰 **Tiết kiệm**: Thay vì $72/month GPU liên tục → chỉ trả $1-2 cho vài giờ training

---

## 📊 So sánh Performance & Cost

### Training Performance
| Phương án | Thời gian | Chi phí/training | GPU Memory |
|-----------|-----------|------------------|------------|
| CPU Droplet | 60-90 phút | ~$0.50 | 0 GB |
| GPU Droplet | 15-30 phút | ~$1-2 | 16 GB |

### Serving Performance  
| Phương án | Response time | Chi phí/tháng | Throughput |
|-----------|---------------|---------------|------------|
| CPU Droplet | 200-500ms | $24-48 | 5-10 req/s |
| GPU Droplet | 50-100ms | $72+ | 20-50 req/s |

### Khuyến nghị tối ưu
- **Training**: GPU Droplet (destroy sau khi dùng)
- **Serving**: CPU Droplet (chạy lâu dài)
- **Re-training**: Tạo GPU Droplet mới khi cần

---

## 🔄 Workflow tối ưu

1. **Monthly/Quarterly**: Tạo GPU Droplet → Train model mới → Destroy
2. **Daily**: CPU Droplet serving 24/7
3. **Update**: Download model mới từ Spaces → Restart serving

**💡 Lợi ích**: 
- Tiết kiệm 70-80% chi phí
- Training nhanh hơn 3-4 lần
- Serving ổn định và rẻ