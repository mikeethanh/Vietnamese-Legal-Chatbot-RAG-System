# 🚀 Hướng dẫn sử dụng GPU Droplet cho Serving

## 🎯 Chiến lược

**Serving**: GPU Droplet (chạy lâu dài) → Ổn định và rẻ

## Bước 7: Setup CPU Droplet cho Serving

### 7.1. Kết nối CPU Droplet
```bash
ssh root@CPU_DROPLET_IP
```

### 7.1. Clone repository
```bash
cd /root
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/embed_serving
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
```

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
cd /root/Vietnamese-Legal-Chatbot-RAG-System/embed_serving

# Build image với all dependencies (bao gồm huggingface_hub)
docker build -f Dockerfile.cpu-serving -t legal-embedding-serving:latest .

# Verify image đã build thành công
docker images | grep legal-embedding-serving
```

### 8.2. Download baseline model BGE-M3 từ Hugging Face

```bash
# Download model bằng Docker container
docker run --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  legal-embedding-serving:latest \
  python scripts/download_model_from_spaces.py

# Verify model đã download
ls -lah models/bge-m3/
```

### 8.4. Hoặc deploy bằng Docker run (Alternative)

**Method 2: Chạy trực tiếp với docker run**
```bash
# Run container serving API
docker run -d \
  --name legal-embedding-api \
  -p 5001:5000 \
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

**Test 2: Embedding endpoint**
```bash
# Test tạo embeddings
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Luật Dân sự năm 2015"]
  }'
```

### 8.6. 🔥 Cấu hình Firewall
```bash
# Kiểm tra firewall status
ufw status

# QUAN TRỌNG: Allow SSH trước khi enable firewall (tránh bị lock out!)
ufw allow OpenSSH
ufw allow 22/tcp

# Allow API port
ufw allow 5000/tcp
ufw allow 5001/tcp
# Enable firewall
ufw --force enable

# Verify firewall rules
ufw status verbose
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
