# 🚀 Hướng dẫn sử dụng AWS EC2 cho Serving

## 🎯 Chiến lược

**Serving**: AWS EC2 Instance (chạy lâu dài) → Ổn định và hiệu quả chi phí

## 📋 Lựa chọn EC2 Instance phù hợp

### Instance types được khuyến nghị:

**1. Cho CPU-only serving (BGE-M3):**

- `t3.large` (2 vCPU, 8 GB RAM) - $0.0832/giờ - Phù hợp cho test/dev
- `t3.xlarge` (4 vCPU, 16 GB RAM) - $0.1664/giờ - Phù hợp cho production nhẹ
- `c5.xlarge` (4 vCPU, 8 GB RAM) - $0.17/giờ - Tối ưu compute

**2. Cho GPU serving (nếu cần GPU acceleration):**

- `g4dn.xlarge` (4 vCPU, 16 GB RAM, 1 GPU T4) - $0.526/giờ
- `g4dn.2xlarge` (8 vCPU, 32 GB RAM, 1 GPU T4) - $0.752/giờ

**💡 Khuyến nghị:** Sử dụng `t3.xlarge` cho serving BGE-M3 model

## Bước 1: Tạo EC2 Instance

### 1.1. Tạo EC2 Instance trên AWS Console

1. Vào AWS EC2 Console
2. Click "Launch Instance"
3. **AMI**: Ubuntu Server 22.04 LTS
4. **Instance Type**: `t3.xlarge` (4 vCPU, 16 GB RAM)
5. **Key Pair**: Tạo hoặc chọn key pair có sẵn
6. **Security Group**: Tạo security group mới với rules:
   - SSH (22) từ My IP
   - Custom TCP (5000-5001) từ Anywhere (0.0.0.0/0)
7. **Storage**: 30 GB gp3 (để lưu model ~4-5GB)
8. Launch instance

### 1.2. Kết nối EC2 Instance

```bash
# Thay YOUR_KEY_PAIR.pem và EC2_PUBLIC_IP
ssh -i "minh.pem" ubuntu@54.81.181.125

# Hoặc nếu sử dụng Windows với PuTTY:
# Dùng PuTTY với private key (.ppk) để connect
```

## Bước 2: Setup Environment trên EC2

### 2.1. Update system và install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker ubuntu

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Logout and login again to apply group changes
exit
```

### 2.2. Kết nối lại và verify Docker

```bash
# Kết nối lại EC2
ssh -i "YOUR_KEY_PAIR.pem" ubuntu@EC2_PUBLIC_IP

# Verify Docker installation
docker --version
docker compose version

# Test Docker
docker run hello-world
```

### 2.3. Clone repository

```bash
cd /home/ubuntu
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/embed_serving
```

### 2.4. Cấu hình environment cho serving

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

### 2.5. Tạo thư mục cần thiết

```bash
# Tạo thư mục models và logs
mkdir -p models logs
```

---

## Bước 3: Download Model và Deploy API

### 3.1. Build Docker image

```bash
# Đảm bảo đang ở đúng thư mục
cd /home/ubuntu/Vietnamese-Legal-Chatbot-RAG-System/embed_serving

# Build image với all dependencies (bao gồm huggingface_hub)
docker build -f Dockerfile.cpu-serving -t legal-embedding-serving:latest .

# Verify image đã build thành công
docker images | grep legal-embedding-serving
```

### 3.2. Download baseline model BGE-M3 từ Hugging Face

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

### 3.3. Deploy API với Docker

**Method: Chạy với docker run**

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

### 3.4. Verify API is running

**Test 1: Health check endpoint**

```bash
# Test từ trong EC2 instance
curl http://localhost:5001/health
```

**Test 2: Embedding endpoint**

```bash
# Test tạo embeddings
curl -X POST http://54.145.68.99:5001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Luật Dân sự năm 2015"]
  }'
```

### 3.5. 🔥 Cấu hình Security Group (AWS Firewall)

**Trên AWS Console:**

1. Vào EC2 → Security Groups
2. Chọn security group của instance
3. Edit Inbound Rules:
   - **SSH**: Port 22, Source: My IP
   - **API**: Port 5001, Source: 0.0.0.0/0 (hoặc specific IPs)
   - **Health Check**: Port 5001, Source: 0.0.0.0/0

**Hoặc dùng AWS CLI:**

```bash
# Get Security Group ID
aws ec2 describe-instances --instance-ids i-0cb01bbedb63c5a0d \
  --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId'

# Add rules (thay YOUR_SG_ID)
aws ec2 authorize-security-group-ingress \
  --group-id sg-05658de3373a66057 \
  --protocol tcp \
  --port 5001 \
  --cidr 0.0.0.0/0
```

### 3.6. 🌐 Test API từ bên ngoài internet

**Từ máy local của bạn:**

```bash
# Thay EC2_PUBLIC_IP bằng Public IP thực của EC2 instance
export EC2_IP="EC2_PUBLIC_IP"

# Test 1: Health check
curl http://54.145.68.99:5001/health

# Test 2: Generate embeddings
curl -X POST http://$EC2_IP:5001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Luật Dân sự năm 2015", "Bộ luật Hình sự năm 2017"]
  }'
```

---

## 🔧 Quản lý và Monitoring

### 4.1. Auto-start service khi EC2 reboot

```bash
# Tạo systemd service để tự động start container
sudo nano /etc/systemd/system/legal-embedding.service
```

**Nội dung file service:**

```ini
[Unit]
Description=Legal Embedding API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/Vietnamese-Legal-Chatbot-RAG-System/embed_serving
ExecStart=/usr/bin/docker start legal-embedding-api
ExecStop=/usr/bin/docker stop legal-embedding-api

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
sudo systemctl enable legal-embedding.service
sudo systemctl start legal-embedding.service

# Check status
sudo systemctl status legal-embedding.service
```

### 4.2. Setup CloudWatch Monitoring (Optional)

```bash
# Install CloudWatch Agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure CloudWatch (cần IAM role với CloudWatch permissions)
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-config-wizard
```

### 4.3. Cost Optimization Tips

**💰 Tiết kiệm chi phí:**

1. **Sử dụng Reserved Instances:** Giảm 30-60% chi phí cho long-term
2. **Spot Instances:** Giảm đến 90% chi phí (có thể bị interrupt)
3. **Scheduled Scaling:** Tự động stop instance vào ban đêm nếu không cần
4. **EBS Optimization:** Sử dụng gp3 thay vì gp2

**Script tự động stop/start:**

```bash
# Stop EC2 vào 11PM (UTC)
echo "0 23 * * * aws ec2 stop-instances --instance-ids YOUR_INSTANCE_ID" | crontab -

# Start EC2 vào 7AM (UTC)
echo "0 7 * * * aws ec2 start-instances --instance-ids YOUR_INSTANCE_ID" | crontab -
```
