# ✅ Checklist GPU + CPU Deployment

## 🎯 Tổng quan Strategy
- **GPU Droplet**: Training model (tạm thời, 2-4 giờ)
- **CPU Droplet**: Serving API (lâu dài, 24/7)
- **Tiết kiệm**: ~70% chi phí so với GPU serving

---

## Phase 1: Tạo GPU Droplet cho Training

### 1.1. Tạo GPU Droplet
- [ ] **Login Digital Ocean**: https://cloud.digitalocean.com/
- [ ] **Create → Droplets**
- [ ] **OS**: Ubuntu 22.04 (LTS) x64
- [ ] **Plan**: Premium Intel with GPU
- [ ] **Size**: $72/month - 8GB RAM, 1 vCPU, 1 GPU (V100)
- [ ] **Region**: Singapore (SGP1)
- [ ] **SSH**: Upload SSH key
- [ ] **Hostname**: `legal-ai-gpu-training`
- [ ] **Tags**: `gpu`, `training`, `temporary`
- [ ] **Create Droplet**
- [ ] **Ghi IP**: `________________`

### 1.2. Setup GPU Environment
```bash
# Kết nối
ssh root@GPU_DROPLET_IP

# Ghi lại commands này để check
```

- [ ] **Kết nối SSH thành công**
- [ ] **Check GPU**: `nvidia-smi` (có thể chưa có)
- [ ] **Update system**: `apt update && apt upgrade -y`
- [ ] **Install drivers**: `ubuntu-drivers autoinstall`
- [ ] **Reboot**: `reboot` (đợi 2-3 phút)
- [ ] **Reconnect và check GPU**: `nvidia-smi` ✅

**Kết quả nvidia-smi mong muốn:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.xx.xx    Driver Version: 470.xx.xx    CUDA Version: 11.4  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
+-------------------------------+----------------------+----------------------+
```

### 1.3. Setup Repository
- [ ] **Install Git**: `apt install -y git curl wget`
- [ ] **Clone repo**: 
```bash
cd /root
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
```
- [ ] **Copy env**: `cp .env.template .env`
- [ ] **Edit .env**: `nano .env`

**Cập nhật .env:**
- [ ] **SPACES_ACCESS_KEY**: `your_key_here`
- [ ] **SPACES_SECRET_KEY**: `your_secret_here`
- [ ] **SPACES_BUCKET**: `legal-datalake`
- [ ] **USE_GPU**: `true`
- [ ] **BATCH_SIZE**: `32` (cao hơn cho GPU)
- [ ] **EPOCHS**: `5`

### 1.4. Training trên GPU
- [ ] **Run setup**: `./gpu_cpu_deploy.sh gpu-setup`
- [ ] **Docker + NVIDIA OK**: Test `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
- [ ] **Start training**: `./gpu_cpu_deploy.sh gpu-train`

**⏰ Training time: 15-30 phút**

**Monitor training:**
```bash
# Terminal mới
ssh root@GPU_DROPLET_IP
./gpu_cpu_deploy.sh gpu-monitor
```

**Training completed checklist:**
- [ ] **Training finished without errors**
- [ ] **Model uploaded to Spaces**: Check logs
- [ ] **Model path noted**: `________________`
- [ ] **Backup created**: `./gpu_cpu_deploy.sh gpu-backup`

---

## Phase 2: Tạo CPU Droplet cho Serving

### 2.1. Tạo CPU Droplet  
- [ ] **Create → Droplets** (lần 2)
- [ ] **OS**: Ubuntu 22.04 (LTS) x64
- [ ] **Plan**: Basic - Regular with SSD
- [ ] **Size**: $24/month - 4GB RAM, 2 vCPUs (hoặc $48/month - 8GB RAM, 4 vCPUs)
- [ ] **Region**: Singapore (SGP1) - cùng region
- [ ] **SSH**: Dùng cùng SSH key
- [ ] **Hostname**: `legal-ai-cpu-serving`
- [ ] **Tags**: `cpu`, `serving`, `production`
- [ ] **Create Droplet**
- [ ] **Ghi IP**: `________________`

### 2.2. Setup CPU Environment
```bash
# Kết nối CPU Droplet
ssh root@CPU_DROPLET_IP
```

- [ ] **Kết nối SSH thành công**
- [ ] **Update system**: `apt update && apt upgrade -y`
- [ ] **Install tools**: `apt install -y git curl wget build-essential`
- [ ] **Install Docker**: `curl -fsSL https://get.docker.com | sh`
- [ ] **Install Docker Compose**: 
```bash
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```
- [ ] **Check Docker**: `docker --version && docker-compose --version`

### 2.3. Setup Repository trên CPU
- [ ] **Clone repo**:
```bash
cd /root
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
```
- [ ] **Copy env**: `cp .env.template .env`
- [ ] **Edit .env**: `nano .env`

**Cập nhật .env cho CPU:**
- [ ] **SPACES_ACCESS_KEY**: Same as GPU
- [ ] **SPACES_SECRET_KEY**: Same as GPU  
- [ ] **SPACES_BUCKET**: `legal-datalake`
- [ ] **MODEL_PATH**: `models/embedding_model_gpu_YYYYMMDD_HHMMSS` (từ GPU training)
- [ ] **USE_GPU**: `false`
- [ ] **BATCH_SIZE**: `16` (thấp hơn cho CPU)
- [ ] **PORT**: `5000`

---

## Phase 3: Deploy Serving trên CPU

### 3.1. Download Model từ GPU
- [ ] **Check model path từ GPU training logs**
- [ ] **Download model**: `./gpu_cpu_deploy.sh download-model MODEL_PATH`
- [ ] **Verify download**: `ls -la models/`

### 3.2. Deploy Services
- [ ] **Setup CPU**: `./gpu_cpu_deploy.sh cpu-setup`
- [ ] **Deploy services**: `./gpu_cpu_deploy.sh cpu-deploy`
- [ ] **Wait for startup**: ~30 seconds
- [ ] **Check health**: `./gpu_cpu_deploy.sh cpu-health`

**Health check expected:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

### 3.3. Test API
**Từ CPU Droplet:**
- [ ] **Health**: `curl http://localhost:5000/health`
- [ ] **Embed test**: 
```bash
curl -X POST http://localhost:5000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Luật Dân sự năm 2015"]}'
```

**Từ máy local:**
- [ ] **Health**: `curl http://CPU_DROPLET_IP/health`
- [ ] **Performance test**: `python test_api.py http://CPU_DROPLET_IP:5000`

---

## Phase 4: Cleanup và Optimization

### 4.1. Cleanup GPU Droplet
**⚠️ Chỉ làm sau khi confirm CPU serving hoạt động tốt!**

- [ ] **Verify CPU serving OK**: Test API hoạt động
- [ ] **Backup GPU artifacts**: `./gpu_cpu_deploy.sh gpu-backup`
- [ ] **Upload backup to Spaces**: Confirm backup uploaded
- [ ] **Destroy GPU Droplet**: 
  1. Digital Ocean Dashboard
  2. Droplets → legal-ai-gpu-training
  3. Settings → Destroy
  4. Type droplet name → Destroy

### 4.2. Production Optimization
**Trên CPU Droplet:**
- [ ] **Setup firewall**:
```bash
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS (if SSL)
ufw enable
```

- [ ] **Setup monitoring**: `./gpu_cpu_deploy.sh cpu-monitor`
- [ ] **Setup backup cron**:
```bash
crontab -e
# Add: 0 2 * * * cd /root/Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup && ./deploy.sh backup
```

---

## 🏁 Final Verification

### Performance Checklist
- [ ] **API Response time**: < 500ms
- [ ] **Throughput**: > 5 requests/second
- [ ] **Memory usage**: < 80%
- [ ] **CPU usage**: < 70%
- [ ] **Uptime**: 99.9%

### Cost Verification
- [ ] **GPU Droplet destroyed**: $0/month
- [ ] **CPU Droplet running**: $24-48/month
- [ ] **Training cost**: ~$2-4 per training session
- [ ] **Total savings**: ~70% vs GPU serving

### Integration Checklist
- [ ] **Copy embedding_adapter.py to backend**
- [ ] **Update backend .env**:
```bash
EMBEDDING_API_URL=http://CPU_DROPLET_IP:5000
LOCAL_EMBEDDING_MODEL_PATH=/backup/path
```
- [ ] **Test backend integration**
- [ ] **Performance acceptable in backend**

---

## 📞 Emergency Procedures

### Serving Down
```bash
# Quick restart
ssh root@CPU_DROPLET_IP
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
./gpu_cpu_deploy.sh cpu-deploy
```

### Re-training Needed
```bash
# Create new GPU Droplet → Follow Phase 1 again
# Training takes ~30 minutes
# Update CPU Droplet with new model
```

### Rollback
```bash
# Use previous model from Spaces
./gpu_cpu_deploy.sh download-model PREVIOUS_MODEL_PATH
./gpu_cpu_deploy.sh cpu-deploy
```

---

## 📊 Success Metrics

✅ **Training**: 15-30 phút thay vì 60-90 phút
✅ **Cost**: $24-48/tháng thay vì $72+/tháng  
✅ **Uptime**: 99.9% serving availability
✅ **Performance**: <500ms response time
✅ **Scalability**: Dễ dàng re-train và update

**🎯 Status: ___/_____ completed**