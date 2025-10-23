# 🚀 Vietnamese Legal AI - Digital Ocean Deployment

## 📋 Tổng quan

Hệ thống training và serving embedding model cho Vietnamese Legal documents sử dụng:
- **GPU Droplet**: Training nhanh và tối ưu (15-30 phút)
- **CPU Droplet**: Serving API 24/7 với chi phí thấp
- **Tiết kiệm**: ~70% chi phí so với GPU serving liên tục

## 📂 Cấu trúc Files

```
digital_ocean_setup/
├── 📚 Documentation
│   ├── GPU_CPU_DEPLOYMENT_GUIDE.md      # Hướng dẫn chi tiết
│   └── GPU_CPU_CHECKLIST.md             # Checklist từng bước
│
├── 🐳 Docker & Deployment  
│   ├── Dockerfile.gpu-training           # GPU training image
│   ├── Dockerfile.serving               # CPU serving image
│   ├── docker-compose.yml               # CPU serving orchestration
│   └── nginx.conf                       # Reverse proxy config
│
├── 🧠 AI & Training
│   ├── train_embedding_gpu.py           # GPU-optimized training
│   ├── embedding_server.py              # API server
│   ├── embedding_adapter.py             # Backend integration
│   └── requirements_gpu.txt             # GPU dependencies
│
├── 🛠️ Scripts & Tools
│   ├── gpu_cpu_deploy.sh               # Main deployment script
│   ├── test_api.py                     # API testing
│   └── .env.template                   # Environment template
│
└── 📊 Runtime (created during use)
    ├── models/                         # Downloaded models
    ├── data/                          # Training data cache  
    └── logs/                          # Training logs
```

## 🚀 Quick Start

### 1. Tạo GPU Droplet (Training)
```bash
# Tạo GPU Droplet $72/month trên Digital Ocean UI
# Ubuntu 22.04, GPU V100, Singapore region

ssh root@GPU_DROPLET_IP
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Configure
cp .env.template .env
nano .env  # Update Spaces credentials

# Train (15-30 phút)
./gpu_cpu_deploy.sh auto-train

# Destroy GPU Droplet sau khi train xong
```

### 2. Tạo CPU Droplet (Serving)  
```bash
# Tạo CPU Droplet $24-48/month trên Digital Ocean UI
# Ubuntu 22.04, Regular CPU, Singapore region

ssh root@CPU_DROPLET_IP  
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup

# Configure
cp .env.template .env
nano .env  # Update Spaces credentials + model path

# Deploy
./gpu_cpu_deploy.sh auto-deploy

# API available at http://CPU_DROPLET_IP:5000
```

## 📖 Chi tiết Documentation

- **[GPU_CPU_DEPLOYMENT_GUIDE.md](./GPU_CPU_DEPLOYMENT_GUIDE.md)**: Hướng dẫn từng bước đầy đủ
- **[GPU_CPU_CHECKLIST.md](./GPU_CPU_CHECKLIST.md)**: Checklist để không bỏ sót bước nào

## 🛠️ Commands chính

```bash
# GPU Droplet (Training)
./gpu_cpu_deploy.sh gpu-setup      # Setup GPU environment  
./gpu_cpu_deploy.sh auto-train     # Auto training workflow
./gpu_cpu_deploy.sh gpu-monitor    # Monitor training
./gpu_cpu_deploy.sh gpu-backup     # Backup artifacts

# CPU Droplet (Serving)
./gpu_cpu_deploy.sh download-model MODEL_PATH  # Download model
./gpu_cpu_deploy.sh auto-deploy    # Auto serving deployment  
./gpu_cpu_deploy.sh cpu-monitor    # Monitor serving
./gpu_cpu_deploy.sh cpu-health     # Health check

# General
./gpu_cpu_deploy.sh estimate-cost  # Cost estimation
./gpu_cpu_deploy.sh help          # Show all commands
```

## 📊 Performance & Cost

| Metric | GPU Training | CPU Serving | 
|--------|-------------|-------------|
| **Time** | 15-30 phút | 24/7 |
| **Cost** | $2-4/session | $24-48/month |  
| **Performance** | 3-4x faster | 200-500ms response |
| **Memory** | 16GB GPU | 4-8GB RAM |

## 🔌 Backend Integration

```python
# Copy embedding_adapter.py to backend/src/
from embedding_adapter import encode_texts

# Thay thế
# embeddings = model.encode(texts)  
embeddings = encode_texts(texts)  # Auto fallback CPU/API
```

## 🆘 Troubleshooting

```bash
# Serving issues
./gpu_cpu_deploy.sh cpu-health     # Check health
./gpu_cpu_deploy.sh cpu-deploy     # Restart serving

# Training issues  
./gpu_cpu_deploy.sh gpu-monitor    # Check GPU status
nvidia-smi                         # GPU utilization

# General issues
docker ps                          # Check containers
docker logs CONTAINER_ID           # Check logs
```

## 💡 Best Practices

1. **Training**: Tạo GPU Droplet → Train → Destroy → Tiết kiệm 70% cost
2. **Serving**: CPU Droplet chạy 24/7 với cost tối ưu
3. **Re-training**: Tạo GPU mới khi cần update model
4. **Backup**: Auto backup model và config lên Spaces
5. **Monitoring**: Setup health checks và alerting

## 📞 Support

- **Issues**: Check logs với `docker logs` và `./gpu_cpu_deploy.sh monitor`  
- **Performance**: Monitor resources với `./gpu_cpu_deploy.sh cpu-monitor`
- **Costs**: Estimate với `./gpu_cpu_deploy.sh estimate-cost`

**Happy deploying! 🎉**