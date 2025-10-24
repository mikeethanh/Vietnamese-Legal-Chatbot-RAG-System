# 🚨 GPU Droplet Quick Troubleshooting Guide

## ⚡ **Quick Diagnostics (30 seconds)**

```bash
# 1. GPU Status
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# 2. Training Process
ps aux | grep -E "python3|train"

# 3. Memory & Disk
free -h && df -h

# 4. Latest Logs
tail -20 training_*.log
```

---

## 🔥 **Common Issues & Instant Fixes**

### **Issue 1: CUDA Out of Memory**
```bash
# Symptoms: RuntimeError: CUDA out of memory
# Quick Fix:
python3 -c "import torch; torch.cuda.empty_cache()"
echo "GPU_BATCH_SIZE=1" >> .env
./stop_training.sh && ./start_training.sh
```

### **Issue 2: Training Not Starting**
```bash
# Symptoms: No GPU utilization, no logs
# Quick Fix:
chmod +x *.sh
./gpu_droplet_setup.sh
source .env
python3 train_embedding_gpu.py
```

### **Issue 3: Slow Training Speed**
```bash
# Symptoms: <10 samples/sec
# Quick Fix:
echo "DATALOADER_NUM_WORKERS=8" >> .env
echo "USE_FP16=true" >> .env
./stop_training.sh && ./start_training.sh
```

### **Issue 4: Connection Lost**
```bash
# Symptoms: SSH disconnected, training stopped
# Prevention:
screen -S training
./start_training.sh
# Ctrl+A, D to detach
# Later: screen -r training
```

### **Issue 5: Disk Full**
```bash
# Symptoms: No space left on device
# Quick Fix:
rm -rf ~/.cache/huggingface/
docker system prune -f
rm -f training_*.log.old
```

---

## 📊 **Performance Benchmarks**

### **H100 80GB Expected Performance:**
- **Memory Usage**: 20-30GB (max 50GB)
- **GPU Utilization**: 85-95%
- **Training Speed**: 80-150 samples/sec
- **Time per Epoch**: 30-45 minutes
- **Total Training**: 2-3 hours

### **Warning Signs:**
- ❌ Memory > 70GB = Memory leak
- ❌ GPU < 50% = CPU bottleneck  
- ❌ Speed < 30 samples/sec = I/O issue
- ❌ No progress > 10min = Hung process

---

## 🛠️ **Emergency Recovery**

### **Nuclear Option (Reset Everything):**
```bash
#!/bin/bash
# Save this as emergency_reset.sh

echo "🚨 Emergency Reset Starting..."

# Kill all training processes
pkill -f python3
pkill -f train

# Clear GPU memory
nvidia-smi --gpu-reset

# Clear system caches
sync && echo 3 > /proc/sys/vm/drop_caches

# Clear Python caches  
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Reset CUDA context
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print('✅ CUDA reset complete')
"

# Restart from clean state
cd Vietnamese-Legal-Chatbot-RAG-System/digital_ocean_setup
./gpu_droplet_setup.sh
./start_training.sh

echo "🎯 Emergency reset completed!"
```

---

## 📞 **When to Contact Support**

### **DigitalOcean Support Issues:**
- Droplet not starting
- GPU not detected
- Billing questions
- Network connectivity issues

### **Model/Code Issues:**
- Memory usage > 100GB on H200
- Training accuracy not improving
- Model convergence problems
- Data pipeline errors

---

## 💡 **Pro Tips**

### **Cost Optimization:**
```bash
# Set auto-shutdown (4 hours max)
echo "shutdown -h +240" | at now

# Use spot pricing (if available)
# Monitor: doctl account get

# Snapshot after successful setup
# Name: "bge-m3-ready-$(date +%Y%m%d)"
```

### **Performance Optimization:**
```bash
# For H100 (80GB VRAM):
export GPU_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export MAX_SEQ_LENGTH=256

# For H200 (141GB VRAM):
export GPU_BATCH_SIZE=8  
export GRADIENT_ACCUMULATION_STEPS=2
export MAX_SEQ_LENGTH=512
```

### **Monitoring Shortcuts:**
```bash
# One-liner status check
alias status='nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits && tail -5 training_*.log'

# Quick memory check  
alias memcheck='python3 -c "import torch; print(f\"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB\")"'

# Training progress
alias progress='grep -E "Epoch|Loss" training_*.log | tail -10'
```

---

**🎯 Keep this guide handy during training for quick problem resolution!**