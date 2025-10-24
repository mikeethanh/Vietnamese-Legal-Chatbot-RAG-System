# ✅ DigitalOcean GPU Training Checklist

## 🚀 **Pre-Training Checklist**

### **1. Droplet Setup** ⏱️ 5 minutes
- [ ] Created H100 GPU Droplet ($4.89/hour)
- [ ] Selected AI/ML Ready PyTorch 2.1 image  
- [ ] SSH access working: `ssh root@DROPLET_IP`
- [ ] GPU detected: `nvidia-smi` shows H100 80GB

### **2. Environment Setup** ⏱️ 10 minutes  
- [ ] Repository cloned: `git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git`
- [ ] Setup script executed: `./gpu_droplet_setup.sh`
- [ ] Dependencies installed: `pip list | grep torch`
- [ ] GPU access confirmed: `python3 -c "import torch; print(torch.cuda.is_available())"`

### **3. Configuration** ⏱️ 5 minutes
- [ ] Environment file created: `.env` exists
- [ ] Spaces credentials added: `SPACES_KEY` and `SPACES_SECRET` set
- [ ] Bucket name configured: `SPACES_BUCKET` set
- [ ] GPU batch size optimized: `GPU_BATCH_SIZE=2` for H100

### **4. Pre-flight Check** ⏱️ 2 minutes
- [ ] Disk space available: `df -h` shows >50GB free
- [ ] Memory available: `free -h` shows >100GB free  
- [ ] No other GPU processes: `nvidia-smi` shows 0% utilization
- [ ] Training script executable: `./start_training.sh --help`

---

## 🏃 **Training Execution** ⏱️ 2-3 hours

### **1. Start Training**
- [ ] Training started: `./start_training.sh`
- [ ] Process ID noted: PID saved to `.training_pid`
- [ ] Initial logs appear: `tail -f training_*.log`
- [ ] GPU utilization begins: `nvidia-smi` shows activity

### **2. First 10 Minutes Monitoring**
- [ ] Model loading successful: "✅ Model loaded successfully" in logs
- [ ] Memory usage stable: <40GB on H100
- [ ] Training started: "🚀 Starting training..." appears
- [ ] No CUDA errors: No "out of memory" messages

### **3. Ongoing Monitoring** 
- [ ] GPU utilization: 85-95% consistently
- [ ] Memory usage: <70GB on H100 
- [ ] Training speed: >50 samples/sec
- [ ] Loss decreasing: Check every 30 minutes

---

## 📊 **Success Indicators**

### **Training Progress** ✅
```bash
# These should appear in logs:
✅ Model loaded successfully
🚀 Starting training...
📊 Epoch 1/3: Loss decreasing
🔍 GPU memory stable at 25-35GB  
⚡ Training speed: 80-120 samples/sec
```

### **Performance Metrics** 📈
```bash
# Target values for H100:
GPU Utilization: 85-95%
Memory Usage: 20-50GB / 80GB
Training Speed: 80-150 samples/sec  
Time per Epoch: 30-45 minutes
```

---

## 🚨 **Red Flags (Stop Training If)**

### **Critical Issues** 🛑
- [ ] Memory usage > 70GB on H100 (memory leak)
- [ ] GPU utilization < 30% for >5 minutes (stuck)
- [ ] Training speed < 20 samples/sec (bottleneck)
- [ ] "CUDA out of memory" errors (config issue)
- [ ] No loss improvement for >1 hour (convergence issue)

### **Warning Signs** ⚠️
- [ ] Memory usage steadily increasing (potential leak)
- [ ] GPU temperature > 85°C (cooling issue)  
- [ ] Disk usage > 90% (space issue)
- [ ] Network errors during upload (connectivity issue)

---

## 🎯 **Completion Checklist**

### **Training Finished** ✅ 
- [ ] "✅ Training completed successfully!" in logs
- [ ] Model uploaded to Spaces successfully
- [ ] Final model size reasonable (~2-4GB)
- [ ] No error messages in final logs

### **Post-Training** 
- [ ] Model files verified in Spaces bucket
- [ ] Training logs archived: `training_logs_YYYYMMDD.tar.gz`
- [ ] GPU memory cleared: `torch.cuda.empty_cache()`
- [ ] Droplet shutdown scheduled: `shutdown -h +30`

### **Cleanup**
- [ ] Training processes stopped: `./stop_training.sh`
- [ ] Temporary files cleaned: `rm -rf /tmp/*`
- [ ] Final costs calculated: ~$15 for H100 3-hour training
- [ ] Droplet destroyed (to stop billing)

---

## 📞 **Emergency Contacts & Commands**

### **Quick Commands**
```bash
# Emergency stop
./stop_training.sh

# Emergency GPU reset  
nvidia-smi --gpu-reset

# Emergency reboot
sudo reboot

# Check costs (if doctl installed)
doctl account get
```

### **Support Resources**
- **DigitalOcean Support**: https://cloud.digitalocean.com/support
- **Model Issues**: Check training logs for error patterns
- **GPU Issues**: `nvidia-smi --help` for diagnostics
- **Billing Questions**: DigitalOcean billing dashboard

---

## 💰 **Cost Summary**

### **Typical Training Costs (H100)**
```
Droplet: $4.89/hour × 3 hours = $14.67
Storage: $0.02/GB × 5GB = $0.10
Bandwidth: Free (under 1TB)
Total: ~$15 USD
```

### **Cost Control**
- [ ] Auto-shutdown enabled: `at -l` shows scheduled shutdown
- [ ] Training time estimated: 2-3 hours max
- [ ] No unnecessary services running: `htop` shows clean processes
- [ ] Droplet will be destroyed after completion

---

**🎉 Use this checklist to ensure smooth training deployment on DigitalOcean GPU Droplets!**

**⏰ Total Setup Time: ~20 minutes**  
**🕐 Training Time: 2-3 hours**  
**💰 Total Cost: ~$15 USD**