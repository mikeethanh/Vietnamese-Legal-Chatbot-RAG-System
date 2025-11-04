# Vietnamese Legal LLM - Finetune & Serving System

Hệ thống hoàn chỉnh để finetune và serving model Llama-3.1-8B cho tư vấn pháp luật Việt Nam trên GPU droplet Digital Ocean.

## 🎯 Mục tiêu

- **Model**: Finetune Llama-3.1-8B-Instruct với dữ liệu pháp luật Việt Nam (~100k examples)
- **Kỹ thuật**: LoRA (Low-Rank Adaptation) với Unsloth để tối ưu hóa
- **Deployment**: Serving trên Digital Ocean GPU droplet H200
- **API**: Compatible với OpenAI API format

## 📁 Cấu trúc thư mục

```
llm_finetuning_serving/
├── data_processing/           # Xử lý dữ liệu
│   ├── analyze_data.py       # Phân tích cấu trúc dữ liệu
│   ├── process_llama_data.py # Chuyển đổi sang format Llama
│   └── split_data.py         # Chia train/val/test với batching
├── finetune/                 # Training với Unsloth LoRA
│   └── train_llama.py        # Script training chính
├── evaluation/               # Đánh giá model
│   └── evaluate_model.py     # ROUGE, BLEU, Perplexity, LLM-eval
├── serving/                  # FastAPI serving system
│   └── serve_model.py        # API server với streaming
├── docker/                   # Docker deployment
│   ├── Dockerfile           # CUDA + Python environment
│   └── docker-compose.yml   # Production deployment
├── requirements.txt          # Python dependencies
├── .env.template            # Environment variables template
└── run_pipeline.sh          # Automation script
```

## 🚀 Quick Start

### Workflow tổng quan:
1. **Local**: Xử lý dữ liệu → Upload lên Digital Ocean Spaces
2. **Training GPU Droplet**: Download dữ liệu → Train → Upload model
3. **Serving GPU Droplet**: Download model → Serve API

### 1. Chuẩn bị dữ liệu (Local)

```bash
# Clone và di chuyển vào thư mục
cd llm_finetuning_serving

# Setup environment và dependencies
./run_pipeline.sh setup

# Copy và chỉnh sửa environment variables
cp .env.template .env
# Chỉnh sửa .env với Digital Ocean Spaces credentials

# Chuẩn bị và upload dữ liệu
./prepare_data.sh
```

### 2. Training trên GPU Droplet

```bash
# Trên Digital Ocean GPU droplet (H200)
git clone <repo>
cd llm_finetuning_serving

# Setup environment
./run_pipeline.sh setup
cp .env.template .env  # Edit với credentials

# Download dữ liệu và train
./run_pipeline.sh train
# Model sẽ tự động được upload lên Spaces sau khi train xong
```

### 3. Serving trên GPU Droplet khác

```bash
# Trên Digital Ocean GPU droplet khác
git clone <repo>
cd llm_finetuning_serving

# Setup environment
./run_pipeline.sh setup
cp .env.template .env  # Edit với credentials và MODEL_NAME

# Download model và serve
./run_pipeline.sh serve
# API sẽ chạy tại http://your-droplet-ip:8000
```

## 📊 Data & Model Management

**Dữ liệu được lưu trên Digital Ocean Spaces:**
- Bucket: `legal-datalake`
- Raw data: `process_data/finetune_data/`
- Processed data: `process_data/processed/`
- Models: `models/`

**Workflow:**
1. **Local**: Xử lý dữ liệu từ JSONL → Llama format → Upload lên Spaces
2. **Training Droplet**: Auto download dữ liệu → Train → Upload model lên Spaces  
3. **Serving Droplet**: Auto download model → Serve API

## 📊 Xử lý dữ liệu

### Format đầu vào (JSONL)
```json
{
  "instruction": "Trả lời câu hỏi pháp luật sau:",
  "input": "Trong Bộ luật Hình sự thì bao nhiêu tuổi được xem là người già?",
  "output": "Người cao tuổi được quy định tại Điều 2 Luật Người cao tuổi 2009..."
}
```

### Format Llama-3.1 Chat (sau khi xử lý)
```
<|start_header_id|>system<|end_header_id|>

Bạn là một chuyên gia tư vấn pháp luật Việt Nam...<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Hãy trả lời chi tiết về quy định pháp lý:
Trong Bộ luật Hình sự thì bao nhiêu tuổi được xem là người già?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Người cao tuổi được quy định tại Điều 2 Luật Người cao tuổi 2009...<|eot_id|>
```

### Cải tiến đã thực hiện

1. **Thêm EOS tokens**: `<|eot_id|>` cho Llama-3.1
2. **Cải thiện instructions**: Phân loại và làm rõ câu hỏi
3. **System prompt**: Chuyên gia pháp luật Việt Nam
4. **Batching strategy**: Padding theo độ dài sequence
5. **Stratified split**: Chia dữ liệu cân bằng theo độ dài

## 🔧 Finetune Configuration

### LoRA Parameters (tối ưu cho 8B model)
```python
lora_r=16              # Rank
lora_alpha=32          # 2 * lora_r 
lora_dropout=0.05      # Dropout
target_modules=[       # Target attention layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Training Hyperparameters
```python
learning_rate=2e-4
num_epochs=3
batch_size=2
gradient_accumulation_steps=4  # Effective batch size = 8
warmup_steps=10
max_seq_length=2048
```

### GPU Memory Optimization
- **4-bit quantization**: Load_in_4bit=True
- **Gradient checkpointing**: Unsloth optimization
- **Mixed precision**: BF16 on supported hardware

## 📈 Evaluation Metrics

### Automatic Metrics
- **ROUGE**: ROUGE-1, ROUGE-2, ROUGE-L
- **BLEU**: Sentence-level BLEU với smoothing
- **Perplexity**: Model confidence measure

### LLM-based Evaluation (GPT-4)
```json
{
  "accuracy": 8.5,      # Độ chính xác (0-10)
  "completeness": 7.8,  # Độ đầy đủ (0-10)
  "clarity": 9.2,       # Độ rõ ràng (0-10)
  "practicality": 8.0,  # Tính thực tiễn (0-10)
  "overall": 8.4        # Điểm tổng thể (0-10)
}
```

## 🌐 API Serving

### Endpoints

#### Chat Completions (OpenAI Compatible)
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Quy định về thời hiệu khởi kiện là gì?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

#### Streaming Response
```bash
POST /v1/chat/completions/stream
```

#### Health Check
```bash
GET /health
```

### Response Format
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1694268190,
  "model": "vietnamese-legal-llama",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Thời hiệu khởi kiện được quy định..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 150,
    "total_tokens": 170
  }
}
```

## 🐋 Docker Deployment

### Local Development
```bash
# Build image
./run_pipeline.sh build-docker

# Deploy với Docker Compose
./run_pipeline.sh deploy

# Check logs
docker-compose -f docker/docker-compose.yml logs -f
```

### Production on Digital Ocean

1. **Setup GPU Droplet H200**
```bash
# SSH vào droplet
ssh root@your-droplet-ip

# Install Docker + NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && apt-get install -y nvidia-docker2
systemctl restart docker
```

2. **Deploy Application**
```bash
# Clone repository
git clone <your-repo>
cd llm_finetuning_serving

# Setup environment
cp .env.template .env
# Edit .env với production values

# Deploy
docker-compose -f docker/docker-compose.yml up -d
```

3. **Monitor & Scale**
```bash
# Check GPU utilization
nvidia-smi

# Monitor containers
docker stats

# Scale replicas (nếu có multiple GPUs)
docker-compose -f docker/docker-compose.yml up -d --scale vietnamese-legal-llm=2
```

## 🔑 Environment Variables

### Required
```bash
MODEL_PATH=/app/model          # Path to finetuned model
CUDA_VISIBLE_DEVICES=0         # GPU device ID
```

### Optional (for full features)
```bash
# Training monitoring
WANDB_API_KEY=your_key

# LLM evaluation
OPENAI_API_KEY=your_key

# Model downloads
HF_TOKEN=your_token

# Data storage
DO_SPACES_KEY=your_key
DO_SPACES_SECRET=your_secret
```

## 📋 Performance Benchmarks

### Training Time (H200 GPU)
- **Data processing**: ~10 minutes (100k examples)
- **Training**: ~4-6 hours (3 epochs)
- **Evaluation**: ~30 minutes

### Inference Performance
- **Latency**: ~200-500ms per response
- **Throughput**: ~20-50 requests/second
- **Memory**: ~12-16GB VRAM (4-bit quantization)

### Model Quality
- **ROUGE-L**: ~0.45-0.55
- **BLEU**: ~0.25-0.35
- **LLM Eval**: ~7.5-8.5/10 overall

## 🛠️ Development

### Custom Data Format
```python
# data_processing/custom_processor.py
class CustomDataProcessor:
    def process_custom_format(self, data):
        # Implement your custom processing
        pass
```

### Custom Evaluation Metrics
```python
# evaluation/custom_metrics.py
def compute_legal_accuracy(predictions, references):
    # Implement domain-specific metrics
    pass
```

### API Extensions
```python
# serving/extensions.py
@app.post("/v1/legal/analyze")
async def analyze_legal_document(document: str):
    # Add specialized endpoints
    pass
```

## 🔍 Troubleshooting

### Common Issues

1. **GPU Memory Error**
```bash
# Reduce batch size in config
per_device_train_batch_size=1
gradient_accumulation_steps=8
```

2. **CUDA Out of Memory**
```bash
# Use smaller model or more quantization
load_in_8bit=True
max_seq_length=1024
```

3. **Slow Training**
```bash
# Enable optimizations
use_flash_attention=True
dataloader_num_workers=4
```

### Debug Commands
```bash
# Check GPU
nvidia-smi

# Monitor training
tail -f finetune/outputs/logs/training.log

# Test API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Test"}]}'
```

## 📚 Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Llama-3.1 Model Card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [Digital Ocean GPU Droplets](https://www.digitalocean.com/products/gpu-droplets)
- [Vietnamese Legal Dataset](https://huggingface.co/datasets/your-legal-dataset)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file

---

## 📞 Support

Nếu có vấn đề gì, vui lòng tạo issue hoặc liên hệ:
- GitHub Issues: [Create Issue](https://github.com/your-repo/issues)
- Email: your-email@domain.com

**Chúc bạn thành công với việc finetune Vietnamese Legal LLM! 🚀**