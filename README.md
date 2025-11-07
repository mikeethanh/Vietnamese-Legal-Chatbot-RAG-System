# Vietnamese Legal Chatbot RAG System 🏛️

A comprehensive Vietnamese legal consultation chatbot system built with RAG (Retrieval-Augmented Generation) technology, modern microservices architecture, and advanced AI technologies.

## 🎯 Overview

This system provides intelligent legal consultation services for Vietnamese users by combining:

- **Large Language Models (LLM)** for natural language understanding and generation
- **Vector Database** for semantic search in legal corpus
- **Legal Document Processing Pipeline** for data preparation
- **Web Chat Interface** for user interaction
- **Scalable Backend Architecture** with microservices

## ✨ Key Features

### 🤖 AI Capabilities
- **Vietnamese Legal Consultation**: Answer questions based on Vietnamese legal documents
- **Semantic Search**: Smart search through 1.9M+ legal document corpus
- **Context-Aware Responses**: Provide accurate answers with source references
- **Real-time Chat Interface**: User-friendly web interface for legal consultation
- **Multi-format Support**: Process PDF, CSV, JSON, JSONL formats

### 🔧 Technical Features
- **RAG Architecture**: Combines retrieval and generation for accurate responses
- **Vector Search**: ChromaDB for efficient semantic search
- **LLM Integration**: Support multiple models (OpenAI GPT, LLaMA, Vietnamese models)
- **Scalable Processing**: Apache Spark for large-scale document processing
- **Microservices**: Dockerized components with Redis queue management
- **Data Pipeline**: Automated ETL for legal document collection and processing

## 🏗️ System Architecture

![System Architecture](asset%20/architecture_template.drawio.svg)

## 📁 Project Structure

```
Vietnamese-Legal-Chatbot-RAG-System/
│
├── 🖥️ backend/                    # Backend API service (FastAPI)
│   ├── src/
│   │   ├── app.py                # Main FastAPI application
│   │   ├── agent.py              # AI agent with tool calling
│   │   ├── brain.py              # LLM integration and chat logic
│   │   ├── cache.py              # Redis caching utilities
│   │   ├── configs.py            # Configuration management
│   │   ├── custom_embedding.py   # Custom embedding models
│   │   ├── database.py           # Database connections
│   │   ├── import_data.py        # Data import utilities
│   │   ├── legal_tools.py        # Legal-specific tools
│   │   ├── models.py             # Pydantic data models
│   │   ├── query_rewriter.py     # Query optimization
│   │   ├── rerank.py             # Result re-ranking
│   │   ├── search.py             # Search functionality
│   │   ├── splitter.py           # Document text splitting
│   │   ├── summarizer.py         # Text summarization
│   │   ├── tasks.py              # Celery background tasks
│   │   ├── tavily_tool.py        # Web search integration
│   │   ├── utils.py              # Utility functions
│   │   └── vectorize.py          # Vector database operations
│   ├── data/
│   │   ├── train.jsonl          # Raw training data
│   │   └── train_qa_format.jsonl # Formatted training data
│   ├── docker-compose.yml       # Backend services
│   ├── Dockerfile               # Container configuration
│   ├── entrypoint.sh           # Container startup script
│   ├── import_data.sh          # Data import script
│   ├── migration_title_to_question.sql # Database migration
│   ├── requirements.txt          # Python dependencies
│   ├── 📚 AGENT_TOOLS_GUIDE.md  # Agent tools documentation
│   ├── 📚 MIGRATION_GUIDE.md    # Migration guide
│   └── 📚 README.md            # Backend documentation
│
├── 🌐 frontend/                   # Web interface (Streamlit)
│   ├── chat_interface.py        # Main chat application
│   ├── config.toml             # Streamlit configuration
│   ├── docker-compose.yml      # Frontend services
│   ├── Dockerfile              # Container configuration
│   ├── entrypoint.sh           # Container startup script
│   └── requirements.txt        # Python dependencies
│
├── 🔄 data_pipeline/             # Data processing pipeline
│   ├── utils/                  # Processing utilities
│   │   ├── download_embed_data.ipynb      # Download legal corpus
│   │   ├── merge_instruction_data.py      # Merge instruction datasets
│   │   ├── process_finetune_data.ipynb    # Process training data
│   │   ├── process_finetune_data_2.ipynb  # ViLQA dataset processing
│   │   └── process_finetune_data_3.ipynb  # Extended dataset processing
│   ├── data/                   # Raw and processed data
│   │   ├── embed/              # Embedding data
│   │   ├── finetune_data/      # Fine-tuning datasets
│   │   ├── finetune_data2/     # ViLQA dataset
│   │   ├── finetune_data3/     # Extended fine-tuning data
│   │   ├── finetune_llm/       # LLM fine-tuning data
│   │   └── rag/                # RAG-specific data
│   ├── requirements.txt        # Python dependencies
│   └── 📚 README.md           # Pipeline documentation
│
├── 🤖 llm_finetuning_serving/    # LLM fine-tuning and serving
│   ├── data_processing/        # Data processing for LLM
│   │   ├── splits/             # Data splits
│   │   ├── analyze_data.py     # Data analysis
│   │   ├── data_analysis.json  # Analysis results
│   │   ├── download_data.py    # Download datasets
│   │   ├── process_llama_data.py # Process LLaMA data
│   │   ├── processed_llama_data.jsonl # Processed data
│   │   ├── sample_processed_data.json # Sample data
│   │   └── split_data.py       # Split datasets
│   ├── docker/                 # Docker configurations
│   │   └── docker-compose.yml  # LLM serving containers
│   ├── evaluation/             # Model evaluation
│   │   └── evaluate_model.py   # Evaluation scripts
│   ├── finetune/               # Fine-tuning scripts
│   │   └── train_llama.py      # LLaMA training
│   ├── serving/                # Model serving
│   │   └── serve_model.py      # Model serving script
│   ├── do_spaces_manager.py    # DigitalOcean Spaces manager
│   ├── prepare_data.sh         # Data preparation script
│   ├── requirements.txt        # Python dependencies
│   ├── run_pipeline.sh         # Pipeline runner
│   ├── test_api.py            # API testing
│   ├── 📚 DEPLOYMENT_GUIDE.md  # Deployment guide
│   ├── 📚 README.md           # LLM documentation
│   └── 📚 SYSTEM_OVERVIEW.md  # System overview
│
├── 🗄️ database/                  # Database setup
│   ├── docker-compose.yml     # Database services
│   ├── init.sql               # Initial database schema
│   └── 📚 README.md           # Database documentation
│
├── 🚀 embed_serving/             # Model serving and deployment
│   ├── docker-compose.serving.yml  # Production deployment
│   ├── Dockerfile.cpu-serving      # CPU serving container
│   ├── GPU_CPU_DEPLOYMENT_GUIDE.md # Deployment guide
│   ├── requirements_serving.txt   # Serving dependencies
│   └── scripts/                   # Serving scripts
│       ├── download_model_from_spaces.py  # Model download utility
│       └── serve_model.py         # Model serving script
│   ├── 📚 API_USAGE.md           # API usage guide
│   └── 📚 GPU_CPU_DEPLOYMENT_GUIDE.md  # Deployment guide
│
├── 🤖 models/                    # AI models and weights
│   └── bge-m3/                  # BGE-M3 embedding model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── ...
│
├── 🧪 tests/                     # Test suite
│   ├── __init__.py             # Test package init
│   ├── conftest.py             # Pytest configuration
│   ├── test_api_simple.py      # Simple API tests
│   ├── test_backend_utils.py   # Backend utility tests
│   ├── test_basic.py           # Basic functionality tests
│   ├── test_brain.py           # Brain module tests
│   ├── test_utils.py           # Utility function tests
│   └── 📚 TESTING_SUMMARY.md   # Testing documentation
│
├── 📝 docs/                      # Documentation
│   └── architecture_drawio_template.md # Architecture template
│   └── architecture_template.drawio    # Draw.io architecture file
│   └── 📚 TESTING.md           # Testing documentation
│
├── 🗂️ scripts/                   # Build and utility scripts
│   └── run_working_tests.sh    # Test runner script
│
├── 🎨 asset/                     # Assets and diagrams
│   └── architecture_template.drawio.svg # System architecture diagram
│
├── .github/                     # GitHub workflows and templates
├── .mypy_cache/                 # MyPy cache files
├── .pytest_cache/               # Pytest cache files
├── coverage.xml                 # Coverage report
├── LICENSE                      # Project license
├── Makefile                     # Build automation
├── mypy.ini                     # MyPy configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
├── pyproject.toml               # Python project configuration
├── pytest.ini                  # Pytest configuration
├── requirements_dev.txt         # Development dependencies
├── setup.cfg                    # Setup configuration
└── 📚 README.md                # This documentation
```

## 🛠️ Technology Stack

### 🖥️ Backend
- **FastAPI**: High-performance API framework with async support
- **Celery**: Distributed queue for background task processing
- **Redis**: Message broker and caching layer
- **MySQL/PostgreSQL**: Metadata and conversation history storage
- **QdrantDB**: Vector database for document embeddings
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: Database ORM
- **Databases**: Async database support

### 🔄 Data Processing
- **LlamaIndex**: Document indexing and retrieval framework
- **Pandas**: Data manipulation and analysis
- **Sentence Transformers**: Specialized embedding models
- **Custom Embedding Models**: Vietnamese-optimized embeddings

### 🤖 AI/ML
- **OpenAI API**: GPT-3.5/4 for production
- **LLaMA**: Open-source alternative for fine-tuning
- **BGE-M3**: Multilingual embedding model
- **Cohere**: Additional AI model support
- **LangChain**: LLM application framework
- **BM25**: Traditional text retrieval

### 🌐 Frontend
- **Streamlit**: Interactive web application framework
- **Python**: Core programming language
- **Tenacity**: Retry mechanism for API calls

### 🏗️ Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **DigitalOcean**: Cloud hosting platform
- **DigitalOcean Spaces**: Object storage

### 🔍 Search & Integration
- **Tavily**: Web search integration
- **Google API**: Additional search capabilities
- **Custom Search**: Legal document specific search

### 🛠️ Development Tools
- **MyPy**: Static type checking
- **Pytest**: Testing framework
- **Pre-commit**: Code quality hooks
- **Black/isort**: Code formatting
- **Coverage**: Test coverage analysis
- **Makefile**: Build automation

## 🚀 Quick Start Guide

### ⚡ Prerequisites

```bash
# System requirements
- Docker and Docker Compose v2.0+
- Python 3.8+ (for development)
- 8GB+ RAM (recommended 16GB)
- GPU with 8GB+ VRAM (optional, for faster inference)
- Ubuntu 20.04+ or macOS 10.15+

# Check requirements
docker --version
docker-compose --version
python3 --version
```

### 📦 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System

# Create Docker network
docker network create legal-chatbot-network

# Copy environment templates
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
cp database/.env.example database/.env
```

### ⚙️ 2. Environment Configuration

#### Backend Configuration
```bash
# Edit backend/.env
nano backend/.env

# Important variables:
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
DATABASE_URL=postgresql://legal_user:legal_pass@postgres:5432/legal_chatbot
REDIS_URL=redis://redis:6379/0
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

#### Database Configuration
```bash
# Edit database/.env
nano database/.env

# Database configuration:
POSTGRES_DB=legal_chatbot
POSTGRES_USER=legal_user
POSTGRES_PASSWORD=legal_pass
REDIS_PASSWORD=redis_secret_password
```

### 🐳 3. Docker Deployment

#### Option A: Full Deployment (Recommended)
```bash
# Start all services with one command
./scripts/start_all.sh

# Or manually:
# 1. Start infrastructure services
cd database && docker-compose up -d
cd ../backend && docker-compose up -d
cd ../frontend && docker-compose up -d

# Verify all services
docker ps
```

#### Option B: Step-by-step Deployment
```bash
# 1. Database services first
cd database
docker-compose up -d
sleep 30  # Wait for database startup

# 2. Backend services
cd ../backend
docker-compose up -d
sleep 60  # Wait for backend startup and migration

# 3. Frontend service
cd ../frontend
docker-compose up -d

# 4. Import initial data
cd ../backend
./import_data.sh
```

### 📊 4. Data Import

#### Automatic sample data import
```bash
cd backend

# Import training data
docker exec -it chatbot-api python /usr/src/app/src/import_data.py \
  --data-file /usr/src/app/data/train_qa_format.jsonl \
  --collection llm \
  --batch-size 100

# Check import success
curl -X POST http://localhost:8000/data/status
```

#### Custom data import
```bash
# Copy your data to backend/data/
cp your_data.jsonl backend/data/

# Import via API
curl -X POST http://localhost:8000/data/import \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/your_data.jsonl", "collection": "llm"}'
```

### 🌐 5. Access Applications

- **🖥️ Web Chat Interface**: http://localhost:8501
- **📚 API Documentation**: http://localhost:8000/docs
- **📊 API Health Check**: http://localhost:8000/health
- **🗄️ ChromaDB Dashboard**: http://localhost:8001
- **💾 Database Admin** (pgAdmin): http://localhost:5050

### ✅ 6. System Verification

```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8501/health

# Test chat API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does Vietnamese civil law say about property rights?",
    "conversation_id": "test-conversation-123"
  }'

# Check logs if there are errors
docker logs chatbot-api
docker logs chatbot-worker
docker logs streamlit-app
```

## 🔧 Advanced Configuration

### 🤖 AI Model Configuration

#### OpenAI Models
```bash
# In backend/.env
OPENAI_API_KEY=sk-your-key-here
DEFAULT_MODEL=gpt-3.5-turbo
FALLBACK_MODEL=gpt-4
MAX_TOKENS=4000
TEMPERATURE=0.1
```

#### Self-hosted Models
```bash
# Configuration for local LLM
LOCAL_MODEL_URL=http://localhost:11434
LOCAL_MODEL_NAME=llama2:13b-chat
USE_LOCAL_MODEL=true
```

### 🔍 Vector Search Configuration
```bash
# Embedding model configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
MAX_SEARCH_RESULTS=10

# ChromaDB configuration
CHROMA_COLLECTION=legal_documents
CHROMA_DISTANCE_FUNCTION=cosine
```

### 💾 Database Configuration
```bash
# PostgreSQL optimization
POSTGRES_SHARED_BUFFERS=256MB
POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
POSTGRES_MAX_CONNECTIONS=100

# Redis configuration
REDIS_MAXMEMORY=512mb
REDIS_MAXMEMORY_POLICY=allkeys-lru
```

## 📊 Data Pipeline - Detailed Guide

### 🔄 RAG Data Processing

Detailed data processing is described in:
📖 **[Data Pipeline Documentation](data_pipeline/README.md)**

#### Quick Start Data Processing
```bash
cd data_pipeline

# 1. Download Vietnamese legal corpus
jupyter notebook utils/download_embed_data.ipynb

# 2. Process fine-tuning data
jupyter notebook utils/process_finetune_data.ipynb

# 3. Validate data quality
python scripts/validate_data.py
```

### 📈 Current Data Statistics
- **📚 Legal Corpus**: 1.9M+ Vietnamese legal documents
- **💬 Training Data**: 225K+ high-quality Q&A pairs
- **🎯 Fine-tuning Sets**: 3 specialized datasets
- **📊 Coverage**: Complete coverage of major legal domains

## 🧪 Testing and Quality Assurance

### 🔍 Automated Testing
```bash
# Run full test suite
./scripts/run_tests.sh

# Unit tests
cd backend && python -m pytest tests/ -v
cd frontend && python -m pytest tests/ -v

# Integration tests
python test/test_smoke.py

# Load testing
cd backend && python test/load_test.py
```

### 📊 Performance Benchmarks
```bash
# API response time
curl -w "@curl-format.txt" -X POST http://localhost:8000/chat

# Vector search performance
python test/benchmark_search.py

# Memory usage monitoring
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## 📈 Performance and Scaling

### ⚡ Expected Performance
- **💬 Chat Response**: 2-5 seconds for complete RAG pipeline
- **🔍 Vector Search**: <1 second for semantic queries
- **📊 Document Processing**: ~1.9M documents in ~30 minutes
- **👥 Concurrent Users**: 50+ simultaneous chat sessions
- **🚀 Throughput**: 100+ requests/minute

### 🔄 Scaling Options

#### Horizontal Scaling
```bash
# Scale backend replicas
docker-compose up -d --scale chatbot-api=3

# Load balancer configuration
cd nginx && docker-compose up -d
```

#### Database Scaling
```bash
# PostgreSQL read replicas
cd database && docker-compose -f docker-compose.replica.yml up -d

# Redis clustering
cd redis-cluster && docker-compose up -d
```

#### Vector Database Scaling
```bash
# ChromaDB cluster mode
CHROMA_CLUSTER_MODE=true
CHROMA_NODES=3

# Alternative: Pinecone cloud
USE_PINECONE=true
PINECONE_API_KEY=your-pinecone-key
```

## 🚀 Production Deployment

### ☁️ Digital Ocean Deployment
```bash
# Detailed deployment guide
cd embed_serving
cat GPU_CPU_DEPLOYMENT_GUIDE.md

# Quick production deploy
docker-compose -f docker-compose.serving.yml up -d
```

### 🔐 Security Checklist
- [ ] API rate limiting enabled
- [ ] Database credentials encrypted
- [ ] HTTPS/SSL certificates setup
- [ ] Firewall rules configured
- [ ] Regular security updates scheduled
- [ ] Backup strategy established

### 📊 Monitoring Setup
```bash
# Prometheus + Grafana
cd monitoring && docker-compose up -d

# Alerts configuration
cp alerts.yml.example alerts.yml
```

## 🛠️ Development Guide

### 🏗️ Local Development
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements_dev.txt

# Start services for development
cd backend && python src/app.py
cd frontend && streamlit run chat_interface.py
```

### 🔄 Code Style and Quality
```bash
# Code formatting
black backend/src/ frontend/
isort backend/src/ frontend/

# Linting
flake8 backend/src/
pylint frontend/

# Type checking
mypy backend/src/
```

### 📚 API Development
```bash
# Generate API documentation
cd backend && python scripts/generate_docs.py

# Test API endpoints
cd backend && python scripts/test_endpoints.py

# Update OpenAPI spec
cd backend && python scripts/update_openapi.py
```

## 🤝 Contributing

### 📋 Contribution Process
1. **Fork repository** and create feature branch
2. **Implement changes** with tests and documentation
3. **Run quality checks**: `./scripts/pre-commit.sh`
4. **Submit Pull Request** with detailed description
5. **Code review** and merge approval

### 🐛 Bug Reports
- Use GitHub Issues with provided template
- Include logs, screenshots, reproduction steps
- Label appropriately (bug, enhancement, question)

### 💡 Feature Requests
- Describe use case and expected behavior
- Consider backward compatibility
- Provide implementation suggestions if possible

## 🆘 Troubleshooting

### ❌ Common Issues

#### Connection Errors
```bash
# Database connection failed
docker exec -it postgres psql -U legal_user -d legal_chatbot

# Redis connection failed
docker exec -it redis redis-cli ping

# ChromaDB not accessible
curl http://localhost:8001/api/v1/heartbeat
```

#### Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits
# In docker-compose.yml
services:
  chatbot-api:
    mem_limit: 4g
    memswap_limit: 4g
```

#### Performance Issues
```bash
# Profile application
cd backend && python -m cProfile -o profile.stats src/app.py

# Database query optimization
cd backend && python scripts/analyze_queries.py

# Vector search optimization
cd backend && python scripts/optimize_embeddings.py
```

### 🔧 Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with detailed logging
docker-compose -f docker-compose.debug.yml up
```

## 📄 Documentation

### 📚 Additional Resources
- **[Backend API Documentation](backend/README.md)** - Detailed backend implementation
- **[Data Pipeline Guide](data_pipeline/README.md)** - Comprehensive data processing
- **[Database Schema](database/README.md)** - Database design and migrations
- **[Deployment Guide](embed_serving/GPU_CPU_DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API Usage Examples](embed_serving/API_USAGE.md)** - API usage patterns

### 🎓 Tutorials
- **Setup Development Environment**: `docs/tutorials/dev-setup.md`
- **Custom Model Integration**: `docs/tutorials/custom-models.md`
- **Data Processing Workflow**: `docs/tutorials/data-processing.md`
- **Production Deployment**: `docs/tutorials/production-deploy.md`

## 📊 Metrics and Analytics

### 📈 Key Performance Indicators (KPIs)
- **User Engagement**: Session duration, questions per session
- **System Performance**: Response time, error rates
- **Content Quality**: User satisfaction ratings
- **Technical Metrics**: Uptime, throughput, resource utilization

### 📊 Dashboard Access
- **System Metrics**: http://localhost:3000 (Grafana)
- **Application Logs**: http://localhost:5601 (Kibana)
- **Database Monitoring**: http://localhost:5050 (pgAdmin)

## 🙏 Acknowledgments

### 🎓 Research & Data Sources
- **Vietnamese Legal Corpus**: Zalo AI Challenge dataset
- **ViLQA Dataset**: Vietnamese Legal Question Answering
- **Legal Document Sources**: Ministry of Justice, Vietnamese National Assembly
- **Academic Research**: VinAI, FPT AI, University partners

### 🛠️ Technology Partners
- **OpenAI**: GPT models and embedding APIs
- **Hugging Face**: Model hosting and transformers library
- **ChromaDB**: Vector database technology
- **Streamlit**: Web interface framework
- **FastAPI**: High-performance API framework

### 👥 Contributors
- **Core Development Team**: [@mikeethanh](https://github.com/mikeethanh)
- **Data Science Team**: Vietnamese legal experts
- **Beta Testers**: Legal practitioners and students
- **Open Source Community**: Contributors and issue reporters

## 📞 Support & Contact

### 🆘 Getting Help
- **📚 Documentation**: [Project Wiki](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/wiki)
- **💬 Community Discussions**: [GitHub Discussions](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/discussions)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/issues)
- **📧 Direct Contact**: mikeethanh@example.com

### 🌟 Community
- **Discord Server**: [Join our community](https://discord.gg/legal-chatbot)
- **Telegram Group**: [Vietnamese Legal AI](https://t.me/vietnamese_legal_ai)
- **LinkedIn**: [Project Updates](https://linkedin.com/company/vietnamese-legal-ai)

## 📄 License

This project is distributed under **MIT License** - see the [LICENSE](LICENSE) file for details.

### 🔒 Usage Terms
- ✅ Commercial use allowed
- ✅ Modification and distribution allowed
- ✅ Private use encouraged
- ❗ No warranty provided
- ❗ Legal advice disclaimer applies

---

## ⚠️ Disclaimer

**Important Statement**: This system is designed for **research, educational, and reference support purposes**.

🚨 **Legal Notice**:
- AI results cannot replace consultation from qualified lawyers
- Always verify information with legal professionals before making decisions
- The system may have errors and does not guarantee 100% accuracy
- Do not use for important legal matters without professional consultation

---

<div align="center">

**⭐ If this project is helpful, please star the repository to support the development team! ⭐**

Made with ❤️ for Vietnamese Legal Community

[🌟 Star](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/stargazers) • [🍴 Fork](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/fork) • [📚 Docs](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/wiki) • [💬 Discord](https://discord.gg/legal-chatbot)

</div>
