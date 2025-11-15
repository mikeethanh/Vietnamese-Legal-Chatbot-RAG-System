# Vietnamese Legal Chatbot RAG System 🏛️

A comprehensive Vietnamese legal consultation chatbot system built with RAG (Retrieval-Augmented Generation) technology, modern microservices architecture, and advanced AI technologies for intelligent legal document retrieval and consultation services.

## Table of Contents

- [I. Overview](#i-overview)
- [II. System Architecture](#ii-system-architecture)
- [III. Project Structure](#iii-project-structure)
- [IV. Technology Stack](#iv-technology-stack)
- [V. Quick Start Guide](#v-quick-start-guide)
- [VI. Data Pipeline Setup](#vi-data-pipeline-setup)
- [VII. Advanced Configuration](#vii-advanced-configuration)
- [VIII. Production Deployment](#viii-production-deployment)
- [IX. Testing and Quality Assurance](#ix-testing-and-quality-assurance)
- [X. Development Guide](#x-development-guide)
- [XI. Performance Monitoring](#xi-performance-monitoring)
- [XII. Troubleshooting](#xii-troubleshooting)
- [XIII. Documentation & Resources](#xiii-documentation--resources)

## I. Overview

Retrieval-augmented generation (RAG) systems combine generative AI with information retrieval to provide contextualized legal consultation services. This project deploys a comprehensive Vietnamese legal chatbot system using modern microservices architecture and advanced AI technologies.


## II. System Architecture

![System Architecture](asset/architecture_template.drawio.svg)


**(Video demonstration of system overview will be provided)**
## III. Project Structure

```
Vietnamese-Legal-Chatbot-RAG-System/
│
├── 🖥️ backend/                    # Backend API service (FastAPI)
│   ├── src/
│   │   ├── app.py                # Main FastAPI application
│   │   ├── agent.py              # AI agent with tool calling
│   │   ├── brain.py              # LLM integration and chat logic
│   │   ├── custom_embedding.py   # Custom embedding models
│   │   ├── legal_tools.py        # Legal-specific tools
│   │   ├── models.py             # Pydantic data models
│   │   ├── query_rewriter.py     # Query optimization
│   │   ├── rerank.py             # Result re-ranking
│   │   ├── search.py             # Search functionality
│   │   ├── splitter.py           # Document text splitting
│   │   ├── summarizer.py         # Text summarization
│   │   ├── tasks.py              # Celery background tasks
│   │   ├── tavily_tool.py        # Web search integration
│   │   └── vectorize.py          # Vector database operations
│   ├── docker-compose.yml       # Backend services
│   ├── Dockerfile               # Container configuration
│   ├── entrypoint.sh           # Container startup script
│   ├── import_data.sh          # Data import script
│   ├── migration_title_to_question.sql # Database migration
│   ├── requirements.txt          # Python dependencies
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
│   ├── requirements.txt        # Python dependencies
│   └── 📚 README.md           # Pipeline documentation
│
├── 🤖 llm_finetuning_serving/    # LLM fine-tuning and serving
│   ├── data_processing/        # Data processing for LLM
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
│   ├── 📚 README.md           # LLM documentation
│
├── 🗄️ database/                  # Database setup
│   ├── docker-compose.yml     # Database services
│   ├── init.sql               # Initial database schema
│
├── 🚀 embed_serving/             # Model serving and deployment
│   ├── docker-compose.serving.yml  # Production deployment
│   ├── Dockerfile.cpu-serving      # CPU serving container
│   ├── requirements_serving.txt   # Serving dependencies
│   └── scripts/                   # Serving scripts
│       ├── download_model_from_spaces.py  # Model download utility
│       └── serve_model.py         # Model serving script
│   └── 📚 GPU_CPU_DEPLOYMENT_GUIDE.md  # Deployment guide
│
├── 🧪 tests/                     # Test suite
│   ├── __init__.py             # Test package init
│   ├── conftest.py             # Pytest configuration
│   ├── test_api_simple.py      # Simple API tests
│   ├── test_backend_utils.py   # Backend utility tests
│   ├── test_basic.py           # Basic functionality tests
│   └── 📚 TESTING_SUMMARY.md   # Testing documentation
│
├── 📝 docs/                      # Documentation
│   └── architecture_drawio_template.md # Architecture template
│   └── architecture_template.drawio    # Draw.io architecture file
│   └── 📚 TESTING.md           # Testing documentation
│
├── .github/                     # GitHub workflows and templates
├── Makefile                     # Build automation
├── mypy.ini                     # MyPy configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
├── pyproject.toml               # Python project configuration
├── pytest.ini                  # Pytest configuration
├── requirements_dev.txt         # Development dependencies
├── setup.cfg                    # Setup configuration
└── 📚 README.md                # This documentation
```

## IV. Technology Stack

### 🖥️ Backend
- **FastAPI**: High-performance API framework with async support
- **Celery**: Distributed queue for background task processing
- **Redis**: Message broker and caching layer
- **MySQL/PostgreSQL**: Metadata and conversation history storage
- **QdrantDB**: Vector database for document embeddings
- **Pydantic**: Data validation and serialization
- **SQLAlchemy**: Database ORM

### 🔄 Data Processing
- **LlamaIndex**: Document indexing and retrieval framework
- **Pandas**: Data manipulation and analysis
- **Sentence Transformers**: Specialized embedding models

### 🤖 AI/ML
- **OpenAI API**: GPT-3.5/4 for production
- **LLaMA**: Open-source alternative for fine-tuning
- **BGE-M3**: Multilingual embedding model
- **Cohere**: Additional AI model support
- **BM25**: Traditional text retrieval

### 🌐 Frontend
- **Streamlit**: Interactive web application framework
- **Python**: Core programming language

### 🏗️ Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **DigitalOcean**: Cloud hosting platform
- **DigitalOcean Spaces**: Object storage

### 🔍 Search & Integration
- **Tavily**: Web search integration
- **Google API**: Additional search capabilities

## V. Quick Start Guide

### 1. Prerequisites

```bash
# System requirements
- Docker and Docker Compose v2.0+
- Python 3.8+ (for development)
- 8GB+ RAM (recommended 16GB)
- Ubuntu 20.04+ or macOS 10.15+

# Check requirements
docker --version
docker-compose --version
python3 --version
```

### 2. Clone and Setup

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

### 3. Environment Configuration

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

### 4. Service Deployment

**(Video tutorial for deployment process will be provided)**

```bash
# Start all services with Docker Compose
cd Vietnamese-Legal-Chatbot-RAG-System

# Build and start core services
docker-compose up -d --build

# Verify services are running
docker-compose ps
```

### 5. Data Import

#### Automatic sample data import
```bash
cd backend

# Import training data
docker exec -it chatbot-api python /usr/src/app/src/import_data.py \
  --data-file /usr/src/app/data/train_qa_format.jsonl \
  --collection llm \
  --batch-size 100

```

### 6. Access Applications

- **🖥️ Web Chat Interface**: http://localhost:8501
- **📚 API Documentation**: http://localhost:8000/docs
- **📊 API Health Check**: http://localhost:8000/health
- **🗄️ ChromaDB Dashboard**: http://localhost:8001
- **💾 Database Admin** (pgAdmin): http://localhost:5050

### 7. System Verification

**(Video demonstration of system verification will be provided)**

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

## VI. Data Pipeline Setup

**(Video tutorial for data pipeline setup will be provided)**

### 1. Legal Document Processing

Detailed data processing workflow is described in the [Data Pipeline Documentation](data_pipeline/README.md).

```bash
cd data_pipeline

# 1. Download Vietnamese legal corpus
jupyter notebook utils/download_embed_data.ipynb

# 2. Process fine-tuning data
jupyter notebook utils/process_finetune_data.ipynb

# 3. Validate data quality
python scripts/validate_data.py
```

### 2. Data Statistics
- **Legal Corpus**: 1.9M+ Vietnamese legal documents
- **Training Data**: 225K+ high-quality Q&A pairs  
- **Fine-tuning Sets**: 3 specialized datasets
- **Coverage**: Complete coverage of major legal domains


## VIII. Production Deployment

**(Video tutorial for production deployment will be provided)**

### 1. Cloud Deployment Guide

Detailed deployment instructions are available in:
📖 **[GPU CPU Deployment Guide](embed_serving/GPU_CPU_DEPLOYMENT_GUIDE.md)**

```bash
# Quick production deployment on Digital Ocean
cd embed_serving
docker-compose -f docker-compose.serving.yml up -d
```

### 2. Security Configuration

- [ ] API rate limiting enabled
- [ ] Database credentials encrypted  
- [ ] HTTPS/SSL certificates setup
- [ ] Firewall rules configured
- [ ] Regular security updates scheduled
- [ ] Backup strategy established


## X. Development Guide

### 4. Contributing Guidelines

1. **Fork repository** and create feature branch
2. **Implement changes** with tests and documentation  
3. **Run quality checks**: `./scripts/pre-commit.sh`
4. **Submit Pull Request** with detailed description
5. **Code review** and merge approval

## XIII. Documentation & Resources

### 1. Technical Documentation
- **[Backend API Documentation](backend/README.md)** - Detailed backend implementation
- **[Data Pipeline Guide](data_pipeline/README.md)** - Comprehensive data processing  
- **[Database Schema](database/README.md)** - Database design and migrations
- **[Deployment Guide](embed_serving/GPU_CPU_DEPLOYMENT_GUIDE.md)** - Production deployment
- **[API Usage Examples](embed_serving/API_USAGE.md)** - API usage patterns


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
