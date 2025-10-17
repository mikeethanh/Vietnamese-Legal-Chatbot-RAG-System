# Vietnamese Legal Chatbot RAG System

A comprehensive RAG (Retrieval-Augmented Generation) system for Vietnamese legal document question-answering, built with modern AI technologies and microservices architecture.

## 🏛️ Overview

This system provides intelligent legal consultation services for Vietnamese users by combining:
- **Large Language Models** for natural language understanding and generation
- **Vector Database** for semantic document retrieval
- **Legal Document Processing Pipeline** for data preparation
- **Web-based Chat Interface** for user interaction
- **Scalable Backend Architecture** with microservices

## 🚀 Features

### Core Capabilities
- **Vietnamese Legal Q&A**: Answers legal questions based on Vietnamese law documents
- **Document Retrieval**: Semantic search through legal corpus using vector embeddings
- **Context-Aware Responses**: Provides accurate answers with source document references
- **Real-time Chat Interface**: User-friendly web interface for legal consultations
- **Multi-format Support**: Processes various document formats (PDF, CSV, JSON, JSONL)

### Technical Features
- **RAG Architecture**: Combines retrieval and generation for accurate responses
- **Vector Search**: ChromaDB for efficient semantic document retrieval
- **LLM Integration**: Support for multiple language models (OpenAI GPT, LLaMA)
- **Scalable Processing**: Apache Spark for large-scale document processing
- **Microservices**: Dockerized components with Redis queue management
- **Data Pipeline**: Automated ETL for legal document ingestion

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │  Data Pipeline  │
│   (Streamlit)   │───▶│   (FastAPI)     │───▶│   (Spark)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Vector DB      │    │    Redis        │    │   PostgreSQL    │
│  (ChromaDB)     │    │   (Queue)       │    │  (Metadata)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Vietnamese-Legal-Chatbot-RAG-System/
├── backend/                    # FastAPI backend service
│   ├── src/
│   │   ├── app.py             # Main FastAPI application
│   │   ├── brain.py           # LLM integration and chat logic
│   │   ├── vectorize.py       # Vector database operations
│   │   ├── database.py        # Database connections
│   │   ├── models.py          # Data models
│   │   ├── tasks.py           # Celery background tasks
│   │   ├── cache.py           # Caching utilities
│   │   ├── configs.py         # Configuration management
│   │   ├── splitter.py        # Document text splitting
│   │   ├── summarizer.py      # Text summarization
│   │   └── utils.py           # Utility functions
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Container configuration
│   ├── docker-compose.yml    # Backend services
│   ├── entrypoint.sh         # Container startup script
│   └── README.md             # Backend documentation
├── frontend/                  # Streamlit web interface
│   ├── chat_interface.py     # Main chat application
│   ├── config.toml          # Streamlit configuration
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Container configuration
│   ├── docker-compose.yml   # Frontend services
│   └── entrypoint.sh        # Container startup script
├── data_pipeline/            # Data processing pipeline
│   ├── utils/               # Processing utilities
│   ├── data/                # Raw and processed data
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # Pipeline documentation (detailed)
├── database/                 # Database setup
│   ├── init.sql             # Initial database schema
│   ├── docker-compose.yml   # Database services
│   ├── .env                 # Database environment
│   └── README.md            # Database documentation
├── test/                     # Test suite
│   └── test_smoke.py        # Smoke tests
├── requirements_dev.txt      # Development dependencies
└── LICENSE                   # Project license
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: High-performance API framework
- **Celery**: Distributed task queue for background processing
- **Redis**: Message broker and caching
- **PostgreSQL**: Metadata and conversation storage
- **ChromaDB**: Vector database for document embeddings

### Data Processing
- **Apache Spark**: Large-scale data processing
- **Pandas**: Data manipulation and analysis
- **Transformers**: Text embedding generation
- **OpenAI API**: GPT model integration

### Frontend
- **Streamlit**: Interactive web application framework
- **Python**: Core programming language

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **AWS S3**: Cloud storage (optional)

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM recommended
- AWS credentials (for S3 integration, optional)

### 1. Clone Repository
```bash
git clone https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System.git
cd Vietnamese-Legal-Chatbot-RAG-System
```

### 2. Environment Setup
```bash
# Copy environment templates
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Edit configurations as needed
nano backend/.env
nano frontend/.env
```

### 3. Start Services
```bash
# Start database services first
cd database && docker-compose up -d

# Start backend services
cd ../backend && docker-compose up -d

# Start frontend application
cd ../frontend && docker-compose up -d

# Alternative: Start individual services manually
cd backend && python src/app.py
cd frontend && streamlit run chat_interface.py
```

### 4. Access Application
- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Database Admin**: http://localhost:5432

## 📊 Data Pipeline

### Processing Legal Documents

1. **Raw Data Preparation**
```bash
cd data_pipeline

# Place your legal documents in data/raw/rag_corpus/
# Supported formats: CSV, JSON, JSONL
```

2. **Run Processing Pipeline**

For detailed data processing instructions, see the comprehensive guide in:
📖 **[Data Pipeline Documentation](data_pipeline/README.md)**

The data pipeline supports:
- RAG corpus processing with Apache Spark
- Fine-tuning data preparation
- Document cleaning and validation
- S3 cloud storage integration

## 🔧 Configuration

### Backend Configuration
```bash
# backend/.env
OPENAI_API_KEY=your_openai_api_key
CHROMA_HOST=localhost
CHROMA_PORT=8000
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/legal_chatbot
```

### Frontend Configuration
```bash
# frontend/.env
BACKEND_URL=http://localhost:8000
STREAMLIT_SERVER_PORT=8501
```

## 🧪 Testing

```bash
# Run smoke tests
python test/test_smoke.py

# Test individual components
cd backend && python -m pytest tests/
cd frontend && python -m pytest tests/
```

## 📈 Performance & Scaling

### Expected Performance
- **Document Processing**: ~1.9M documents processed in ~30 minutes
- **Vector Search**: Sub-second response for semantic queries
- **Chat Response**: 2-5 seconds for complete RAG pipeline
- **Concurrent Users**: 50+ simultaneous chat sessions

### Scaling Options
- **Horizontal Scaling**: Multiple backend replicas with load balancer
- **Vector DB Scaling**: ChromaDB cluster or Pinecone cloud
- **Data Processing**: Spark cluster for large document volumes
- **Caching**: Redis for frequent queries and embeddings

## 🤖 Model Integration

### Supported Models
- **OpenAI GPT-3.5/4**: Production-ready commercial API
- **LLaMA 2/3**: Open-source alternative for self-hosting
- **Vietnamese-specific models**: Fine-tuned on legal corpus

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Vietnamese legal document sources
- Open-source AI/ML community
- Contributors and beta testers

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/discussions)
- **Email**: [mikeethanh@example.com](mailto:mikeethanh@example.com)

---

**Note**: This system is designed for educational and research purposes. Legal advice should always be verified with qualified legal professionals.