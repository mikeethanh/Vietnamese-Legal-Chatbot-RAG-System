# Vietnamese Legal Chatbot Backend

Backend API for Vietnamese legal consultation chatbot system, built with FastAPI and microservices architecture.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │    Celery       │    │   ChromaDB      │
│   (API Server)  │───▶│   (Workers)     │───▶│  (Vector DB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │   File Storage  │
│  (Metadata DB)  │    │  (Message Broker) │    │     (S3)        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
backend/
├── src/                        # Main source code
│   ├── app.py                 # FastAPI application entry point
│   ├── brain.py               # LLM integration and chat logic
│   ├── vectorize.py           # Vector database operations
│   ├── database.py            # Database connections and models
│   ├── models.py              # Pydantic data models
│   ├── tasks.py               # Celery background tasks
│   ├── cache.py               # Redis caching utilities
│   ├── configs.py             # Configuration management
│   ├── splitter.py            # Document text splitting
│   ├── summarizer.py          # Text summarization
│   ├── legal_tools.py         # Legal-specific tools
│   ├── agent.py               # AI agent logic
│   ├── query_rewriter.py      # Query rewriting for better search
│   ├── rerank.py              # Result re-ranking
│   ├── tavily_tool.py         # Tavily search integration
│   ├── custom_embedding.py    # Custom embedding models
│   ├── import_data.py         # Data import utilities
│   └── utils.py               # Utility functions
├── data/                       # Data files
│   └── train_qa_format.jsonl  # Training data
├── requirements.txt           # Python dependencies
├── Dockerfile                # Container configuration
├── docker-compose.yml        # Multi-container setup
├── entrypoint.sh             # Container startup script
├── import_data.sh            # Data import script
└── README.md                 # This documentation
```

## 🚀 Main Features

### Core API Endpoints
- **Chat Interface**: `/chat` - Main endpoint for conversation
- **Document Import**: `/data/import` - Import data into vector database
- **Health Check**: `/health` - Check system status
- **Search**: `/search` - Semantic search in legal corpus

### AI Capabilities
- **RAG (Retrieval-Augmented Generation)**: Combines search and generation
- **Query Rewriting**: Optimize questions for better search
- **Context-Aware Responses**: Answers based on context and chat history
- **Legal Document Processing**: Process and understand legal documents
- **Multi-format Support**: Support multiple input data formats

### Backend Services
- **Async Processing**: Celery workers for background tasks
- **Caching**: Redis for cache and session management
- **Vector Storage**: ChromaDB for semantic search
- **Metadata Storage**: PostgreSQL for conversation history

## 🛠️ Installation and Deployment

### 1. Prerequisites
```bash
# System requirements
- Docker and Docker Compose
- Python 3.8+
- 8GB+ RAM (recommended)
- GPU (optional, for faster inference)
```

### 2. Environment Setup
```bash
# Clone repository
cd backend

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Configuration (.env)
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/legal_chatbot
REDIS_URL=redis://localhost:6379/0

# Vector Database
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION=llm

# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_TOKENS=4000
TEMPERATURE=0.1

# File Storage
S3_BUCKET=legal-documents
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=50
```

### 4. Deploy with Docker
```bash
# Build and start all services
docker compose up -d --build

# Check logs
docker logs -f chatbot-api
docker logs -f chatbot-worker
docker logs -f redis
docker logs -f postgres
```

### 5. Manual Deploy (Development)
```bash
# Install dependencies
pip install -r requirements.txt

# Start database services
docker compose up -d postgres redis chromadb

# Start API server
python src/app.py

# Start Celery worker (separate terminal)
celery -A src.tasks worker --loglevel=info

# Start Celery beat (separate terminal, if needed)
celery -A src.tasks beat --loglevel=info
```

## 📊 Data Import

### Using automatic script
```bash
# Import from JSONL file
./import_data.sh

# Or use Python script
python src/import_data.py --data-file data/train_qa_format.jsonl --collection llm --batch-size 100
```

### Using API endpoint
```bash
# POST request to API
curl -X POST http://localhost:8000/data/import \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/train_qa_format.jsonl", "collection": "llm"}'
```

### Using Docker
```bash
# Import in container
docker exec -it chatbot-api python /usr/src/app/src/import_data.py \
  --data-file /usr/src/app/data/train_qa_format.jsonl \
  --collection llm \
  --batch-size 100
```

## 🗄️ Database Setup

### PostgreSQL
```sql
-- Create database
CREATE DATABASE legal_chatbot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE legal_chatbot;

-- Conversations table
CREATE TABLE chat_conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(50) NOT NULL DEFAULT '',
    bot_id VARCHAR(100) NOT NULL,
    user_id VARCHAR(100) NOT NULL,
    message TEXT,
    is_request BOOLEAN DEFAULT TRUE,
    completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents metadata table
CREATE TABLE document (
    id SERIAL PRIMARY KEY,
    question VARCHAR(2000) NOT NULL,
    content TEXT,
    source VARCHAR(500),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_conversations_id ON chat_conversations(conversation_id);
CREATE INDEX idx_conversations_user ON chat_conversations(user_id);
CREATE INDEX idx_document_source ON document(source);
```

### Vector Database
```bash
# Access ChromaDB dashboard
http://localhost:8000/docs

# Check collections
curl http://localhost:8000/api/v1/collections
```

## 🧪 Testing

### Unit Tests
```bash
# Run test suite
python -m pytest tests/ -v

# Test coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What does Vietnamese civil law say about property rights?", "conversation_id": "test-123"}'

# Test search endpoint
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "property rights", "limit": 5}'
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8000/health

# Using wrk
wrk -t12 -c400 -d30s --script=post.lua http://localhost:8000/chat
```

## 📈 Monitoring and Logging

### Logs
```bash
# View logs realtime
docker logs -f chatbot-api
docker logs -f chatbot-worker

# Logs with filter
docker logs chatbot-api 2>&1 | grep ERROR
```

### Metrics
- **Response Time**: API response time
- **Throughput**: Requests per second
- **Error Rate**: Error percentage
- **Database Performance**: Query time, connection pool
- **Memory Usage**: RAM and GPU utilization

### Health Monitoring
```bash
# Health check endpoint
GET /health

# Detailed system info
GET /health/detailed

# Database connection check
GET /health/database
```

## 🔧 Configuration Details

### Model Configuration
```python
# In configs.py
MODEL_CONFIG = {
    "gpt-3.5-turbo": {
        "max_tokens": 4000,
        "temperature": 0.1,
        "top_p": 1.0
    },
    "gpt-4": {
        "max_tokens": 8000,
        "temperature": 0.1,
        "top_p": 1.0
    }
}
```

### Embedding Configuration
```python
EMBEDDING_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "normalize": True,
    "batch_size": 32
}
```

### Caching Strategy
```python
CACHE_CONFIG = {
    "redis_url": "redis://localhost:6379/0",
    "default_ttl": 3600,  # 1 hour
    "embedding_ttl": 86400,  # 24 hours
    "conversation_ttl": 1800  # 30 minutes
}
```

## 🔐 Security

### API Security
- Rate limiting to prevent abuse
- API key authentication
- Input validation and sanitization
- CORS configuration

### Data Security
- Database encryption at rest
- Redis password protection
- Secure environment variables
- Regular security updates

## 📚 API Documentation

### Swagger UI
Access: `http://localhost:8000/docs`

### Key Endpoints

#### POST /chat
```json
{
    "message": "User question",
    "conversation_id": "unique-conversation-id",
    "user_id": "user-123"
}
```

#### POST /search
```json
{
    "query": "Search keyword",
    "limit": 10,
    "collection": "llm"
}
```

#### POST /data/import
```json
{
    "file_path": "path/to/data.jsonl",
    "collection": "llm",
    "batch_size": 100
}
```

## 🚀 Performance Optimization

### Database Optimization
- Connection pooling
- Query optimization
- Proper indexing
- Regular VACUUM and ANALYZE

### Caching Strategy
- Query result caching
- Embedding caching
- Session caching
- CDN for static assets

### Scaling Options
- Horizontal scaling with load balancer
- Database read replicas
- Redis clustering
- Vector database sharding

## 🆘 Troubleshooting

### Common Issues

#### Connection Errors
```bash
# Check database connection
docker exec -it postgres psql -U user -d legal_chatbot

# Check Redis
docker exec -it redis redis-cli ping

# Check ChromaDB
curl http://localhost:8000/api/v1/heartbeat
```

#### Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limit in docker-compose.yml
services:
  chatbot-api:
    mem_limit: 4g
```

#### Performance Issues
```bash
# Profile API requests
python -m cProfile -o profile.stats src/app.py

# Analyze with snakeviz
pip install snakeviz
snakeviz profile.stats
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: mikeethanh@example.com

## 📄 License

This project is distributed under MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Note**: This system is designed for research and educational purposes. Legal advice should always be verified with qualified legal professionals.
