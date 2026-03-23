# Vietnamese Legal Chatbot Backend

Backend API for Vietnamese legal consultation chatbot system, built with FastAPI and microservices architecture.

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
