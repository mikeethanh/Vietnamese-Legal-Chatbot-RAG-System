# Vietnamese Legal Chatbot Backend - Comprehensive Technical Documentation

## 📋 Tổng quan về Backend System

**Vietnamese Legal Chatbot Backend** là một hệ thống backend phức tạp được thiết kế để cung cấp tư vấn pháp luật Việt Nam thông qua AI. Hệ thống được xây dựng trên **FastAPI** với kiến trúc microservices, tích hợp **RAG (Retrieval-Augmented Generation)**, **vector database**, và **AI agents** để cung cấp tư vấn pháp luật chính xác và contextual.

## 🏗️ Kiến trúc Hệ thống (System Architecture)

### Tổng quan Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Frontend Interface]
        API[REST API Clients]
    end
    
    subgraph "API Gateway Layer"
        FASTAPI[FastAPI Server]
        CORS[CORS Middleware]
        AUTH[Authentication]
    end
    
    subgraph "Processing Layer"
        BRAIN[Brain Module]
        AGENT[AI Agent System]
        QUERY[Query Rewriter]
        RERANK[Re-ranking Engine]
    end
    
    subgraph "Search & Retrieval Layer"
        SEARCH[Search Engine]
        VECTOR[Vectorization]
        EMBED[Custom Embedding]
    end
    
    subgraph "Task Processing Layer"
        CELERY[Celery Workers]
        TASKS[Background Tasks]
        REDIS[Redis Message Broker]
    end
    
    subgraph "Data Layer"
        QDRANT[Qdrant Vector DB]
        POSTGRES[PostgreSQL]
        CACHE[Redis Cache]
        FILES[File Storage]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API]
        TAVILY[Tavily Search]
        CUSTOM[Custom LLM Service]
    end
    
    UI --> FASTAPI
    API --> FASTAPI
    FASTAPI --> BRAIN
    FASTAPI --> AGENT
    BRAIN --> SEARCH
    BRAIN --> OPENAI
    AGENT --> QUERY
    AGENT --> RERANK
    SEARCH --> VECTOR
    VECTOR --> EMBED
    VECTOR --> QDRANT
    CELERY --> TASKS
    TASKS --> REDIS
    TASKS --> POSTGRES
    TASKS --> CACHE
    AGENT --> TAVILY
    EMBED --> CUSTOM
```

### Kiến trúc Chi tiết (Detailed Architecture)

#### **Layer 1: API Gateway (FastAPI)**

**Vai trò**: Entry point cho tất cả requests, handling authentication, validation, và routing.

**Core Components**:
```python
# app.py - Main FastAPI application
@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    """
    Main endpoint cho chat completion
    - Synchronous: Direct response
    - Asynchronous: Task-based với Celery
    """
    
@app.post("/collection/create") 
async def create_vector_collection():
    """Vector collection management"""
    
@app.post("/document/create")
async def create_document():
    """Document ingestion endpoint"""
```

**Features**:
- **Async/Sync Modes**: Support cả synchronous và asynchronous processing
- **Task Management**: Integration với Celery cho long-running tasks
- **Health Monitoring**: Health check endpoints
- **Error Handling**: Comprehensive error handling và logging

#### **Layer 2: Brain Module (LLM Integration)**

**Vai trò**: Central intelligence của system, orchestrating LLM calls và RAG pipeline.

**Core Functions**:
```python
# brain.py - Core AI logic
def openai_chat_complete(messages, model="gpt-4o-mini"):
    """OpenAI integration với conversation handling"""
    
def detect_user_intent(history, message):
    """
    Phân tích intent và context từ conversation history
    - Follow-up question detection
    - Query reformulation
    - Context preservation
    """
    
def get_embedding(text, model=None):
    """Custom Vietnamese embedding generation"""
```

**Capabilities**:
- **Multi-turn Conversations**: Context-aware conversation handling
- **Intent Detection**: Phân biệt câu hỏi độc lập vs follow-up questions
- **Query Reformulation**: Chuyển đổi follow-up questions thành standalone queries
- **Vietnamese Legal Context**: Specialized prompt engineering cho legal domain

#### **Layer 3: AI Agent System**

**Vai trò**: Intelligent agent system với specialized legal tools và reasoning capabilities.

```python
# agent.py - ReAct Agent Implementation
class LegalAgent(ReActAgent):
    """
    Specialized legal agent với integrated tools:
    - Legal calculations (penalty, inheritance, etc.)
    - Document search và retrieval
    - External search integration
    """
```

**Agent Tools**:

1. **Legal Calculation Tools**:
   ```python
   def contract_penalty_calculator(contract_value, penalty_rate, days_late):
       """Tính tiền phạt vi phạm hợp đồng theo Bộ luật Dân sự"""
   
   def inheritance_share_calculator(estate_value, heirs_info):
       """Tính phần thừa kế theo luật Việt Nam"""
   
   def legal_age_checker(birth_year, action_type):
       """Kiểm tra tuổi pháp lý cho các hành vi dân sự"""
   ```

2. **Search Tools**:
   ```python
   def legal_document_search(query, top_k=5):
       """Tìm kiếm trong database pháp luật Việt Nam"""
   
   def external_legal_search(query):
       """Tìm kiếm external với Tavily API"""
   ```

#### **Layer 4: Search & Retrieval System**

**Core Components**:

1. **Search Engine** (`search.py`):
   ```python
   def search_engine(collection_name, query_text, top_k=5):
       """
       Advanced search với multiple strategies:
       - Semantic search (dense vectors)
       - Keyword matching 
       - Hybrid scoring
       - Metadata filtering
       """
   ```

2. **Vectorization System** (`vectorize.py`):
   ```python
   def add_vector(collection_name, vectors, batch_size=100):
       """
       Optimized vector insertion với:
       - Batch processing
       - Metadata enrichment
       - Error handling
       - Performance monitoring
       """
   ```

3. **Custom Embedding Service** (`custom_embedding.py`):
   ```python
   class CustomEmbeddingService:
       """
       Vietnamese-optimized embedding service:
       - BGE-M3 model integration
       - Fallback mechanisms
       - Health monitoring
       - Caching capabilities
       """
   ```

#### **Layer 5: Data Processing Pipeline**

1. **Document Splitter** (`splitter.py`):
   ```python
   def split_documents(documents, chunk_size=1000, overlap=200):
       """
       Intelligent document chunking:
       - Semantic boundary preservation
       - Vietnamese text handling
       - Legal document structure awareness
       """
   ```

2. **Reranking System** (`rerank.py`):
   ```python
   def rerank_results(query, documents, top_k=5):
       """
       Advanced result reranking:
       - Semantic relevance scoring
       - Legal domain expertise
       - Query-document matching
       """
   ```

3. **Query Rewriter** (`query_rewriter.py`):
   ```python
   def rewrite_query(original_query, context):
       """
       Intelligent query enhancement:
       - Legal terminology expansion
       - Context integration
       - Ambiguity resolution
       """
   ```

## 🛠️ Core Modules Deep Dive

### 1. Brain Module (`brain.py`)

#### **Conversation Management**

```python
def detect_user_intent(history, message):
    """
    Sophisticated intent detection system for Vietnamese legal queries
    
    Features:
    - Follow-up question detection using linguistic indicators
    - Context preservation across multi-turn conversations  
    - Query reformulation for standalone processing
    - Vietnamese legal context optimization
    """
    
    # Vietnamese follow-up indicators
    follow_up_indicators = [
        "đó", "này", "kia", "thế", "vậy", "nữa", 
        "còn", "như vậy", "như thế"
    ]
    
    # Advanced prompt engineering cho legal context
    system_prompt = """
    Bạn là trợ lý AI chuyên về luật pháp Việt Nam. 
    Nhiệm vụ: Viết lại câu hỏi tiếp theo thành câu hỏi độc lập, 
    rõ ràng và đầy đủ ngữ cảnh pháp lý.
    """
```

#### **LLM Integration Strategies**

```python
class LLMManager:
    """
    Advanced LLM management với multiple strategies:
    - Model selection based on query complexity
    - Token optimization
    - Cost management  
    - Error handling và fallbacks
    """
    
    def select_model(self, query_complexity):
        """Dynamic model selection"""
        if query_complexity == "simple":
            return "gpt-4o-mini"  # Cost-effective
        elif query_complexity == "complex":
            return "gpt-4o"       # High accuracy
        else:
            return "gpt-4o-mini"  # Default
```

### 2. Vectorization System (`vectorize.py`)

#### **Qdrant Integration**

```python
class EnhancedVectorStore:
    """
    Production-ready vector store với advanced features:
    - Optimized batch processing
    - Metadata enrichment
    - Health monitoring
    - Performance analytics
    """
    
    def __init__(self):
        self.client = QdrantClient(url="http://qdrant-db:6333")
        self.default_vector_size = 1024  # BGE-M3 embedding dimension
        
    def create_collection_with_config(self, name, vector_size=1024):
        """
        Enhanced collection creation với optimized settings:
        - DOT product distance (tối ưu cho normalized vectors)
        - HNSW indexing parameters
        - Quantization settings
        """
        return self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size, 
                distance=Distance.DOT,  # Optimized cho normalized vectors
                hnsw_config={
                    "m": 16,           # Số connections per node
                    "ef_construct": 100, # Construction parameter
                    "full_scan_threshold": 10000
                }
            ),
            optimizers_config={
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 2
            }
        )
```

#### **Advanced Search Capabilities**

```python
def hybrid_search(collection_name, query_text, filters=None, top_k=5):
    """
    Hybrid search combining multiple strategies:
    1. Semantic vector search
    2. Keyword matching
    3. Metadata filtering
    4. Score fusion
    """
    
    # Get embedding
    query_vector = get_custom_embedding(query_text)
    
    # Semantic search
    semantic_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k * 2,  # Over-retrieve for reranking
        query_filter=filters
    )
    
    # Score combination và reranking
    final_results = rerank_and_fuse_scores(
        semantic_results, 
        query_text,
        top_k=top_k
    )
    
    return final_results
```

### 3. Custom Embedding System (`custom_embedding.py`)

#### **BGE-M3 Integration**

```python
class VietnameseLegalEmbedding:
    """
    Specialized embedding service cho Vietnamese legal documents:
    - BGE-M3 model optimization
    - Legal terminology handling
    - Multi-functionality support (dense, sparse, multi-vector)
    """
    
    def __init__(self):
        self.api_url = os.getenv("CUSTOM_EMBEDDING_API_URL")
        self.embedding_dim = 1024  # BGE-M3 output dimension
        self.cache = {}  # LRU cache cho frequent queries
        
    def get_embedding(self, text, embedding_type="dense"):
        """
        Multi-modal embedding generation:
        - dense: Semantic understanding (default)
        - sparse: Keyword matching
        - multi_vector: Fine-grained interactions
        """
        
        # Text preprocessing cho Vietnamese legal content
        cleaned_text = self._preprocess_legal_text(text)
        
        # Check cache first
        cache_key = f"{embedding_type}:{hash(cleaned_text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # API call với error handling
        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={
                    "text": cleaned_text,
                    "type": embedding_type,
                    "model": "bge-m3"
                },
                timeout=30
            )
            
            embedding = response.json()["embedding"]
            self.cache[cache_key] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return self._get_fallback_embedding(text)
```

### 4. Agent System (`agent.py`)

#### **ReAct Agent Implementation**

```python
class VietnameseLegalAgent(ReActAgent):
    """
    Specialized ReAct agent for Vietnamese legal consultation:
    - Legal reasoning capabilities
    - Tool integration
    - Multi-step problem solving
    """
    
    def __init__(self):
        # Initialize LLM
        self.llm = OpenAI(
            model="gpt-4o",
            temperature=0.1,  # Low temperature cho legal accuracy
            system_message=self._get_legal_system_prompt()
        )
        
        # Initialize tools
        self.tools = self._initialize_legal_tools()
        
        # Create agent
        super().__init__(llm=self.llm, tools=self.tools, verbose=True)
        
    def _get_legal_system_prompt(self):
        return """
        Bạn là một luật sư chuyên nghiệp về pháp luật Việt Nam.
        
        NHIỆM VỤ:
        - Tư vấn pháp luật chính xác theo quy định của Việt Nam
        - Sử dụng các công cụ để tính toán và tìm kiếm thông tin
        - Cung cấp căn cứ pháp lý rõ ràng cho mỗi tư vấn
        
        NGUYÊN TẮC:
        - Luôn dựa trên văn bản pháp luật hiện hành
        - Không đưa ra tư vấn nếu thiếu thông tin
        - Khuyến nghị tìm hiểu thêm hoặc tham khảo luật sư khi cần
        """
```

#### **Legal Tool Integration**

```python
class LegalCalculationTools:
    """Comprehensive legal calculation toolkit"""
    
    @staticmethod
    def contract_penalty_tool():
        """Tính tiền phạt vi phạm hợp đồng"""
        return FunctionTool.from_defaults(
            fn=calculate_contract_penalty,
            name="contract_penalty_calculator",
            description="""
            Tính tiền phạt vi phạm hợp đồng theo Bộ luật Dân sự Việt Nam.
            Input: contract_value (giá trị hợp đồng), penalty_rate (% phạt/ngày), days_late (số ngày chậm)
            Output: Chi tiết tính toán tiền phạt
            """
        )
    
    @staticmethod 
    def inheritance_calculation_tool():
        """Tính thừa kế theo pháp luật"""
        return FunctionTool.from_defaults(
            fn=calculate_inheritance_share,
            name="inheritance_calculator", 
            description="""
            Tính phần thừa kế theo pháp luật Việt Nam.
            Input: estate_value (tài sản), heirs_info (thông tin người thừa kế)
            Output: Phân chia tài sản thừa kế chi tiết
            """
        )
```

## 📊 Data Flow và Processing Pipeline

### Request Processing Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Brain
    participant Agent  
    participant Search
    participant Vector
    participant LLM
    
    Client->>FastAPI: POST /chat/complete
    FastAPI->>Brain: llm_handle_message()
    Brain->>Brain: detect_user_intent()
    Brain->>Agent: query_agent()
    Agent->>Search: search_legal_docs()
    Search->>Vector: hybrid_search()
    Vector->>Vector: get_embedding()
    Vector-->>Search: search_results
    Search-->>Agent: relevant_docs
    Agent->>LLM: generate_response()
    LLM-->>Agent: response
    Agent-->>Brain: final_answer
    Brain-->>FastAPI: response
    FastAPI-->>Client: JSON response
```

### Document Ingestion Pipeline

```mermaid
graph LR
    A[Legal Documents] --> B[Text Extraction]
    B --> C[Document Splitting]
    C --> D[Metadata Extraction] 
    D --> E[Embedding Generation]
    E --> F[Vector Storage]
    F --> G[Index Building]
    G --> H[Quality Validation]
```

### Advanced Processing Strategies

#### **Multi-Strategy Search**

```python
class HybridSearchEngine:
    """
    Advanced search combining multiple retrieval strategies:
    1. Semantic Vector Search (BGE-M3 dense)
    2. Lexical Search (BGE-M3 sparse) 
    3. Metadata Filtering
    4. Result Fusion và Reranking
    """
    
    def search(self, query, top_k=5, strategy="hybrid"):
        if strategy == "semantic":
            return self._semantic_search(query, top_k)
        elif strategy == "lexical":
            return self._lexical_search(query, top_k)  
        elif strategy == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _hybrid_search(self, query, top_k):
        """
        Fusion của semantic và lexical search:
        - Semantic: Hiểu ý nghĩa sâu
        - Lexical: Exact term matching
        - Score fusion: RRF (Reciprocal Rank Fusion)
        """
        
        # Parallel search
        semantic_results = self._semantic_search(query, top_k * 2)
        lexical_results = self._lexical_search(query, top_k * 2)
        
        # Result fusion với RRF
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, 
            lexical_results,
            k=60  # RRF parameter
        )
        
        return fused_results[:top_k]
```

#### **Intelligent Query Processing**

```python
class QueryProcessor:
    """
    Sophisticated query processing for Vietnamese legal queries:
    - Intent detection
    - Query expansion  
    - Context integration
    - Ambiguity resolution
    """
    
    def process_query(self, query, conversation_history=None):
        """Multi-stage query processing"""
        
        # Stage 1: Intent Detection
        intent = self._detect_intent(query, conversation_history)
        
        # Stage 2: Query Reformulation (if needed)
        if intent["type"] == "follow_up":
            query = self._reformulate_query(query, conversation_history)
        
        # Stage 3: Query Expansion
        expanded_query = self._expand_legal_terms(query)
        
        # Stage 4: Context Integration
        contextualized_query = self._integrate_context(
            expanded_query, 
            conversation_history
        )
        
        return {
            "original": query,
            "processed": contextualized_query,
            "intent": intent,
            "metadata": self._extract_metadata(query)
        }
```

## ⚙️ Configuration Management (`configs.py`)

### Environment-based Configuration

```python
class Config:
    """
    Comprehensive configuration management:
    - Environment-specific settings
    - Feature flags
    - Service endpoints
    - Model parameters
    """
    
    # Database Configuration
    POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost/db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    # AI Service Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CUSTOM_EMBEDDING_URL = os.getenv("CUSTOM_EMBEDDING_URL", "http://localhost:5001")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Model Configuration
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # Search Configuration  
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Performance Configuration
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))
    
    # Feature Flags
    ENABLE_AGENT_MODE = os.getenv("ENABLE_AGENT_MODE", "true").lower() == "true"
    ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
```

## 🔧 Task Management (`tasks.py`)

### Celery Integration

```python
from celery import Celery

# Celery app initialization
celery_app = Celery(
    "legal_chatbot",
    broker=Config.CELERY_BROKER_URL,
    backend=Config.CELERY_RESULT_BACKEND,
    include=['tasks']
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Ho_Chi_Minh',
    enable_utc=True,
    task_routes={
        'tasks.llm_handle_message': {'queue': 'llm_queue'},
        'tasks.index_document_v2': {'queue': 'indexing_queue'},
        'tasks.generate_embeddings': {'queue': 'embedding_queue'},
    }
)

@celery_app.task(bind=True)
def llm_handle_message(self, bot_id, user_id, user_message):
    """
    Async task cho LLM message processing:
    - Long-running conversation handling
    - Resource-intensive operations
    - Error handling và retry logic
    """
    try:
        # Task processing logic
        result = process_llm_message(bot_id, user_id, user_message)
        return result
    except Exception as exc:
        # Retry logic với exponential backoff
        raise self.retry(exc=exc, countdown=60, max_retries=3)
```

### Background Processing

```python
class BackgroundTaskManager:
    """
    Sophisticated background task management:
    - Priority queues
    - Resource monitoring
    - Task scheduling
    - Error recovery
    """
    
    @staticmethod
    @celery_app.task
    def index_document_batch(documents, collection_name):
        """
        Batch document indexing task:
        - Parallel processing
        - Progress tracking
        - Error handling
        """
        
        results = []
        for doc_batch in chunk_list(documents, batch_size=50):
            try:
                # Process batch
                batch_result = process_document_batch(doc_batch, collection_name)
                results.append(batch_result)
                
                # Update progress
                progress = len(results) / len(documents) * 100
                index_document_batch.update_state(
                    state='PROGRESS',
                    meta={'current': len(results), 'total': len(documents)}
                )
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Continue với next batch
                continue
                
        return {"processed": len(results), "total": len(documents)}
```

## 🛡️ Error Handling và Monitoring

### Comprehensive Error Handling

```python
class LegalChatbotError(Exception):
    """Base exception class for legal chatbot"""
    pass

class EmbeddingServiceError(LegalChatbotError):
    """Embedding service related errors"""
    pass

class VectorStoreError(LegalChatbotError):
    """Vector database related errors"""
    pass

class LLMServiceError(LegalChatbotError):
    """LLM service related errors"""
    pass

class ErrorHandler:
    """Centralized error handling và recovery strategies"""
    
    @staticmethod
    def handle_embedding_error(func):
        """Decorator cho embedding service error handling"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.RequestException as e:
                logger.error(f"Embedding service error: {e}")
                # Fallback to cached embeddings hoặc alternative service
                return ErrorHandler._fallback_embedding(*args, **kwargs)
            except Exception as e:
                logger.error(f"Unexpected embedding error: {e}")
                raise EmbeddingServiceError(f"Embedding generation failed: {e}")
        return wrapper
    
    @staticmethod
    def handle_llm_error(func):
        """Decorator cho LLM service error handling"""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                # Retry với exponential backoff
                return ErrorHandler._retry_llm_call(func, *args, **kwargs)
            except Exception as e:
                logger.error(f"Unexpected LLM error: {e}")
                raise LLMServiceError(f"LLM processing failed: {e}")
        return wrapper
```

### Performance Monitoring

```python
class PerformanceMonitor:
    """
    Comprehensive performance monitoring:
    - Response time tracking
    - Resource usage monitoring  
    - Error rate tracking
    - Quality metrics
    """
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "embedding_times": [],
            "search_times": [],
            "llm_times": [],
            "error_counts": defaultdict(int),
            "request_counts": defaultdict(int)
        }
    
    def track_request(self, endpoint, response_time, status_code):
        """Track API request metrics"""
        self.metrics["response_times"].append(response_time)
        self.metrics["request_counts"][endpoint] += 1
        
        if status_code >= 400:
            self.metrics["error_counts"][endpoint] += 1
            
    def get_performance_report(self):
        """Generate performance report"""
        return {
            "avg_response_time": np.mean(self.metrics["response_times"]),
            "p95_response_time": np.percentile(self.metrics["response_times"], 95),
            "error_rate": sum(self.metrics["error_counts"].values()) / 
                         sum(self.metrics["request_counts"].values()),
            "total_requests": sum(self.metrics["request_counts"].values())
        }
```

## 🚀 Deployment và Scaling

### Docker Configuration

```dockerfile
# Dockerfile - Production-ready container
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ /app/src/
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Configuration

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql://user:pass@postgres:5432/legal_db
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - postgres
      - redis
      - qdrant
    restart: unless-stopped
    
  celery-worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    environment:
      - CELERY_BROKER_URL=redis://redis:6379
      - CELERY_RESULT_BACKEND=redis://redis:6379
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=legal_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    restart: unless-stopped
    
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  postgres_data:
  qdrant_data:
```

### Scaling Strategies

```python
class AutoScaler:
    """
    Auto-scaling logic based on metrics:
    - CPU/Memory usage
    - Request queue length
    - Response time thresholds
    """
    
    def __init__(self):
        self.min_workers = 2
        self.max_workers = 10
        self.target_cpu_percent = 70
        self.target_response_time = 2.0  # seconds
        
    def should_scale_up(self, metrics):
        """Determine if scaling up is needed"""
        return (
            metrics["cpu_percent"] > self.target_cpu_percent or
            metrics["avg_response_time"] > self.target_response_time or
            metrics["queue_length"] > 100
        )
        
    def should_scale_down(self, metrics):
        """Determine if scaling down is needed"""
        return (
            metrics["cpu_percent"] < 30 and
            metrics["avg_response_time"] < 1.0 and
            metrics["queue_length"] < 10
        )
```

## 📈 Performance Optimization

### Caching Strategy

```python
class CacheManager:
    """
    Multi-layer caching strategy:
    - Redis for session data
    - In-memory for embeddings
    - Database query caching
    """
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(Config.REDIS_URL)
        self.memory_cache = {}
        self.ttl_default = 3600  # 1 hour
        
    def get_or_compute_embedding(self, text):
        """Cache-first embedding retrieval"""
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Check Redis first
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
            
        # Compute if not cached
        embedding = get_custom_embedding(text)
        
        # Cache for future use
        self.redis_client.setex(
            cache_key, 
            self.ttl_default, 
            json.dumps(embedding)
        )
        
        return embedding
```

### Database Optimization

```python
class DatabaseOptimizer:
    """
    Database performance optimization:
    - Connection pooling
    - Query optimization
    - Index management
    """
    
    def __init__(self):
        self.connection_pool = create_engine(
            Config.POSTGRES_URL,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
    def optimize_vector_queries(self):
        """Optimize vector similarity queries"""
        # Create appropriate indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING hnsw(embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);",
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);"
        ]
        
        for index_sql in indexes:
            self.connection_pool.execute(text(index_sql))
```

## ✅ Testing Strategy

### Unit Testing

```python
class TestLegalAgent:
    """Comprehensive test suite for legal agent functionality"""
    
    def test_contract_penalty_calculation(self):
        """Test contract penalty calculation accuracy"""
        result = contract_penalty_calculator(
            contract_value=100_000_000,  # 100 million VND
            penalty_rate=0.1,            # 0.1% per day
            days_late=30                 # 30 days late
        )
        
        expected_penalty = 100_000_000 * 0.001 * 30  # 3,000,000 VND
        assert result["penalty_amount"] == expected_penalty
        
    def test_vietnamese_intent_detection(self):
        """Test Vietnamese follow-up question detection"""
        history = [
            {"role": "user", "content": "Thời gian làm việc tối đa là bao nhiêu?"},
            {"role": "assistant", "content": "Theo Bộ luật Lao động, thời gian làm việc không quá 8 giờ/ngày..."}
        ]
        
        follow_up = "Còn thời gian nghỉ giữa giờ thì sao?"
        result = detect_user_intent(history, follow_up)
        
        assert "thời gian nghỉ giữa giờ" in result
        assert "Bộ luật Lao động" in result  # Context should be preserved
```

### Integration Testing

```python
class TestSystemIntegration:
    """End-to-end system testing"""
    
    def test_chat_completion_flow(self):
        """Test complete chat flow from API to response"""
        client = TestClient(app)
        
        # Test data
        request_data = {
            "user_id": "test_user",
            "user_message": "Tôi muốn hỏi về thủ tục thành lập công ty",
            "sync_request": True
        }
        
        # API call
        response = client.post("/chat/complete", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        assert "response" in response.json()
        assert len(response.json()["response"]) > 0
```

---

*Tài liệu này cung cấp comprehensive technical deep dive vào Vietnamese Legal Chatbot Backend system, covering architecture, implementation details, và best practices cho production deployment.*