# 📡 Legal Embedding API - Hướng dẫn sử dụng

API endpoint để generate embeddings và tính similarity cho văn bản pháp luật tiếng Việt.

## 🌐 Base URL

```
http://YOUR_DROPLET_IP:5000
```

## 📋 Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Mô tả:** Kiểm tra trạng thái API và model

**Request:**
```bash
curl http://YOUR_DROPLET_IP:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "embedding_dim": 1024,
  "timestamp": 1761541371.844
}
```

---

### 2. Generate Embeddings

**Endpoint:** `POST /embed`

**Mô tả:** Chuyển đổi text thành embedding vectors (1024 dimensions)

**Request:**
```bash
curl -X POST http://YOUR_DROPLET_IP:5000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Luật Dân sự năm 2015 quy định về quyền sở hữu",
      "Bộ luật Hình sự năm 2017"
    ]
  }'
```

**Parameters:**
- `texts` (array of strings, required): Danh sách các text cần embedding
- `batch_size` (int, optional): Batch size cho processing (default: 32)
- `normalize` (bool, optional): Normalize embeddings (default: true)

**Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, 0.789, ..., 0.321],  // 1024 dimensions
    [0.234, -0.567, 0.890, ..., 0.432]   // 1024 dimensions
  ],
  "processing_time": 0.123,
  "count": 2,
  "embedding_dim": 1024
}
```

**Error Response:**
```json
{
  "error": "texts field is required",
  "status": "error"
}
```

---

### 3. Calculate Similarity

**Endpoint:** `POST /similarity`

**Mô tả:** Tính cosine similarity giữa 2 tập text

**Request:**
```bash
curl -X POST http://YOUR_DROPLET_IP:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts1": ["Quyền sở hữu tài sản trong Luật Dân sự"],
    "texts2": [
      "Quy định về tài sản chung",
      "Luật Hình sự về tội phạm",
      "Luật Đất đai năm 2013"
    ]
  }'
```

**Parameters:**
- `texts1` (array of strings, required): Tập text thứ nhất
- `texts2` (array of strings, required): Tập text thứ hai

**Response:**
```json
{
  "similarities": [
    [0.85, 0.23, 0.67]  // similarity matrix: [len(texts1), len(texts2)]
  ],
  "processing_time": 0.089,
  "shape": [1, 3]
}
```

**Giải thích:**
- `similarities[i][j]` = cosine similarity giữa `texts1[i]` và `texts2[j]`
- Giá trị từ -1 đến 1 (càng gần 1 càng giống nhau)

---

## 💻 Code Examples

### Python

**1. Basic Usage:**
```python
import requests

API_URL = "http://YOUR_DROPLET_IP:5000"

# Generate embeddings
response = requests.post(
    f"{API_URL}/embed",
    json={"texts": ["Luật Dân sự", "Bộ luật Hình sự"]}
)
data = response.json()
embeddings = data["embeddings"]
print(f"Got {len(embeddings)} embeddings of dimension {data['embedding_dim']}")
```

**2. Semantic Search:**
```python
import requests
import numpy as np

API_URL = "http://YOUR_DROPLET_IP:5000"

def semantic_search(query: str, documents: list[str], top_k: int = 5):
    """
    Tìm top-k documents giống nhất với query
    """
    # Calculate similarities
    response = requests.post(
        f"{API_URL}/similarity",
        json={"texts1": [query], "texts2": documents}
    )
    
    similarities = np.array(response.json()["similarities"][0])
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [
        {
            "document": documents[idx],
            "score": float(similarities[idx]),
            "rank": i + 1
        }
        for i, idx in enumerate(top_indices)
    ]
    
    return results

# Example usage
query = "Quy định về quyền sở hữu tài sản"
documents = [
    "Luật Dân sự 2015 quy định về quyền sở hữu",
    "Bộ luật Hình sự về tội phạm tài sản",
    "Luật Đất đai 2013 về quyền sử dụng đất",
    "Luật Nhà ở về quyền sở hữu nhà",
]

results = semantic_search(query, documents, top_k=3)

for result in results:
    print(f"{result['rank']}. {result['document']}")
    print(f"   Score: {result['score']:.3f}\n")
```

**3. Clustering Documents:**
```python
import requests
import numpy as np
from sklearn.cluster import KMeans

API_URL = "http://YOUR_DROPLET_IP:5000"

def cluster_documents(documents: list[str], n_clusters: int = 3):
    """
    Gom nhóm documents dựa trên embeddings
    """
    # Get embeddings
    response = requests.post(
        f"{API_URL}/embed",
        json={"texts": documents}
    )
    embeddings = np.array(response.json()["embeddings"])
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Group by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(documents[i])
    
    return clusters

# Example
documents = [
    "Luật Dân sự về quyền sở hữu",
    "Luật Hình sự về tội giết người",
    "Luật Đất đai về quyền sử dụng đất",
    "Luật Hình sự về tội trộm cắp",
    "Luật Dân sự về hợp đồng",
]

clusters = cluster_documents(documents, n_clusters=2)

for cluster_id, docs in clusters.items():
    print(f"Cluster {cluster_id}:")
    for doc in docs:
        print(f"  - {doc}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

const API_URL = 'http://YOUR_DROPLET_IP:5000';

// Generate embeddings
async function getEmbeddings(texts) {
  try {
    const response = await axios.post(`${API_URL}/embed`, {
      texts: texts
    });
    return response.data.embeddings;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Calculate similarity
async function getSimilarity(texts1, texts2) {
  try {
    const response = await axios.post(`${API_URL}/similarity`, {
      texts1: texts1,
      texts2: texts2
    });
    return response.data.similarities;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Example usage
(async () => {
  // Test 1: Embeddings
  const embeddings = await getEmbeddings([
    'Luật Dân sự năm 2015',
    'Bộ luật Hình sự'
  ]);
  console.log(`Got ${embeddings.length} embeddings`);

  // Test 2: Similarity
  const similarities = await getSimilarity(
    ['Quyền sở hữu tài sản'],
    ['Tài sản chung', 'Tài sản riêng', 'Quyền kế thừa']
  );
  console.log('Similarities:', similarities);
})();
```

### cURL

```bash
# Health check
curl http://YOUR_DROPLET_IP:5000/health

# Embeddings
curl -X POST http://YOUR_DROPLET_IP:5000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Luật Dân sự", "Luật Hình sự"]}'

# Similarity
curl -X POST http://YOUR_DROPLET_IP:5000/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "texts1": ["Quyền sở hữu"],
    "texts2": ["Tài sản", "Quyền kế thừa"]
  }'
```

---

## ⚡ Performance Tips

1. **Batch Processing**: Gửi nhiều texts cùng lúc thay vì gọi API nhiều lần
   ```python
   # ❌ Slow
   for text in texts:
       embedding = get_embedding([text])
   
   # ✅ Fast
   embeddings = get_embeddings(texts)
   ```

2. **Caching**: Cache embeddings cho documents không thay đổi

3. **Use similarity endpoint**: Nhanh hơn là tính embeddings rồi tính similarity

4. **Adjust batch_size**: Tăng nếu server có nhiều RAM

---

## 🚨 Error Handling

**Common Errors:**

1. **Connection Refused**
   - Kiểm tra firewall: `ufw status`
   - Kiểm tra API đang chạy: `docker ps`

2. **Model Not Loaded**
   - Check logs: `docker logs legal-embedding-api`
   - Restart: `docker restart legal-embedding-api`

3. **Out of Memory**
   - Giảm batch_size trong request
   - Upgrade droplet RAM

---

## 📊 Rate Limits

**Current Configuration:**
- No hard rate limits
- Recommended: < 100 requests/second
- Consider adding rate limiting nếu public API

**To add rate limiting:**
```python
# In serve_model.py
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per minute"]
)
```

---

## 🔒 Security

**Recommendations:**

1. **API Key Authentication:**
```python
# Add to serve_model.py
API_KEY = os.getenv('API_KEY')

@app.before_request
def verify_api_key():
    if request.path == '/health':
        return
    
    key = request.headers.get('X-API-Key')
    if key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
```

2. **IP Whitelist:**
```bash
# On droplet
ufw delete allow 5000/tcp
ufw allow from YOUR_BACKEND_IP to any port 5000
```

3. **HTTPS/SSL:**
- Use reverse proxy (nginx) với SSL certificate
- Consider using Cloudflare for DDoS protection

---

## 📞 Support

Issues? Check:
1. [GPU_CPU_DEPLOYMENT_GUIDE.md](./GPU_CPU_DEPLOYMENT_GUIDE.md)
2. Docker logs: `docker logs legal-embedding-api`
3. API health: `curl http://YOUR_DROPLET_IP:5000/health`
