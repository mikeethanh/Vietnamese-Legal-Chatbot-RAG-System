# 🔍 Monitoring Stack Đơn Giản

## Tổng Quan

Monitoring stack đơn giản cho Legal Chatbot RAG System sử dụng:
- **Prometheus**: Thu thập metrics
- **Grafana**: Hiển thị dashboard
- **Node Exporter**: System metrics 
- **cAdvisor**: Container metrics

## 🚀 Khởi Chạy Nhanh

### 1. Khởi động monitoring
```bash
cd monitoring
./start-monitoring.sh
```

### 2. Truy cập dashboard
- **Grafana**: http://localhost:3000
  - Username: `admin`
  - Password: `admin123`
- **Prometheus**: http://localhost:9090

### 3. Dừng monitoring
```bash
./stop-monitoring.sh
```

## 📊 Dashboard

### Legal Chatbot System Monitor
- 🔥 CPU Usage
- 💾 Memory Usage  
- 🚀 Backend API Status
- 🖥️ Frontend Status
- 💽 Disk Usage

## 📋 Ports

| Service | Port | URL |
|---------|------|-----|
| Grafana | 3000 | http://localhost:3000 |
| Prometheus | 9090 | http://localhost:9090 |
| Node Exporter | 9100 | http://localhost:9100 |
| cAdvisor | 8080 | http://localhost:8080 |

## 🚨 Alerts

Các alerts được cấu hình:
- **HighCpuUsage**: CPU > 80% trong 5 phút
- **HighMemoryUsage**: Memory > 85% trong 5 phút  
- **DiskSpaceLow**: Disk > 90% trong 5 phút
- **BackendAPIDown**: Backend API không phản hồi
- **ContainerRestartHigh**: Container restart thường xuyên

## 🔧 Cấu Hình

### Thêm Backend Metrics

Thêm vào backend API (`app.py`):
```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_duration_seconds', 'Request latency')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_LATENCY.observe(time.time() - start_time)
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Thêm Custom Metrics

Chỉnh sửa `prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'my-custom-app'
    static_configs:
      - targets: ['host.docker.internal:8080']
    scrape_interval: 15s
```

## 🛠️ Commands Hữu Ích

```bash
# Xem logs
docker-compose logs -f

# Xem status containers
docker-compose ps

# Restart service
docker-compose restart grafana

# Rebuild và restart
docker-compose up -d --build

# Xóa volumes (reset data)
docker-compose down -v
```

## 📈 Monitoring Best Practices

1. **CPU & Memory**: Theo dõi usage patterns
2. **Disk Space**: Set alerts cho 85-90%
3. **API Response Time**: Monitor latency trends
4. **Error Rates**: Track 4xx/5xx responses
5. **Container Health**: Monitor restarts và uptime

## 🎯 Metrics Quan Trọng

### System Metrics
- `node_cpu_seconds_total`
- `node_memory_MemAvailable_bytes`
- `node_filesystem_avail_bytes`

### Container Metrics  
- `container_cpu_usage_seconds_total`
- `container_memory_usage_bytes`
- `container_start_time_seconds`

### Application Metrics
- `requests_total` 
- `request_duration_seconds`
- `up{job="legal-chatbot-backend"}`

## 🔧 Troubleshooting

### Prometheus không thu thập được metrics
```bash
# Kiểm tra config
docker-compose exec prometheus promtool check config /etc/prometheus/prometheus.yml

# Reload config
curl -X POST http://localhost:9090/-/reload
```

### Grafana không hiển thị data
1. Kiểm tra datasource connection
2. Verify query syntax trong Prometheus
3. Check time range selection

### Services không start
```bash
# Kiểm tra logs
docker-compose logs service_name

# Kiểm tra ports
netstat -tulpn | grep :3000
```

## 🎉 Kết Luận

Monitoring stack đơn giản nhưng hiệu quả để theo dõi:
- ✅ System health (CPU, Memory, Disk)
- ✅ Application status (Backend, Frontend)
- ✅ Container metrics 
- ✅ Custom business metrics

**Happy monitoring!** 📊