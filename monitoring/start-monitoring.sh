#!/bin/bash

# 🔍 Script khởi chạy monitoring stack đơn giản
# Chạy: ./start-monitoring.sh

echo "🚀 Starting Legal Chatbot Monitoring Stack..."
echo "=============================================="

# Kiểm tra Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker không được cài đặt!"
    echo "💡 Cài đặt Docker trước: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose không được cài đặt!"
    echo "💡 Cài đặt Docker Compose trước: https://docs.docker.com/compose/install/"
    exit 1
fi

# Tạo network nếu chưa có
echo "🌐 Creating monitoring network..."
docker network create monitoring 2>/dev/null || true

# Khởi chạy monitoring stack
echo "🔥 Starting monitoring services..."
docker compose up -d

# Kiểm tra trạng thái
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Kiểm tra services
echo ""
echo "📊 Service Status:"
echo "=================="

if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus: http://localhost:9090"
else
    echo "❌ Prometheus: Failed to start"
fi

if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana: http://localhost:3000 (admin/admin123)"
else
    echo "❌ Grafana: Failed to start"
fi

if curl -s http://localhost:9100/metrics > /dev/null; then
    echo "✅ Node Exporter: http://localhost:9100"
else
    echo "❌ Node Exporter: Failed to start"
fi

if curl -s http://localhost:8080/containers/ > /dev/null; then
    echo "✅ cAdvisor: http://localhost:8080"
else
    echo "❌ cAdvisor: Failed to start"
fi

echo ""
echo "🎉 Monitoring Setup Complete!"
echo "============================="
echo ""
echo "📊 Access URLs:"
echo "  Grafana Dashboard: http://localhost:3000"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "  Prometheus: http://localhost:9090"
echo "  Node Exporter: http://localhost:9100" 
echo "  cAdvisor: http://localhost:8080"
echo ""
echo "🛑 To stop: docker compose down"
echo "📋 To see logs: docker compose logs -f"