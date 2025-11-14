#!/bin/bash

# 🛑 Script dừng monitoring stack
# Chạy: ./stop-monitoring.sh

echo "🛑 Stopping Legal Chatbot Monitoring Stack..."
echo "============================================="

# Dừng tất cả containers
echo "🔥 Stopping all monitoring services..."
docker compose down

echo ""
echo "🧹 Cleaning up..."

# Xóa network (tuỳ chọn)
# docker network rm monitoring 2>/dev/null || true

echo ""
echo "✅ Monitoring stack stopped!"
echo "=========================="
echo ""
echo "💡 Tips:"
echo "  - Data volumes vẫn được giữ lại"
echo "  - Để khởi động lại: ./start-monitoring.sh"
echo "  - Để xóa data: docker compose down -v"