.PHONY: help install test lint format security docker-build ci-check ci-local clean

# Default target
help:
	@echo "🚀 Vietnamese Legal Chatbot - Simple Commands"
	@echo "=============================================="
	@echo ""
	@echo "📦 Setup:"
	@echo "  install          Install all dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  test             Run tests"
	@echo "  lint             Run linting (flake8)"
	@echo "  format           Format code (black)"
	@echo "  security         Security check (bandit)"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo ""
	@echo "📊 Monitoring:"
	@echo "  monitoring-start Start monitoring (Grafana + Prometheus)"
	@echo "  monitoring-stop  Stop monitoring"
	@echo ""
	@echo "🚀 CI/CD:"
	@echo "  ci-check         Run full CI check locally"
	@echo "  ci-local         Quick local CI check"
	@echo "  clean            Clean up generated files"

# 📦 Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements_dev.txt
	if [ -f backend/requirements.txt ]; then pip install -r backend/requirements.txt; fi

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -r requirements_dev.txt
	pre-commit install

# 🧪 Testing
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --tb=short

# 🧹 Code Quality  
lint:
	@echo "🧹 Running linter..."
	flake8 backend/src/ || true

format:
	@echo "✨ Formatting code..."
	black backend/src/ frontend/ || true

security:
	@echo "🔒 Running security check..."
	bandit -r backend/src/ || true

# 🐳 Docker
docker-build:
	@echo "🐳 Building Docker images..."
	cd backend && docker build -t legal-chatbot-backend .
	cd frontend && docker build -t legal-chatbot-frontend .

# 🚀 CI/CD Commands
ci-local:
	@echo "🚀 Running quick local CI check..."
	./scripts/check-ci-local.sh

ci-check: format lint test security
	@echo "✅ Full CI check completed!"

clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf htmlcov/ .pytest_cache/ .coverage coverage.xml
	@echo "✅ Cleanup completed!"

# 🚀 Development Server
run-backend:
	@echo "🚀 Starting backend server..."
	cd backend && uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	@echo "🖥️ Starting frontend server..."
	cd frontend && streamlit run chat_interface.py --server.port 8501

# 🐳 Docker Compose
docker-up:
	@echo "🐳 Starting all services..."
	docker-compose up -d

docker-down:
	@echo "🛑 Stopping all services..."
	docker-compose down

docker-logs:
	@echo "📋 Showing logs..."
	docker-compose logs -f

# 📊 Monitoring
monitoring-start:
	@echo "📊 Starting monitoring stack..."
	cd monitoring && ./start-monitoring.sh

monitoring-stop:
	@echo "🛑 Stopping monitoring stack..."
	cd monitoring && ./stop-monitoring.sh

monitoring-logs:
	@echo "📋 Showing monitoring logs..."
	cd monitoring && docker-compose logs -f
