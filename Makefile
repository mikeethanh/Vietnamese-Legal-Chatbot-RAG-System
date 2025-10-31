.PHONY: help install test test-unit test-integration test-performance test-docker clean lint format security check-all

# Default target
help:
	@echo "Vietnamese Legal Chatbot Development Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup:"
	@echo "  install          Install all dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  test-docker      Test Docker builds and containers"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  security         Run security checks"
	@echo "  type-check       Run type checking with mypy"
	@echo ""
	@echo "Comprehensive:"
	@echo "  check-all        Run all quality checks and tests"
	@echo "  clean            Clean up generated files"

# Installation
install:
	pip install -r requirements_dev.txt
	pip install -r backend/requirements.txt

install-dev:
	pip install -r requirements_dev.txt
	pre-commit install

# Testing targets
test:
	./scripts/run_tests.sh all

test-unit:
	./scripts/run_tests.sh unit

test-integration:
	./scripts/run_tests.sh integration

test-performance:
	./scripts/run_tests.sh performance

test-docker:
	./scripts/test_docker.sh

# Code quality
lint:
	flake8 backend/src tests/
	pydocstyle backend/src --convention=google

format:
	black backend/src tests/
	isort backend/src tests/

security:
	bandit -r backend/src -f json -o bandit-report.json
	@echo "Security report generated: bandit-report.json"

type-check:
	mypy backend/src --config-file mypy.ini

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Comprehensive check
check-all: format lint type-check security test

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf pytest-results.xml
	rm -rf bandit-report.json
	rm -rf dist/
	rm -rf build/

# Development server
run-backend:
	cd backend && uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

run-frontend:
	cd frontend && streamlit run chat_interface.py --server.port 8501

# Docker commands
docker-build-backend:
	cd backend && docker build -t legal-chatbot-backend .

docker-build-frontend:
	cd frontend && docker build -t legal-chatbot-frontend .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database
db-migrate:
	cd backend && python src/database.py migrate

db-seed:
	cd backend && python src/import_data.py --data-file data/train_qa_format.jsonl

# Monitoring
coverage-report:
	pytest --cov=backend/src --cov-report=html
	@echo "Coverage report generated: htmlcov/index.html"

performance-report:
	pytest tests/test_performance.py --benchmark-only --benchmark-sort=mean

# CI/CD simulation
ci-check:
	@echo "Running CI checks locally..."
	make format
	make lint
	make type-check
	make security
	make test-unit
	@echo "CI checks completed successfully!"