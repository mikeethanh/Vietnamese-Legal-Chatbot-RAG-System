# Testing & CI/CD Guide

## Overview

Hệ thống test và CI/CD cho Vietnamese Legal Chatbot RAG System được thiết kế để đảm bảo chất lượng code và tính ổn định của hệ thống.

## Cấu trúc Testing

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration và fixtures
├── test_utils.py           # Utilities cho testing
├── test_api.py             # Test FastAPI endpoints
├── test_backend_utils.py   # Test backend utilities
├── test_brain.py           # Test core query processing
├── test_integration.py     # Integration tests
└── test_performance.py     # Performance tests
```

## Loại Tests

### 1. Unit Tests
- Test các function riêng lẻ
- Mock tất cả external dependencies
- Chạy nhanh và độc lập

### 2. Integration Tests
- Test sự tương tác giữa các components
- Test với database và external services
- Đánh dấu với marker `@pytest.mark.integration`

### 3. Performance Tests
- Test hiệu suất xử lý queries
- Test concurrent processing
- Test memory usage
- Đánh dấu với marker `@pytest.mark.performance`

### 4. API Tests
- Test FastAPI endpoints
- Test request/response validation
- Test error handling

## Chạy Tests

### Sử dụng Makefile (Khuyến nghị)

```bash
# Chạy tất cả tests
make test

# Chạy unit tests
make test-unit

# Chạy integration tests
make test-integration

# Chạy performance tests
make test-performance

# Chạy Docker tests
make test-docker
```

### Sử dụng Script

```bash
# Chạy tất cả tests
./scripts/run_tests.sh all

# Chạy specific test type
./scripts/run_tests.sh unit
./scripts/run_tests.sh integration
./scripts/run_tests.sh performance
```

### Sử dụng Pytest trực tiếp

```bash
# Chạy tất cả tests với coverage
pytest tests/ --cov=backend/src --cov-report=html

# Chạy tests theo marker
pytest -m "unit"
pytest -m "integration"
pytest -m "performance"

# Chạy specific test file
pytest tests/test_api.py -v

# Chạy specific test function
pytest tests/test_api.py::TestFastAPIApp::test_health_check -v
```

## Code Quality

### Pre-commit Hooks

Cài đặt pre-commit hooks:

```bash
make install-dev
# hoặc
pre-commit install
```

Chạy pre-commit checks:

```bash
make pre-commit
# hoặc
pre-commit run --all-files
```

### Linting và Formatting

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security check
make security
```

### Tất cả checks cùng lúc

```bash
make check-all
```

## CI/CD Pipeline

### GitHub Actions Workflow

Pipeline được định nghĩa trong `.github/workflows/ci.yml` với các job:

1. **Code Quality & Security**
   - Pre-commit hooks
   - Type checking với mypy
   - Security scanning với bandit
   - Linting với flake8

2. **Backend Testing**
   - Unit tests với pytest
   - Integration tests
   - Code coverage
   - Matrix testing với Python 3.11, 3.12

3. **Frontend Testing**
   - Syntax check cho Streamlit app

4. **Docker Build & Test**
   - Build Docker images
   - Test container functionality
   - Health checks

5. **Security Scanning**
   - Trivy vulnerability scanner
   - SARIF report upload

6. **Performance Testing**
   - Chỉ chạy trên main branch
   - Load testing với Locust

7. **Deploy Staging**
   - Tự động deploy staging khi push vào main

### Trigger Events

- **Pull Request**: Chạy tất cả tests
- **Push to main/develop**: Full pipeline
- **Manual**: Workflow dispatch

## Environment Variables

### For Testing

```bash
export OPENAI_API_KEY="test-key"
export REDIS_URL="redis://localhost:6379"
export DATABASE_URL="mysql://test:test@localhost/test"
export QDRANT_URL="http://localhost:6333"
export DEBUG="true"
```

### For CI/CD

Các secrets cần thiết trong GitHub:

- `OPENAI_API_KEY`
- `DOCKER_HUB_USERNAME`
- `DOCKER_HUB_PASSWORD`
- `STAGING_DEPLOY_KEY`

## Configuration Files

- `pytest.ini`: Pytest configuration
- `.pre-commit-config.yaml`: Pre-commit hooks
- `mypy.ini`: Type checking configuration
- `requirements_dev.txt`: Development dependencies

## Coverage Reports

Coverage reports được tạo trong:
- `htmlcov/index.html`: HTML report
- `coverage.xml`: XML report cho CI/CD

## Performance Monitoring

Performance tests kiểm tra:
- Query processing time (< 2 giây)
- Concurrent processing (10 queries < 5 giây)
- Memory usage (< 100MB increase)

## Best Practices

1. **Viết tests trước khi implement feature**
2. **Maintain coverage > 70%**
3. **Mock external dependencies trong unit tests**
4. **Sử dụng descriptive test names**
5. **Group related tests trong classes**
6. **Test cả success và error cases**
7. **Chạy tests locally trước khi push**

## Troubleshooting

### Common Issues

1. **Tests fail với import errors**
   ```bash
   pip install -r requirements_dev.txt
   pip install -r backend/requirements.txt
   ```

2. **Pre-commit hooks fail**
   ```bash
   pre-commit clean
   pre-commit install
   ```

3. **Coverage too low**
   - Thêm tests cho uncovered code
   - Xem HTML report để identify missing coverage

4. **Docker tests fail**
   ```bash
   docker system prune -f
   ./scripts/test_docker.sh
   ```

## Contributing

Khi contribute code mới:

1. Viết tests cho features mới
2. Ensure all tests pass: `make test`
3. Check code quality: `make check-all`
4. Update documentation nếu cần

## Monitoring và Alerts

- Coverage reports uploaded to Codecov
- Security alerts từ GitHub Security tab
- Performance metrics tracking
- Failed build notifications

## Liên hệ

Nếu có vấn đề với testing hoặc CI/CD, tạo issue hoặc liên hệ team.