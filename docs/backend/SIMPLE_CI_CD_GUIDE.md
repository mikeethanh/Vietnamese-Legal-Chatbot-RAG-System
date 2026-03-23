# Hướng Dẫn CI/CD Đơn Giản - Vietnamese Legal Chatbot

## Tổng Quan
Tài liệu này hướng dẫn setup **CI/CD (Continuous Integration/Continuous Deployment)** đơn giản cho dự án Vietnamese Legal Chatbot sử dụng **GitHub Actions**.

---

## 1. CI/CD Là Gì?

### Continuous Integration (CI)
**Tích hợp liên tục** - Tự động kiểm tra code mỗi khi có thay đổi:
- ✅ Chạy tests
- ✅ Kiểm tra code quality (linting)
- ✅ Kiểm tra type hints
- ✅ Build Docker images

### Continuous Deployment (CD)
**Triển khai liên tục** - Tự động deploy khi code pass tests:
- ✅ Build Docker images
- ✅ Push lên Docker Hub/Registry
- ✅ Deploy lên server (production/staging)

### Lợi Ích

| Không có CI/CD | Có CI/CD |
|----------------|----------|
| Manual testing mỗi lần commit | ✅ Tự động test |
| Phát hiện bugs muộn | ✅ Phát hiện bugs sớm |
| Deploy manual, dễ lỗi | ✅ Deploy tự động, consistent |
| Mất nhiều thời gian | ✅ Tiết kiệm thời gian |

---

## 2. GitHub Actions - Công Cụ CI/CD

### GitHub Actions là gì?
- CI/CD platform miễn phí của GitHub
- Chạy workflows tự động khi có events (push, pull request, etc.)
- 2000 minutes/month miễn phí cho repos public

### Workflow File Structure

```
.github/
└── workflows/
    ├── ci.yml           # CI workflow (test + lint)
    ├── docker-build.yml # Build Docker images
    └── deploy.yml       # Deploy to production
```

---

## 3. Setup CI Workflow - Kiểm Tra Code Tự Động

### Bước 1: Tạo file `.github/workflows/ci.yml`

```yaml
name: CI - Test and Lint

# Khi nào chạy workflow này?
on:
  push:
    branches: [ main, develop ]  # Khi push lên main hoặc develop
  pull_request:
    branches: [ main ]           # Khi tạo pull request vào main

# Jobs (công việc) cần thực hiện
jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest  # Chạy trên Ubuntu server
    
    steps:
    # Step 1: Checkout code
    - name: Checkout code
      uses: actions/checkout@v3
    
    # Step 2: Setup Python
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    # Step 3: Cache dependencies (tăng tốc)
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
    # Step 4: Install dependencies
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install -r requirements_dev.txt
    
    # Step 5: Run tests
    - name: Run pytest
      run: |
        cd backend
        pytest tests/ -v --tb=short
      env:
        # Set environment variables nếu cần
        POSTGRES_HOST: localhost
        REDIS_HOST: localhost
    
    # Step 6: Upload test results (nếu fail)
    - name: Upload test results
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: test-results
        path: backend/pytest-report.xml

  lint:
    name: Code Quality Check
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install linting tools
      run: |
        pip install black flake8 mypy
    
    # Check code formatting với Black
    - name: Check code formatting (Black)
      run: |
        cd backend/src
        black --check .
    
    # Check code style với Flake8
    - name: Lint with Flake8
      run: |
        cd backend/src
        flake8 . --max-line-length=100 --exclude=__pycache__
    
    # Check type hints với mypy
    - name: Type check with mypy
      run: |
        cd backend/src
        mypy . --ignore-missing-imports
      continue-on-error: true  # Không fail nếu mypy có lỗi
```

### Giải Thích Chi Tiết

#### `on:` - Triggers

```yaml
on:
  push:
    branches: [ main, develop ]
```

**Ý nghĩa**: Chạy workflow khi:
- Push code lên branch `main` hoặc `develop`
- Tạo Pull Request vào branch `main`

**Các triggers khác**:
```yaml
on:
  push:                    # Mỗi khi push
  pull_request:           # Mỗi khi tạo PR
  schedule:               # Chạy định kỳ
    - cron: '0 0 * * *'   # Mỗi ngày 00:00
  workflow_dispatch:      # Chạy manual từ UI
```

#### `jobs:` - Công Việc

```yaml
jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
```

- **`test`**: Job ID (unique)
- **`name`**: Tên hiển thị trên UI
- **`runs-on`**: OS để chạy (ubuntu-latest, windows-latest, macos-latest)

#### `steps:` - Các Bước

**Step 1: Checkout code**
```yaml
- uses: actions/checkout@v3
```
Clone repo về runner

**Step 2: Setup Python**
```yaml
- uses: actions/setup-python@v4
  with:
    python-version: '3.11'
```
Cài Python 3.11

**Step 3: Cache dependencies**
```yaml
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
```
Cache pip packages → chạy nhanh hơn (từ 3 phút xuống 30 giây)

**Step 4-5: Install và Run tests**
```yaml
- name: Run pytest
  run: |
    cd backend
    pytest tests/ -v
```

---

## 4. Setup Docker Build Workflow

### File `.github/workflows/docker-build.yml`

```yaml
name: Build Docker Images

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'  # Chạy khi tag version (v1.0.0, v1.1.0)

jobs:
  build-backend:
    name: Build Backend Image
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    # Login vào Docker Hub
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    # Setup Docker Buildx (build nhanh hơn)
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    # Build và push image
    - name: Build and push Backend image
      uses: docker/build-push-action@v4
      with:
        context: ./backend
        file: ./backend/Dockerfile
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-backend:latest
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-backend:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    # Thông báo thành công
    - name: Image build successful
      run: |
        echo "✅ Backend image built successfully"
        echo "Image: ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-backend:latest"

  build-frontend:
    name: Build Frontend Image
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build and push Frontend image
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        file: ./frontend/Dockerfile
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-frontend:latest
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-frontend:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

### Setup Docker Hub Secrets

**Bước 1: Tạo Access Token trên Docker Hub**
1. Đăng nhập https://hub.docker.com
2. Account Settings → Security → New Access Token
3. Copy token

**Bước 2: Thêm Secrets vào GitHub**
1. Vào repo → Settings → Secrets and variables → Actions
2. New repository secret:
   - Name: `DOCKER_USERNAME`, Value: username Docker Hub
   - Name: `DOCKER_PASSWORD`, Value: access token vừa tạo

**Bước 3: Test workflow**
```bash
git add .
git commit -m "Add Docker build workflow"
git push origin main
```

→ Vào tab "Actions" trên GitHub để xem kết quả

---

## 5. Setup Deployment Workflow (Optional)

### File `.github/workflows/deploy.yml`

```yaml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'  # Chỉ deploy khi tag version

jobs:
  deploy:
    name: Deploy to Server
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    # Deploy qua SSH
    - name: Deploy to Production Server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          cd /opt/legal-chatbot
          
          # Pull latest images
          docker-compose pull
          
          # Restart services
          docker-compose down
          docker-compose up -d
          
          # Clean up old images
          docker image prune -f
          
          echo "✅ Deployment completed!"
```

### Setup Server Secrets

Thêm vào GitHub Secrets:
- `SERVER_HOST`: IP server (e.g., 192.168.1.100)
- `SERVER_USER`: SSH username (e.g., root)
- `SSH_PRIVATE_KEY`: Private SSH key

**Tạo SSH key**:
```bash
# Trên local machine
ssh-keygen -t rsa -b 4096 -C "github-actions"

# Copy public key lên server
ssh-copy-id user@server-ip

# Copy private key vào GitHub Secret
cat ~/.ssh/id_rsa
```

---

## 6. Workflow Hoàn Chỉnh - Best Practice

### File `.github/workflows/main.yml` (All-in-one)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  # Job 1: Test
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    
    services:
      # Start PostgreSQL for testing
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      # Start Redis for testing
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements.txt
        pip install -r requirements_dev.txt
    
    - name: Run tests
      run: |
        cd backend
        pytest tests/ -v --cov=src --cov-report=xml
      env:
        POSTGRES_HOST: localhost
        POSTGRES_PORT: 5432
        POSTGRES_USER: postgres
        POSTGRES_PASSWORD: postgres
        POSTGRES_DB: test_db
        REDIS_HOST: localhost
        REDIS_PORT: 6379
    
    # Upload coverage report
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.xml
        flags: backend
        name: backend-coverage

  # Job 2: Lint
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install tools
      run: pip install black flake8 mypy
    
    - name: Check formatting
      run: cd backend/src && black --check .
    
    - name: Lint code
      run: cd backend/src && flake8 . --max-line-length=100
    
    - name: Type check
      run: cd backend/src && mypy . --ignore-missing-imports
      continue-on-error: true

  # Job 3: Build Docker (chỉ chạy khi test pass)
  build:
    name: Build Docker Images
    runs-on: ubuntu-latest
    needs: [test, lint]  # Chờ test và lint pass
    if: github.ref == 'refs/heads/main'  # Chỉ build khi push lên main
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build and push Backend
      uses: docker/build-push-action@v4
      with:
        context: ./backend
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-backend:latest
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-backend:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Build and push Frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: true
        tags: |
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-frontend:latest
          ${{ secrets.DOCKER_USERNAME }}/legal-chatbot-frontend:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

---

## 7. Badges - Hiển Thị Status Trên README

### Thêm vào `README.md`

```markdown
# Vietnamese Legal Chatbot

[![CI/CD Pipeline](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/actions/workflows/main.yml/badge.svg)](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/actions/workflows/main.yml)
[![Docker Build](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/actions/workflows/docker-build.yml/badge.svg)](https://github.com/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/actions/workflows/docker-build.yml)
[![codecov](https://codecov.io/gh/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System/branch/main/graph/badge.svg)](https://codecov.io/gh/mikeethanh/Vietnamese-Legal-Chatbot-RAG-System)

...
```

Badges sẽ hiển thị:
- 🟢 Passing: Tests đang pass
- 🔴 Failing: Tests đang fail
- 🟡 Running: Đang chạy

---

## 8. Local Testing - Kiểm Tra Trước Khi Push

### Install Act (GitHub Actions simulator)

```bash
# macOS
brew install act

# Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Windows (with chocolatey)
choco install act-cli
```

### Chạy workflow locally

```bash
# List tất cả workflows
act -l

# Chạy CI workflow
act push

# Chạy specific job
act -j test

# Chạy với secrets
act -s DOCKER_USERNAME=myuser -s DOCKER_PASSWORD=mypass
```

**Lợi ích**:
- ✅ Test workflow trước khi push
- ✅ Debug nhanh hơn
- ✅ Không tốn GitHub Actions minutes

---

## 9. Monitoring & Notifications

### Slack Notification

Thêm vào cuối workflow:

```yaml
- name: Slack Notification
  if: always()  # Chạy dù success hay failure
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'CI/CD Pipeline: ${{ job.status }}'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

Setup:
1. Tạo Slack App: https://api.slack.com/apps
2. Enable Incoming Webhooks
3. Copy webhook URL
4. Add secret `SLACK_WEBHOOK` vào GitHub

### Email Notification

GitHub tự động gửi email khi workflow fail (nếu bật trong settings)

---

## 10. Best Practices

### 10.1. Caching Dependencies

```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

**Lợi ích**: Giảm thời gian từ 3 phút → 30 giây

### 10.2. Matrix Testing (Test nhiều versions)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
```

Test cùng lúc 3 versions Python!

### 10.3. Conditional Steps

```yaml
# Chỉ chạy khi push lên main
- name: Deploy
  if: github.ref == 'refs/heads/main'
  run: ./deploy.sh

# Chỉ chạy khi test fail
- name: Upload logs
  if: failure()
  uses: actions/upload-artifact@v3
```

### 10.4. Secrets Management

```yaml
# ❌ BAD - hardcoded secrets
env:
  API_KEY: "abc123xyz"

# ✅ GOOD - use GitHub Secrets
env:
  API_KEY: ${{ secrets.API_KEY }}
```

### 10.5. Parallel Jobs

```yaml
jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps: [...]
  
  test-frontend:
    runs-on: ubuntu-latest
    steps: [...]
  
  # Chạy song song → nhanh gấp đôi!
```

---

## 11. Troubleshooting

### Issue 1: Workflow không chạy

**Nguyên nhân**: File YAML sai indent hoặc syntax

**Giải pháp**:
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# Hoặc dùng online validator
https://www.yamllint.com/
```

### Issue 2: Tests fail trên CI nhưng pass local

**Nguyên nhân**: Environment khác nhau (database, env vars)

**Giải pháp**:
```yaml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: postgres

env:
  POSTGRES_HOST: localhost
  POSTGRES_PASSWORD: postgres
```

### Issue 3: Docker build timeout

**Nguyên nhân**: Không dùng cache

**Giải pháp**:
```yaml
- uses: docker/build-push-action@v4
  with:
    cache-from: type=gha  # ← Thêm cache
    cache-to: type=gha,mode=max
```

### Issue 4: Rate limit exceeded

**GitHub Actions limits**:
- Public repos: 2000 minutes/month
- Private repos: 500 minutes/month (free tier)

**Giải pháp**:
1. Optimize workflows (cache, parallel jobs)
2. Chỉ chạy khi cần (skip CI cho docs)
```yaml
on:
  push:
    paths-ignore:
      - 'docs/**'
      - '*.md'
```

---

## 12. Workflow Examples Cho Dự Án

### Minimal CI (Chỉ test)

```yaml
name: Minimal CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: cd backend && pip install -r requirements.txt
      - run: cd backend && pytest tests/
```

### Full Production CI/CD

Xem section 6 phía trên - có test, lint, build, deploy.

---

## 13. Tóm Tắt Commands

### Setup CI/CD từ đầu

```bash
# 1. Tạo thư mục workflows
mkdir -p .github/workflows

# 2. Tạo CI workflow
cat > .github/workflows/ci.yml << 'EOF'
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: cd backend && pip install -r requirements.txt
      - run: cd backend && pytest tests/ -v
EOF

# 3. Commit và push
git add .github/
git commit -m "Add CI workflow"
git push origin main

# 4. Kiểm tra trên GitHub
# Vào tab "Actions" để xem kết quả
```

### Test locally với Act

```bash
# Install act
brew install act  # macOS
# hoặc
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Chạy workflow
act push

# Chạy specific job
act -j test
```

---

## 14. Kết Luận

### CI/CD Workflow Cơ Bản:

```
Code Change
    ↓
Push to GitHub
    ↓
GitHub Actions Triggered
    ↓
┌─────────────────────────────┐
│  1. Run Tests (pytest)      │ → ✅ Pass / ❌ Fail
│  2. Check Code Quality      │ → ✅ Pass / ❌ Fail
│  3. Build Docker Images     │ → ✅ Success
│  4. Push to Docker Hub      │ → ✅ Success
│  5. Deploy to Server        │ → ✅ Success
└─────────────────────────────┘
    ↓
✅ Deployment Complete!
```

### Key Takeaways:

1. **GitHub Actions** = Công cụ CI/CD miễn phí, mạnh mẽ
2. **Workflows** trong `.github/workflows/*.yml`
3. **Secrets** cho credentials (Docker, SSH keys)
4. **Cache** để tăng tốc (pip, Docker layers)
5. **Badges** để show status trên README

### Resources:

- GitHub Actions Docs: https://docs.github.com/actions
- Workflow examples: https://github.com/actions/starter-workflows
- Act (local testing): https://github.com/nektos/act
- Docker build action: https://github.com/docker/build-push-action

**Happy CI/CD! 🚀**
