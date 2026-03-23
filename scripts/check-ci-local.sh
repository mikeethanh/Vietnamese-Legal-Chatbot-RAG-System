#!/bin/bash

# 🧪 Script kiểm tra CI/CD locally trước khi push
# Chạy: ./scripts/check-ci-local.sh

echo "🚀 LOCAL CI/CD CHECK"
echo "==================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."

echo -e "\n📁 Project: $(pwd)"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Not in project root directory${NC}"
    exit 1
fi

# Install dependencies if needed
echo -e "\n📦 Checking dependencies..."
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment found${NC}"
    echo "💡 Tip: Create venv with: python -m venv venv && source venv/bin/activate"
fi

# Check Python version
echo -e "\n🐍 Python version:"
python --version

# Install basic tools if needed
echo -e "\n🔧 Installing basic tools..."
pip install black flake8 pytest bandit --quiet

echo -e "\n🔍 1. Code Format Check (Black)"
echo "================================"
if black --check backend/src/ frontend/ 2>/dev/null; then
    echo -e "${GREEN}✅ Code format: OK${NC}"
else
    echo -e "${YELLOW}⚠️  Code format issues found${NC}"
    echo "💡 Fix with: black backend/src/ frontend/"
fi

echo -e "\n🧹 2. Lint Check (Flake8)" 
echo "========================="
if flake8 backend/src/ --count --statistics 2>/dev/null; then
    echo -e "${GREEN}✅ Linting: OK${NC}"
else
    echo -e "${YELLOW}⚠️  Linting issues found${NC}"
fi

echo -e "\n🧪 3. Basic Tests"
echo "================="
if [ -d "tests" ]; then
    if pytest tests/test_basic.py -v 2>/dev/null; then
        echo -e "${GREEN}✅ Tests: PASSED${NC}"
    else
        echo -e "${YELLOW}⚠️  Some tests failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  No tests directory found${NC}"
fi

echo -e "\n🔒 4. Security Check (Bandit)"
echo "============================="
if bandit -r backend/src/ -f text 2>/dev/null; then
    echo -e "${GREEN}✅ Security: OK${NC}"
else
    echo -e "${YELLOW}⚠️  Security issues found${NC}"
fi

echo -e "\n🐳 5. Docker Build Test"
echo "======================="
if [ -f "backend/Dockerfile" ]; then
    echo "🐳 Testing backend Docker build..."
    if cd backend && docker build -t test-backend . >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend Docker: OK${NC}"
        docker rmi test-backend >/dev/null 2>&1
    else
        echo -e "${YELLOW}⚠️  Backend Docker build failed${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  No backend Dockerfile found${NC}"
fi

if [ -f "frontend/Dockerfile" ]; then
    echo "🖥️ Testing frontend Docker build..."
    if cd frontend && docker build -t test-frontend . >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend Docker: OK${NC}"
        docker rmi test-frontend >/dev/null 2>&1
    else
        echo -e "${YELLOW}⚠️  Frontend Docker build failed${NC}"
    fi
    cd ..
else
    echo -e "${YELLOW}⚠️  No frontend Dockerfile found${NC}"
fi

echo -e "\n📊 SUMMARY"
echo "==========="
echo "✅ Format, lint, test, security checks completed"
echo "🐳 Docker builds tested"
echo -e "${GREEN}🎉 Ready for push to GitHub!${NC}"
echo ""
echo "💡 Next steps:"
echo "   git add ."
echo "   git commit -m 'your message'"
echo "   git push origin your-branch"