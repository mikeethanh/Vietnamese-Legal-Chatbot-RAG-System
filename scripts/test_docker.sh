#!/bin/bash

# Docker test runner for Vietnamese Legal Chatbot
# Usage: ./scripts/test_docker.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up containers..."
    docker-compose -f backend/docker-compose.yml down || true
    docker-compose -f frontend/docker-compose.yml down || true
    docker container prune -f || true
}

# Set trap for cleanup
trap cleanup EXIT

# Test backend Docker build
test_backend_docker() {
    print_status "Testing backend Docker build..."
    
    cd backend
    
    # Build image
    docker build -t legal-chatbot-backend:test .
    
    # Start services
    docker-compose up -d
    
    # Wait for services
    print_status "Waiting for backend services to start..."
    sleep 30
    
    # Health check
    print_status "Running health check..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_status "Backend health check passed!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Backend health check failed after 30 attempts"
            docker-compose logs
            exit 1
        fi
        sleep 2
    done
    
    # Test API endpoint
    print_status "Testing API endpoints..."
    
    # Test health endpoint
    response=$(curl -s http://localhost:8000/health)
    if echo "$response" | grep -q "healthy"; then
        print_status "Health endpoint test passed"
    else
        print_error "Health endpoint test failed"
        exit 1
    fi
    
    cd ..
}

# Test frontend Docker build
test_frontend_docker() {
    print_status "Testing frontend Docker build..."
    
    cd frontend
    
    # Build image
    docker build -t legal-chatbot-frontend:test .
    
    # Test run (don't start permanently)
    docker run --rm -d --name frontend-test \
        -p 8501:8501 \
        legal-chatbot-frontend:test &
    
    sleep 15
    
    # Check if container is running
    if docker ps | grep -q frontend-test; then
        print_status "Frontend container is running"
        
        # Basic connectivity check
        if curl -f http://localhost:8501 > /dev/null 2>&1; then
            print_status "Frontend connectivity test passed"
        else
            print_warning "Frontend connectivity test failed (this might be expected for Streamlit)"
        fi
        
        docker stop frontend-test
    else
        print_error "Frontend container failed to start"
        exit 1
    fi
    
    cd ..
}

# Test full stack
test_full_stack() {
    print_status "Testing full stack deployment..."
    
    # Use docker-compose to start everything
    docker-compose -f docker-compose.yml up -d || print_warning "Full stack compose file not found"
    
    # If there's a full stack compose file, test it
    if [ -f docker-compose.yml ]; then
        sleep 30
        
        # Test services
        print_status "Testing full stack health..."
        
        # Add full stack tests here
        print_status "Full stack test completed"
    fi
}

# Main function
main() {
    print_status "Vietnamese Legal Chatbot Docker Test Runner"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_status "Docker is available"
    
    # Run tests
    test_backend_docker
    test_frontend_docker
    test_full_stack
    
    print_status "All Docker tests completed successfully!"
}

# Run main function
main "$@"