#!/bin/bash

# Test runner script for Vietnamese Legal Chatbot
# Usage: ./scripts/run_tests.sh [test_type]
# Test types: unit, integration, performance, all

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
check_venv() {
    if [[ -z "${VIRTUAL_ENV}" ]]; then
        print_warning "No virtual environment detected. Consider activating one."
    else
        print_status "Using virtual environment: ${VIRTUAL_ENV}"
    fi
}

# Install dependencies
install_deps() {
    print_status "Installing dependencies..."
    pip install -r requirements_dev.txt
    if [ -f backend/requirements.txt ]; then
        pip install -r backend/requirements.txt
    fi
}

# Run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    pytest tests/ -v \
        --cov=backend/src \
        --cov-report=html:htmlcov \
        --cov-report=term-missing \
        --cov-report=xml \
        --junit-xml=pytest-results.xml \
        -m "not integration and not performance" \
        --tb=short
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    pytest tests/ -v \
        -m "integration" \
        --tb=short
}

# Run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    pytest tests/ -v \
        -m "performance" \
        --tb=short
}

# Run all tests
run_all_tests() {
    print_status "Running all tests..."
    pytest tests/ -v \
        --cov=backend/src \
        --cov-report=html:htmlcov \
        --cov-report=term-missing \
        --cov-report=xml \
        --junit-xml=pytest-results.xml \
        --tb=short
}

# Run code quality checks
run_quality_checks() {
    print_status "Running code quality checks..."
    
    # Pre-commit hooks
    if [ -f .pre-commit-config.yaml ]; then
        print_status "Running pre-commit hooks..."
        pre-commit run --all-files
    fi
    
    # Type checking
    print_status "Running type checking..."
    mypy backend/src --config-file mypy.ini || true
    
    # Security check
    print_status "Running security check..."
    bandit -r backend/src -f json -o bandit-report.json || true
}

# Main function
main() {
    local test_type=${1:-"all"}
    
    print_status "Vietnamese Legal Chatbot Test Runner"
    print_status "Test type: $test_type"
    
    check_venv
    
    case $test_type in
        "unit")
            install_deps
            run_unit_tests
            ;;
        "integration")
            install_deps
            run_integration_tests
            ;;
        "performance")
            install_deps
            run_performance_tests
            ;;
        "quality")
            install_deps
            run_quality_checks
            ;;
        "all")
            install_deps
            run_quality_checks
            run_all_tests
            ;;
        *)
            print_error "Unknown test type: $test_type"
            print_status "Available types: unit, integration, performance, quality, all"
            exit 1
            ;;
    esac
    
    print_status "Test execution completed!"
}

# Run main function with all arguments
main "$@"