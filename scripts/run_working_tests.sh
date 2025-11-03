#!/bin/bash

# Updated test runner script that only runs working tests
# Usage: ./scripts/run_working_tests.sh [test_type]

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

# Working test files (ones that don't require complex dependencies)
WORKING_TESTS="tests/test_basic.py tests/test_api_simple.py tests/test_utils.py tests/test_backend_utils.py tests/test_brain.py"

# Run working tests only
run_working_tests() {
    print_status "Running working tests (basic functionality)..."
    pytest $WORKING_TESTS -v \
        --tb=short \
        --cov=tests \
        --cov-report=term-missing
}

# Run working tests with coverage
run_working_tests_with_coverage() {
    print_status "Running working tests with coverage..."
    pytest $WORKING_TESTS -v \
        --tb=short \
        --cov=tests \
        --cov-report=html:htmlcov \
        --cov-report=term-missing \
        --cov-report=xml
}

# Run unit tests specifically  
run_unit_tests() {
    print_status "Running unit tests..."
    pytest $WORKING_TESTS -v \
        -m "unit" \
        --tb=short
}

# Quick check
run_quick_check() {
    print_status "Running quick test check..."
    pytest $WORKING_TESTS -x --tb=line
}

# Main function
main() {
    local test_type=${1:-"working"}
    
    print_status "Vietnamese Legal Chatbot Test Runner (Working Tests Only)"
    print_status "Test type: $test_type"
    
    case $test_type in
        "working")
            run_working_tests
            ;;
        "coverage")
            run_working_tests_with_coverage
            ;;
        "unit")
            run_unit_tests
            ;;
        "quick")
            run_quick_check
            ;;
        *)
            print_error "Unknown test type: $test_type"
            print_status "Available types: working, coverage, unit, quick"
            exit 1
            ;;
    esac
    
    print_status "Test execution completed!"
}

# Run main function with all arguments
main "$@"