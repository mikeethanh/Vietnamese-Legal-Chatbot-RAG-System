#!/bin/sh

echo "=== Entrypoint Debug Info ==="
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Python path: $PYTHONPATH"
echo "Files in current directory:"
ls -la
echo "Files in src directory:"
ls -la src/ || echo "src directory not found"
echo "Environment variables:"
env | grep -E "(PYTHON|DEBUG|OPENAI)" || echo "No relevant env vars found"
echo "=== End Debug Info ==="

# If no command is provided, start the FastAPI server
if [ $# -eq 0 ]; then
    echo "Starting FastAPI server..."
    echo "Trying to run: uvicorn src.app:app --host 0.0.0.0 --port 8000"
    exec uvicorn src.app:app --host 0.0.0.0 --port 8000
else
    echo "Running custom command: $@"
    exec "$@"
fi
