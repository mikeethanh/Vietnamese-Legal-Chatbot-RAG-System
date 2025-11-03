#!/bin/sh

# If no command is provided, start the FastAPI server
if [ $# -eq 0 ]; then
    echo "Starting FastAPI server..."
    exec uvicorn src.app:app --host 0.0.0.0 --port 8000
else
    exec "$@"
fi
