#!/bin/bash

# Frontend Entrypoint Script for Vietnamese Legal Chatbot
# This script sets up the environment and runs the Streamlit application

echo "🚀 Starting Vietnamese Legal Chatbot Frontend..."

# Set environment variables if not already set
export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8051}
export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
export API_BASE_URL=${API_BASE_URL:-http://chatbot-api:8000}

# Create necessary directories
mkdir -p /app/logs
mkdir -p /app/data

# Set permissions
chmod -R 755 /app

# Function to check if backend is ready
check_backend() {
    echo "⏳ Checking backend availability at $API_BASE_URL..."
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$API_BASE_URL/health" > /dev/null 2>&1; then
            echo "✅ Backend is ready!"
            return 0
        fi
        
        echo "⏳ Attempt $attempt/$max_attempts - Backend not ready yet, waiting..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "⚠️  Backend not available after $max_attempts attempts. Starting frontend anyway..."
    return 1
}

# Wait for backend (optional - comment out if not needed)
# check_backend

# Install any additional requirements if requirements.txt has been updated
if [ -f requirements.txt ]; then
    echo "📦 Installing/updating Python packages..."
    pip install --no-cache-dir -r requirements.txt
fi

# Create Streamlit config if it doesn't exist
if [ ! -f ~/.streamlit/config.toml ]; then
    echo "⚙️  Creating Streamlit configuration..."
    mkdir -p ~/.streamlit
    cp /app/config.toml ~/.streamlit/config.toml
fi

# Check which interface to run
INTERFACE_FILE="chat_interface_new.py"
if [ "$USE_LEGACY_INTERFACE" = "true" ]; then
    INTERFACE_FILE="chat_interface.py"
    echo "📱 Using legacy interface: $INTERFACE_FILE"
else
    echo "📱 Using new interface: $INTERFACE_FILE"
fi

# Verify the interface file exists
if [ ! -f "$INTERFACE_FILE" ]; then
    echo "❌ Error: $INTERFACE_FILE not found!"
    echo "📂 Available files:"
    ls -la *.py
    exit 1
fi

# Set up logging
export STREAMLIT_LOGGER_LEVEL=${LOG_LEVEL:-INFO}

echo "🌟 Starting Streamlit application..."
echo "🌐 Interface: $INTERFACE_FILE"
echo "🔗 Port: $STREAMLIT_SERVER_PORT"
echo "📡 Backend: $API_BASE_URL"

# Health check function for the container
health_check() {
    curl -f http://localhost:$STREAMLIT_SERVER_PORT/_stcore/health || exit 1
}

# Export health check function
export -f health_check

# Run Streamlit with proper configuration
exec streamlit run "$INTERFACE_FILE" \
    --server.port="$STREAMLIT_SERVER_PORT" \
    --server.address="$STREAMLIT_SERVER_ADDRESS" \
    --server.baseUrlPath="" \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false \
    --theme.primaryColor="#1e40af" \
    --theme.backgroundColor="#f8fafc" \
    --theme.secondaryBackgroundColor="#ffffff" \
    --theme.textColor="#1e293b" \
    --theme.font="sans serif"