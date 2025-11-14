# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for GPU optimization
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Copy application code
COPY data_processing/ ./data_processing/
COPY finetune/ ./finetune/
COPY evaluation/ ./evaluation/
COPY serving/ ./serving/

# Copy environment file template
COPY .env.template .env

# Create directories for model and data
RUN mkdir -p /app/model /app/data /app/outputs

# Set permissions
RUN chmod +x /app/serving/serve_model.py

# Expose port
EXPOSE 6000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6000/health || exit 1

# Default command (can be overridden)
CMD ["python", "/app/serving/serve_model.py"]