# Production Cloud Model Deployment Container (Compatible with Hugging Face Spaces, Render, AWS, GCP)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cpu \
    PORT=7860

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create a non-root user (UID 1000 required for Hugging Face Spaces security)
RUN useradd -m -u 1000 user && \
    mkdir -p /app && \
    chown -R user:user /app

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files, weights, and configuration
COPY --chown=user:user api/ api/
COPY --chown=user:user models/ models/
COPY --chown=user:user inference/ inference/
COPY --chown=user:user core/ core/
COPY --chown=user:user weights/ weights/
COPY --chown=user:user yolov8n.pt .
COPY --chown=user:user .env.example .env

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose cloud inference port (Hugging Face Spaces default 7860)
EXPOSE 7860

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/api/v1/health || exit 1

# Start high-performance ASGI Cloud Server with dynamic port resolution
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]

