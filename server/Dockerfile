# Render-optimized Dockerfile for AI Matchmaker
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages using pre-compiled wheels only
# This avoids compilation and significantly reduces build time
RUN pip install --no-cache-dir \
    --only-binary=all \
    -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY templates/ templates/
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/

# Copy pre-trained model files and outputs
COPY outputs/ outputs/

# Ensure all output directories exist
RUN mkdir -p outputs/data outputs/embeddings outputs/vector_db

# Set environment variables
ENV APP_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Render uses PORT environment variable
EXPOSE $PORT

# Health check for Render
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application
CMD ["python", "server.py"]
