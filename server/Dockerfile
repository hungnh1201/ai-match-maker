# Production Dockerfile for AI Matchmaker with full AI model
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .
COPY templates/ templates/
COPY scripts/ scripts/
COPY models/ models/
COPY config/ config/

# Copy data files if they exist, create directories if not
RUN mkdir -p data/processed models/trained outputs
COPY data/processed/ data/processed/ 
COPY models/trained/ models/trained/
COPY outputs/ outputs/

# Set environment variables
ENV APP_ENV=production
ENV PYTHONPATH=/app

# Expose port (will be set by deployment platform)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application
CMD ["python", "server.py"]
