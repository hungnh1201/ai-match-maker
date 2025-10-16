# Production AI Matchmaker - Clean Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (only what exists)
COPY server.py .
COPY templates/ templates/
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/
COPY outputs/ outputs/

# Set environment variables
ENV APP_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application
CMD ["python", "server.py"]
