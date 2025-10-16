# Production AI Matchmaker - Server Dockerfile (Optimized)
# Multi-stage build to reduce final image size

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-docker.txt .

# Install Python packages with optimizations for CPU-only PyTorch
RUN pip install --no-cache-dir --user \
    torch==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user \
    flask==3.0.0 flask-cors==4.0.0 pandas==2.1.3 numpy==1.25.2 \
    transformers==4.35.2 sentence-transformers==2.2.2 \
    scikit-learn==1.3.2 faiss-cpu==1.7.4 geopy==2.4.1 PyYAML==6.0.1

# Production stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application files (excluding outputs/ - will be created at runtime)
COPY server.py .
COPY templates/ templates/
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/

# Create outputs directory for runtime
RUN mkdir -p outputs/data outputs/embeddings outputs/vector_db

# Set environment variables
ENV APP_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application
CMD ["python", "server.py"]
