# Multi-stage build to reduce image size
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy only essential application files
COPY server.py .
COPY templates/ templates/
COPY scripts/step5_generate_recommendations.py scripts/
COPY scripts/__init__.py scripts/
COPY models/__init__.py models/
COPY models/cross_attention_recommendation_model.py models/
COPY models/sentence_encoder.py models/

# Copy only the trained model (not all training files)
COPY models/trained/best_model.pt models/trained/

# Copy only essential data files
COPY data/processed/ data/processed/
COPY outputs/ outputs/

# Set environment variables
ENV APP_ENV=production
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Run the application
CMD ["python", "server.py"]
