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

# Copy application code
COPY server.py .
COPY templates/ templates/
COPY scripts/ scripts/


# Copy pre-trained models, embeddings, and vector database from outputs/
# This includes:
# - outputs/data/ (processed data for recommendations)
# - outputs/embeddings/ (user embeddings)
# - outputs/models/ (trained models)
# - outputs/vector_db/ (FAISS vector database and configs)
COPY outputs/ outputs/

# Ensure all output directories exist (in case some are empty)
RUN mkdir -p outputs/data outputs/embeddings outputs/models outputs/vector_db

# Create symlinks for vector_db files (script expects latest_ prefix)
RUN cd outputs/vector_db && \
    ln -sf gender_aware_vector_db_config.json latest_config.json && \
    ln -sf male_candidate_ids.pkl latest_male_candidate_ids.pkl && \
    ln -sf male_candidates_index.faiss latest_male_index.faiss && \
    ln -sf user_embeddings.pkl latest_user_embeddings.pkl

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
