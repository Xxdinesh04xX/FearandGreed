# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the application
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs data models/cache

# Create non-root user
RUN useradd --create-home --shell /bin/bash goquant && \
    chown -R goquant:goquant /app

# Switch to non-root user
USER goquant

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Default command
CMD ["python", "-m", "goquant.main"]
