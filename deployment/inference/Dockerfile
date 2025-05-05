FROM python:3.10-slim

WORKDIR /

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir runpod torch nltk requests
# Copy the handler script FIRST so it can be called for download
COPY deployment/inference/handler.py .

# --- Model & Data Download --- 
# Create data directory before downloading into it
RUN mkdir -p /data
# Download AtomicCards.json directly
RUN curl -L -o /data/AtomicCards.json https://mtgjson.com/api/v5/AtomicCards.json
# Download other models/data using the script function
RUN python -c "import rp_handler; rp_handler.ensure_models_downloaded()"

# Run rp_handler.py when the container launches
# The -u flag ensures that print statements are sent straight to stdout without being buffered
CMD ["python3", "-u", "rp_handler.py"] 