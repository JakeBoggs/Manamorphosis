FROM python:3.10-slim

WORKDIR /

# Install system dependencies required for building some Python packages and downloading
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies including gdown
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir runpod torch numpy nltk requests gdown

# Copy the handler script FIRST so it can be called if needed (though download moved)
COPY deployment/inference/handler.py .

# --- Model & Data Download ---
# Create the target and temporary directories
RUN mkdir -p /models /data /tmp/download
# Download models/data from GDrive into the temporary directory
RUN gdown --folder https://drive.google.com/drive/folders/1ZvVbUGXa8FGzL97lplQGea2Ech7yfR-0 -O /tmp/download --quiet
# Move the downloaded contents to their final destinations
RUN mv /tmp/download/models/* /models/ || echo "No models found in download to move."
RUN mv /tmp/download/data/* /data/   || echo "No data found in download to move."
# Clean up the temporary directory
RUN rm -rf /tmp/download
# Download AtomicCards.json directly using curl into the final data directory
RUN curl -L -o /data/AtomicCards.json https://mtgjson.com/api/v5/AtomicCards.json

# Run handler.py when the container launches
# The -u flag ensures that print statements are sent straight to stdout without being buffered
CMD ["python3", "-u", "handler.py"] 