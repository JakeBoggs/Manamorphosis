# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Add any other system dependencies needed by your libraries here (e.g., for specific hardware acceleration)
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Using --no-cache-dir reduces image size
COPY deployment/inference/requirements.txt . 
# Ensure requirements.txt exists or list packages directly:
# RUN pip install --no-cache-dir runpod torch torchvision torchaudio numpy scipy gensim nltk requests flask
RUN pip install --no-cache-dir -r requirements.txt \
    # Install PyTorch with CUDA support if needed (adjust version and CUDA capability)
    # Example for CUDA 11.8 - check PyTorch website for current commands
    # torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Example for CPU only:
    # torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    # Assuming requirements.txt handles the correct PyTorch version

# Copy the handler script FIRST so it can be called for download
COPY deployment/inference/rp_handler.py .

# --- Model & Data Download --- 
# The rp_handler.py script will handle downloading models on first run if not present.
# This avoids bloating the image if models are large and frequently updated.
# Alternatively, uncomment the RUN command below to download during build:
RUN python -c "import rp_handler; rp_handler.ensure_models_downloaded()"

# Copy the handler script into the container at / 
# COPY deployment/inference/rp_handler.py . # No longer needed here, copied above

# Make port 80 available to the world outside this container (if needed, RunPod usually handles this)
# EXPOSE 80 

# Define environment variable
ENV NAME World 
# (You can add other environment variables here if needed)

# Run rp_handler.py when the container launches
# The -u flag ensures that print statements are sent straight to stdout without being buffered
CMD ["python3", "-u", "rp_handler.py"] 