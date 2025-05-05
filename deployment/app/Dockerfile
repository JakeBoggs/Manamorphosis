# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /

# Install system dependencies (if any)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY deployment/app/requirements.txt .
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and templates
COPY deployment/app /app
COPY templates /templates # Assuming templates are in the root
# If templates are inside deployment/app, adjust the COPY command:
# COPY deployment/app/templates /templates 

# Set working directory to the app code
WORKDIR /app

# --- Data Download ---
# Download embeddings, doc2vec model, and AtomicCards for search functionality
# The app.py script handles this on startup if files are missing.
# Optionally, uncomment the RUN command below to download during build:
RUN python -c "import app; app.ensure_search_data_downloaded()"

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables for RunPod (can be overridden at runtime)
ENV RUNPOD_ENDPOINT_URL="YOUR_RUNPOD_ENDPOINT_URL_HERE"
ENV RUNPOD_API_KEY="YOUR_RUNPOD_API_KEY_HERE"
ENV FLASK_APP=app.py
# Optional: Set FLASK_ENV to production for production builds
# ENV FLASK_ENV=production 

# Run the application using Waitress (a production-ready WSGI server)
# Use 0.0.0.0 to allow connections from outside the container
# Use the port exposed above (8080)
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "app:app"] 