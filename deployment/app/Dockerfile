FROM python:3.10-slim

WORKDIR /

# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir Flask gensim nltk requests numpy scipy waitress gdown

COPY deployment/app/app.py .
COPY ../../templates /templates
COPY ../../static /static

# --- Data Download ---
# Create target and temporary directories
RUN mkdir -p /models /data /tmp/download
# Download data from GDrive into the temporary directory
RUN gdown --folder https://drive.google.com/drive/folders/1ZvVbUGXa8FGzL97lplQGea2Ech7yfR-0 -O /tmp/download --quiet
# Move the downloaded contents to their final destinations
# Note: This assumes gdown downloads the *contents* (models/, data/) into /tmp/download
RUN mv /tmp/download/models/* /models/ || echo "No models found in download to move." # Use wildcard and handle potential empty dir
RUN mv /tmp/download/data/* /data/   || echo "No data found in download to move."
# Clean up the temporary directory
RUN rm -rf /tmp/download

# Make port 8080 available to the world outside this container
EXPOSE 8080

ENV FLASK_APP=app.py
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "app:app"] 