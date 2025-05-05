FROM python:3.10-slim

WORKDIR /

# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r Flask gensim nltk requests numpy scipy pickle5 waitress gdown

# Copy application code and templates
COPY deployment/app /app
COPY templates /templates
COPY static /static

# Set working directory to the app code
WORKDIR /app

# --- Data Download ---
RUN python -c "import app; app.ensure_search_data_downloaded()"

# Make port 8080 available to the world outside this container
EXPOSE 8080

ENV FLASK_APP=app.py
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "app:app"] 