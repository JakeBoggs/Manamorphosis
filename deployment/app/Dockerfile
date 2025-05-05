FROM python:3.10-slim

WORKDIR /app

# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir Flask gensim nltk requests numpy scipy pickle5 waitress gdown

# Copy application code (source is '.', the build context) into WORKDIR (/app)
COPY . .

# Copy templates and static directories from parent directories relative to build context
COPY ../../templates /templates
COPY ../../static /static

# --- Data Download ---
RUN python -c "import app; app.ensure_search_data_downloaded()"

# Make port 8080 available to the world outside this container
EXPOSE 8080

ENV FLASK_APP=app.py
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8080", "app:app"] 