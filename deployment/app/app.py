import os
import re
import json
import requests
import pickle
import time
import numpy as np
from flask import Flask, request, jsonify, render_template, current_app
from collections import Counter
import gensim # Still needed for search
from scipy.spatial.distance import cosine # Still needed for search
import nltk
from nltk.corpus import stopwords
import logging
import zipfile
from io import BytesIO

# --- NLTK Stopwords Download ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    try:
        nltk.download('stopwords')
        print("NLTK stopwords downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK stopwords: {e}. Search might be impaired.")

# --- Configuration ---
# Paths relative to the container's WORKDIR ('/')
MODEL_DIR = "/models"
DATA_DIR = "/data"
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "card_embeddings.pkl") # Needed for search
DOC2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "embedding_model") # Needed for search
ATOMIC_CARDS_PATH = os.path.join(DATA_DIR, "AtomicCards.json") # Needed for search name matching, potentially

# Google Drive Folder ID (Needed for downloading data for search)
GDRIVE_FOLDER_ID = "1ZvVbUGXa8FGzL97lplQGea2Ech7yfR-0"

SCRYFALL_API_BASE = "https://api.scryfall.com"

# --- RunPod Endpoint Configuration ---
# !! IMPORTANT: Replace these with your actual RunPod endpoint URL and API Key !!
RUNPOD_ENDPOINT_URL = os.environ.get("RUNPOD_ENDPOINT_URL", "YOUR_RUNPOD_ENDPOINT_URL_HERE")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "YOUR_RUNPOD_API_KEY_HERE")

if RUNPOD_ENDPOINT_URL == "YOUR_RUNPOD_ENDPOINT_URL_HERE" or RUNPOD_API_KEY == "YOUR_RUNPOD_API_KEY_HERE":
    print("WARNING: RunPod endpoint URL or API Key not set. Deck/Sideboard completion will fail.")
    print("Please set the RUNPOD_ENDPOINT_URL and RUNPOD_API_KEY environment variables.")


# --- Constants ---
# Constants needed for search/UI
ALLOWED_FORMATS = {'standard', 'pioneer', 'modern', 'legacy', 'vintage', 'pauper'}
DEFAULT_FORMAT = 'modern'
DECK_SIZE = 60 # Still useful for validation
SIDEBOARD_SIZE = 15 # Still useful for validation

# --- Helper: Download Data (Simplified for app) ---
def download_and_extract_gdrive_folder(folder_id, destination_dir):
    """Downloads and extracts files from a public Google Drive folder."""
    print(f"Attempting to download files from Google Drive folder: {folder_id}")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    try:
        response = session.get(URL, params={'id': folder_id}, stream=True)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        print(f"Response Content-Type: {content_type}")

        if 'zip' in content_type or 'octet-stream' in content_type:
            print("Received ZIP file, extracting...")
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(destination_dir)
            print(f"Extracted files to {destination_dir}")
            # Post-Extraction Check and Move (same as inference worker)
            extracted_items = os.listdir(destination_dir)
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(destination_dir, extracted_items[0])):
                gdrive_subfolder = os.path.join(destination_dir, extracted_items[0])
                print(f"Detected GDrive subfolder: {gdrive_subfolder}. Moving contents up.")
                for item_name in os.listdir(gdrive_subfolder):
                    source_item = os.path.join(gdrive_subfolder, item_name)
                    dest_item = os.path.join(destination_dir, item_name)
                    os.rename(source_item, dest_item)
                os.rmdir(gdrive_subfolder)
            else:
                print("Files extracted directly or multiple items found.")
        else:
            print("Warning: Received unexpected content type. Download might have failed.")
            print("Response Text (first 500 chars):", response.text[:500])
            raise requests.exceptions.RequestException("Failed to download file from Google Drive (unexpected content type).")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading from Google Drive: {e}")
        print("Ensure the Google Drive folder is public ('Anyone with the link can view').")
        raise
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid ZIP archive.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")
        raise

def ensure_search_data_downloaded():
    """Checks if data files needed for search exist, downloads if not."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    data_exist = os.path.exists(EMBEDDINGS_PATH) and \
                 os.path.exists(DOC2VEC_MODEL_PATH) and \
                 os.path.exists(ATOMIC_CARDS_PATH)

    if data_exist:
        print("Search data files already exist. Skipping download.")
        return

    print("Search data files missing. Attempting download from Google Drive...")
    try:
        download_and_extract_gdrive_folder(GDRIVE_FOLDER_ID, "/") # Download to root
        print("Download and extraction complete.")
        # Verify again
        if not (os.path.exists(EMBEDDINGS_PATH) and \
                os.path.exists(DOC2VEC_MODEL_PATH) and \
                os.path.exists(ATOMIC_CARDS_PATH)):
            print("Error: Essential search data files still missing after download attempt.")
            # Add detailed checks like in the inference worker if needed
            raise FileNotFoundError("Essential search data files not found after download.")

    except Exception as e:
        print(f"Failed to download or extract search data: {e}")
        raise

# --- Global Variables for Search Data ---
card_embeddings = None
doc2vec_model = None
cards = None # Optional, but good for checking search results against known names

def load_search_data():
    """Loads only the data required for the search functionality."""
    global card_embeddings, doc2vec_model, cards

    # ensure_search_data_downloaded() # Removed: Handled by Dockerfile build

    print("Loading search data...")

    # Load AtomicCards data (optional but helpful)
    try:
        if os.path.exists(ATOMIC_CARDS_PATH):
            with open(ATOMIC_CARDS_PATH, 'r', encoding='utf-8') as f:
                cards_data = json.load(f)
                if isinstance(cards_data.get('data'), dict):
                     cards = cards_data['data']
                     print(f"Loaded {len(cards)} card entries from AtomicCards.json")
                else:
                     print("Warning: AtomicCards.json structure unexpected. Card name validation might be limited.")
                     cards = {}
        else:
            print(f"Warning: AtomicCards file not found at {ATOMIC_CARDS_PATH}")
            cards = {}
    except Exception as e:
        print(f"Error loading AtomicCards.json: {e}")
        cards = {}

    # Load Embeddings
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            card_embeddings = pickle.load(f)
        print(f"Loaded {len(card_embeddings)} card embeddings.")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise

    # Load Doc2Vec Model
    if not os.path.exists(DOC2VEC_MODEL_PATH):
        raise FileNotFoundError(f"Doc2Vec model file not found: {DOC2VEC_MODEL_PATH}")
    try:
        print(f"Loading Doc2Vec model from {DOC2VEC_MODEL_PATH}...")
        doc2vec_model = gensim.models.Doc2Vec.load(DOC2VEC_MODEL_PATH)
        print("Doc2Vec model loaded.")
    except Exception as e:
        print(f"Error loading Doc2Vec model: {e}")
        raise

    print("Search data loaded successfully.")


# --- Flask App Initialization ---
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load search data when the app starts
try:
    load_search_data()
except Exception as e:
    app.logger.error(f"FATAL: Failed to load search data on startup: {e}")
    # Depending on requirements, you might exit or let Flask start with search disabled.

# --- Text Processing (for search) ---
reminder_remover = re.compile(r'\(.*?\)') # Match parentheses and content (fixed regex)
stop_words = set(stopwords.words('english'))

def clean_search_text(text):
    """Cleans the user's search description similarly to card text preprocessing."""
    if not text:
        return []
    text = text.lower()
    text = re.sub(reminder_remover, '', text.replace('}{', '} {'))
    text = text.replace('&', 'and').replace('\n', ' ').replace(';', ' ').replace(':', ' :')
    text = text.replace('−', '-').replace('—', '-') # Handle different dash types
    text = text.replace('’', "'").replace('`', "'") # Handle apostrophes
    text = text.replace(',', '').replace('.', '').replace("'", '').replace('"', '') # Remove punctuation
    words = text.split()
    filtered_words = [word for word in words if word and word not in stop_words]
    return filtered_words

def parse_deck_input(deck_text):
    """Parses deck input text. Needed for validation before sending to RunPod."""
    parsed_cards = []
    lines = deck_text.strip().split('\n')
    pattern = re.compile(r"^\s*(\d+)\s*[xX]?\s*(.+?)\s*$")
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        match = pattern.match(line)
        if match:
            count = int(match.group(1))
            name = match.group(2).strip()
            if count > 0 and name:
                # Simple check if name seems plausible (optional)
                # if cards and name not in cards and name not in card_embeddings:
                #    raise ValueError(f"Warning: Card name '{name}' not found in local data. Check spelling or format.")
                parsed_cards.append({"name": name, "count": count})
        else:
            name = line.strip()
            if name:
                 # if cards and name not in cards and name not in card_embeddings:
                 #     raise ValueError(f"Warning: Card name '{name}' not found in local data. Check spelling or format.")
                 parsed_cards.append({"name": name, "count": 1})

    card_counts = Counter()
    for card in parsed_cards:
        card_counts[card['name']] += card['count']
    return [{'name': name, 'count': count} for name, count in card_counts.items()]


# --- Scryfall Image Fetching (Identical to inference worker) ---
image_cache = {}
def get_card_image_urls(card_names):
    """Fetches image URLs from Scryfall for a list of card names."""
    # ... (Keep the same implementation as in rp_handler.py) ...
    urls = {}
    names_to_fetch = set()
    name_map = {} # Maps the part sent to Scryfall back to the original full name

    for name in card_names:
        if name not in image_cache:
            scryfall_query_name = name.split("//")[0].strip()
            names_to_fetch.add(scryfall_query_name)
            if scryfall_query_name not in name_map:
                name_map[scryfall_query_name] = []
            if name not in name_map[scryfall_query_name]:
                name_map[scryfall_query_name].append(name)
        else:
            urls[name] = image_cache[name]

    if not names_to_fetch:
        return urls

    current_app.logger.info(f"Fetching {len(names_to_fetch)} unique card names/parts from Scryfall...")
    identifiers = [{"name": name} for name in names_to_fetch]
    payload = {"identifiers": identifiers}

    try:
        response = requests.post(f"{SCRYFALL_API_BASE}/cards/collection", json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data and 'data' in data:
            for card_data in data['data']:
                found_query_name = card_data.get('name')
                if "//" in found_query_name:
                    found_query_name = found_query_name.split("//")[0].strip()

                original_full_names = name_map.get(found_query_name)
                if not original_full_names:
                     current_app.logger.warning(f"Scryfall returned card '{found_query_name}', but couldn't map it back.")
                     continue

                image_info = None
                if card_data.get('card_faces') and len(card_data['card_faces']) > 1:
                    face1 = card_data['card_faces'][0]
                    face2 = card_data['card_faces'][1]
                    url1 = face1.get('image_uris', {}).get('normal')
                    url2 = face2.get('image_uris', {}).get('normal')
                    if url1 or url2:
                        image_info = {'front': url1, 'back': url2}
                    elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                        image_info = card_data['image_uris']['normal']
                elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                    image_info = card_data['image_uris']['normal']

                for original_name in original_full_names:
                    if image_info is not None:
                        image_cache[original_name] = image_info
                        urls[original_name] = image_info
                    else:
                        image_cache[original_name] = None
                        urls[original_name] = None
                        current_app.logger.warning(f"Card '{original_name}' found but no suitable image URI.")

        if 'not_found' in data:
            for not_found_identifier in data['not_found']:
                if 'name' in not_found_identifier:
                    missing_query_name = not_found_identifier['name']
                    original_full_names = name_map.get(missing_query_name)
                    if original_full_names:
                        for original_name in original_full_names:
                            if original_name not in urls:
                                image_cache[original_name] = None
                                urls[original_name] = None
                                current_app.logger.warning(f"Card '{original_name}' not found by Scryfall.")

    except requests.exceptions.Timeout:
        current_app.logger.error("Scryfall API request timed out.")
        for query_name, original_names in name_map.items():
            for original_name in original_names:
                 if original_name not in urls:
                    image_cache[original_name] = None
                    urls[original_name] = None
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Scryfall API request failed: {e}")
        for query_name, original_names in name_map.items():
            for original_name in original_names:
                 if original_name not in urls:
                    image_cache[original_name] = None
                    urls[original_name] = None
    except json.JSONDecodeError as e:
         current_app.logger.error(f"Failed to decode Scryfall JSON response: {e}")
         for query_name, original_names in name_map.items():
            for original_name in original_names:
                 if original_name not in urls:
                    image_cache[original_name] = None
                    urls[original_name] = None

    for name in card_names:
        if name not in urls:
            urls[name] = image_cache.get(name, None)

    return urls

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    # Ensure you have a templates/index.html file
    return render_template('index.html')

@app.route('/search-cards', methods=['POST'])
def search_cards():
    """Handles card search requests based on text description using local data."""
    try:
        if not doc2vec_model or not card_embeddings:
             # Maybe add a check for cards dict too if validation is strict
            raise RuntimeError("Search models or data not loaded properly.")

        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' in request"}), 400

        description = data['description'].strip()
        top_n = data.get('top_n', 10)

        if not description:
            return jsonify({"error": "Description cannot be empty"}), 400

        # --- Determine Query Vector ---
        query_vector = None
        trimmed_description = description.strip()
        if trimmed_description in card_embeddings:
            current_app.logger.info(f"Input '{trimmed_description}' matches a known card. Using its embedding.")
            query_vector = card_embeddings[trimmed_description]
        else:
            current_app.logger.info(f"Input '{description}' does not match a known card. Inferring vector from text.")
            cleaned_tokens = clean_search_text(description)
            if not cleaned_tokens:
                return jsonify({"error": "Description contained no searchable words after cleaning."}), 400
            query_vector = doc2vec_model.infer_vector(cleaned_tokens)

        if query_vector is None:
            return jsonify({"error": "Failed to determine query vector."}), 500

        # --- Calculate Similarities ---
        similarities = []
        for card_name, embedding in card_embeddings.items():
            similarity_score = 1 - cosine(np.array(query_vector), np.array(embedding))
            similarities.append((card_name, similarity_score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_cards = similarities[:top_n]

        # --- Prepare results and fetch images ---
        result_cards = []
        top_card_names = [name for name, score in top_cards]
        image_urls = get_card_image_urls(top_card_names) # Use the shared image fetcher

        for name, score in top_cards:
            result_cards.append({
                "name": name,
                "similarity": float(score),
                "image_url": image_urls.get(name)
            })

        return jsonify({"results": result_cards})

    except ValueError as e:
        current_app.logger.error(f"Value Error during card search: {e}")
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
        current_app.logger.error(f"Runtime Error during card search: {e}")
        return jsonify({"error": "Server configuration error during search."}), 500
    except Exception as e:
        current_app.logger.exception("An unexpected error occurred during card search:")
        return jsonify({"error": "An internal server error occurred."}), 500


# --- Routes calling the RunPod Inference Endpoint --- 

def call_runpod_endpoint(payload):
    """Helper function to call the RunPod serverless endpoint."""
    if RUNPOD_ENDPOINT_URL == "YOUR_RUNPOD_ENDPOINT_URL_HERE" or RUNPOD_API_KEY == "YOUR_RUNPOD_API_KEY_HERE":
        return {"error": "RunPod endpoint URL or API Key not configured in the server."}, 503

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    run_url = f"{RUNPOD_ENDPOINT_URL.replace('/runsync', '')}/run" # Use async run

    try:
        response = requests.post(run_url, headers=headers, json=payload, timeout=10) # Timeout for starting the job
        response.raise_for_status()
        run_result = response.json()
        job_id = run_result.get('id')

        if not job_id:
            return {"error": "Failed to start RunPod job", "details": run_result}, 500

        # Poll for the result (implement with backoff and timeout)
        status_url = f"{RUNPOD_ENDPOINT_URL.replace('/runsync', '')}/status/{job_id}"
        max_wait_seconds = 180 # Adjust as needed (e.g., 3 minutes)
        poll_interval = 2 # Start polling every 2 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            time.sleep(poll_interval)
            status_response = requests.get(status_url, headers=headers, timeout=10)
            status_response.raise_for_status()
            status_result = status_response.json()

            job_status = status_result.get('status')
            if job_status == "COMPLETED":
                current_app.logger.info(f"RunPod job {job_id} completed.")
                return status_result.get('output', {"error": "Job completed but output missing."}), 200
            elif job_status in ["FAILED", "CANCELLED"]:
                current_app.logger.error(f"RunPod job {job_id} failed or was cancelled. Status: {job_status}")
                error_details = status_result.get('output', {}).get('error', "Unknown error")
                return {"error": f"RunPod job failed: {error_details}", "details": status_result}, 500
            elif job_status in ["IN_QUEUE", "IN_PROGRESS"]:
                 current_app.logger.debug(f"RunPod job {job_id} status: {job_status}. Waiting...")
                 # Optional: Implement exponential backoff for poll_interval
                 poll_interval = min(poll_interval * 1.5, 15) # Example backoff
            else:
                current_app.logger.warning(f"RunPod job {job_id} has unexpected status: {job_status}")
                # Continue polling

        return {"error": "RunPod job timed out after {max_wait_seconds} seconds."}, 504 # Gateway Timeout

    except requests.exceptions.Timeout:
        current_app.logger.error(f"Request to RunPod endpoint {run_url} timed out.")
        return {"error": "Request to inference service timed out."}, 504
    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Failed to call RunPod endpoint: {e}")
        # Check if the response object exists and has content
        error_detail = "Network error or invalid response from service."
        if e.response is not None:
            try:
                error_detail = e.response.json() # Get JSON error if available
            except json.JSONDecodeError:
                error_detail = e.response.text # Get raw text otherwise
        return {"error": "Failed to communicate with inference service.", "details": str(error_detail)}, 502 # Bad Gateway

@app.route('/complete-deck', methods=['POST'])
def complete_deck_proxy():
    """Proxies the deck completion request to the RunPod endpoint."""
    try:
        data = request.get_json()
        if not data or 'deck_list' not in data:
            return jsonify({"error": "Missing 'deck_list' in request"}), 400

        deck_text = data['deck_list']
        format_input = data.get('format', DEFAULT_FORMAT).lower()
        if format_input not in ALLOWED_FORMATS:
             return jsonify({"error": f"Invalid format specified. Allowed: {', '.join(ALLOWED_FORMATS)}"}), 400

        # Basic validation using local parser before sending
        try:
            known_cards = parse_deck_input(deck_text)
            total_known = sum(c['count'] for c in known_cards)
            if total_known > DECK_SIZE:
                return jsonify({"error": f"Input deck has more than {DECK_SIZE} cards ({total_known})."}), 400
            if total_known == 0:
                return jsonify({"error": "Input deck cannot be empty."}), 400
        except ValueError as e:
             return jsonify({"error": f"Input Error: {str(e)}"}), 400

        # Prepare payload for RunPod endpoint
        runpod_payload = {
            "input": {
                "task": "complete_deck",
                "deck_list": deck_text,
                "format": format_input
            }
        }

        # Call the RunPod endpoint
        result, status_code = call_runpod_endpoint(runpod_payload)

        # Return the result from RunPod (or the error)
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.exception("An unexpected error occurred during deck completion proxy:")
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route('/complete-sideboard', methods=['POST'])
def complete_sideboard_proxy():
    """Proxies the sideboard completion request to the RunPod endpoint."""
    try:
        data = request.get_json()
        if not data or 'completed_deck' not in data:
            return jsonify({"error": "Missing 'completed_deck' in request body"}), 400
        if 'current_sideboard' not in data: # Expect text format from UI
             return jsonify({"error": "Missing 'current_sideboard' text in request body"}), 400

        main_deck_list = data['completed_deck']
        current_sideboard_text = data['current_sideboard']
        format_input = data.get('format', DEFAULT_FORMAT).lower()

        # --- Validation --- 
        if not isinstance(main_deck_list, list):
             return jsonify({"error": "'completed_deck' must be a list."}), 400
        main_deck_count = sum(c.get('count', 0) for c in main_deck_list)
        if main_deck_count != DECK_SIZE:
             # Be slightly more lenient here as the deck came from inference
             current_app.logger.warning(f"Received completed_deck with {main_deck_count} cards, expected {DECK_SIZE}")
             # return jsonify({"error": f"'completed_deck' must contain exactly {DECK_SIZE} cards."}), 400

        if format_input not in ALLOWED_FORMATS:
            return jsonify({"error": f"Invalid format specified. Allowed: {', '.join(ALLOWED_FORMATS)}"}), 400

        try:
            # Validate current sideboard text input
            current_sideboard_list = parse_deck_input(current_sideboard_text) if current_sideboard_text else []
            current_sb_count = sum(c.get('count', 0) for c in current_sideboard_list)
            if current_sb_count > SIDEBOARD_SIZE:
                return jsonify({"error": f"'current_sideboard' cannot contain more than {SIDEBOARD_SIZE} cards ({current_sb_count} found)."}), 400
        except ValueError as e:
             return jsonify({"error": f"Input Error (Sideboard): {str(e)}"}), 400
        # --- End Validation ---

        # Prepare payload for RunPod endpoint
        runpod_payload = {
            "input": {
                "task": "complete_sideboard",
                "completed_deck": main_deck_list, # Send the list directly
                "current_sideboard": current_sideboard_text, # Send the text
                "format": format_input
            }
        }

        # Call the RunPod endpoint
        result, status_code = call_runpod_endpoint(runpod_payload)

        # Return the result from RunPod (or the error)
        return jsonify(result), status_code

    except Exception as e:
        current_app.logger.exception("An unexpected error occurred during sideboard completion proxy:")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == '__main__':
    # Use waitress or gunicorn for production
    # Example: waitress-serve --host 0.0.0.0 --port 8080 deployment.app.app:app
    # For local dev:
    app.run(debug=False, host='0.0.0.0', port=8080) # Use a port like 8080 