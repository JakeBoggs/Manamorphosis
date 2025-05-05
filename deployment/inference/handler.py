import os
import re
import json
import requests
import pickle
import math
import time
import logging
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import runpod

# --- Configuration ---
MODEL_DIR = "/models"
DATA_DIR = "/data"
DIFFUSION_MODEL_PATH = os.path.join(MODEL_DIR, "diffusion_model.pth")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "card_classifier.pt")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "card_embeddings.pkl")
ATOMIC_CARDS_PATH = os.path.join(DATA_DIR, "AtomicCards.json")

# Google Drive Folder ID containing the models and data
GDRIVE_FOLDER_ID = "1ZvVbUGXa8FGzL97lplQGea2Ech7yfR-0"

SCRYFALL_API_BASE = "https://api.scryfall.com"

# Model & Inference Constants
EMB_DIM = 128
DECK_SIZE = 60
SIDEBOARD_SIZE = 15
TIMESTEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ALLOWED_FORMATS = {'standard', 'pioneer', 'modern', 'legacy', 'vintage', 'pauper'}
DEFAULT_FORMAT = 'modern'

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def cosine_beta_schedule(T, s=0.008):
    """Cosine variance schedule"""
    steps = torch.linspace(0, T, T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0, 0.999).float() # Ensure float32 output

def sinusoidal_embedding(t: torch.Tensor, dim: int = EMB_DIM):
    """Sinusoidal time embedding"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * freqs[None]
    emb = torch.cat((args.sin(), args.cos()), dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

# --- Model Definitions ---
class DiffusionModel(nn.Module):
    # Identical to the one in app.py
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # Store config
        model_dim = cfg["model_dim"]
        nhead = cfg["heads"]
        dim_feedforward = cfg["dim_feedforward"]
        num_layers = cfg["layers"]
        sb_num_layers = cfg["sb_layers"]
        activation = "gelu"
        batch_first = True

        # Time MLPs (Main separate, SB shared)
        ff_dim = cfg["dim_feedforward"]
        self.main_time_mlp = nn.Sequential(
            nn.Linear(EMB_DIM, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, EMB_DIM),
        )
        self.sb_time_mlp = nn.Sequential(
            nn.Linear(EMB_DIM, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, EMB_DIM),
        )

        # Mask MLPs (Main separate, SB Decoder only)
        self.main_mask_mlp = nn.Sequential(
            nn.Linear(1, EMB_DIM),
            nn.SiLU(),
            nn.Linear(EMB_DIM, EMB_DIM),
        )
        self.sb_mask_mlp = nn.Sequential(
            nn.Linear(1, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, EMB_DIM),
        )

        # --- Main Deck Encoder ---
        self.main_input_proj = nn.Linear(EMB_DIM, model_dim)
        main_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=batch_first
        )
        self.main_transformer_encoder = nn.TransformerEncoder(
            main_encoder_layer,
            num_layers=num_layers
        )
        self.main_output_proj = nn.Linear(model_dim, EMB_DIM)

        # --- Sideboard Context Encoder ---
        self.sb_context_input_proj = nn.Linear(EMB_DIM, model_dim)
        sb_context_encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=batch_first
        )
        self.sideboard_context_encoder = nn.TransformerEncoder(
            sb_context_encoder_layer,
            num_layers=1 # Fixed at 1 in original code
        )

        # --- Sideboard Decoder ---
        self.sb_input_proj = nn.Linear(EMB_DIM, model_dim)
        sb_decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=batch_first
        )
        self.sb_transformer_decoder = nn.TransformerDecoder(
            sb_decoder_layer,
            num_layers=1 # Fixed at 1 in original code
        )
        # Reusing sb_context_encoder_layer definition for the output encoder
        self.sb_transformer_output = nn.TransformerEncoder(
             nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                batch_first=batch_first
            ),
            num_layers=sb_num_layers
        )
        self.sb_output_proj = nn.Linear(model_dim, EMB_DIM)


    def predict_main_noise(self, x_t, t, mask):
        """Predicts noise added to the main deck."""
        sin_emb = sinusoidal_embedding(t, EMB_DIM)
        main_t_emb_flat = self.main_time_mlp(sin_emb)
        main_t_emb = main_t_emb_flat[:, None, :].expand(-1, DECK_SIZE, -1)
        main_mask_emb = self.main_mask_mlp(mask)
        h_main = x_t + main_t_emb + main_mask_emb
        h_main_proj = self.main_input_proj(h_main)
        main_encoded = self.main_transformer_encoder(h_main_proj)
        main_noise_pred = self.main_output_proj(main_encoded)
        return main_noise_pred

    def encode_main_deck_context(self, x0):
        """Encodes the main deck (x0) to be used as context for sideboard generation."""
        h_sb_context_proj = self.sb_context_input_proj(x0)
        sb_context_encoded = self.sideboard_context_encoder(h_sb_context_proj)
        return sb_context_encoded

    def predict_sideboard_noise(self, sb_x_t, t, sb_mask, main_deck_context_encoded):
        """Predicts noise added to the sideboard, conditioned on main deck context."""
        sin_emb = sinusoidal_embedding(t, EMB_DIM)
        sb_decoder_t_emb_flat = self.sb_time_mlp(sin_emb)
        sb_decoder_t_emb = sb_decoder_t_emb_flat[:, None, :].expand(-1, SIDEBOARD_SIZE, -1)
        sb_decoder_mask_emb = self.sb_mask_mlp(sb_mask)
        h_sb = sb_x_t + sb_decoder_t_emb + sb_decoder_mask_emb
        h_sb_proj = self.sb_input_proj(h_sb)
        sb_decoded = self.sb_transformer_decoder(tgt=h_sb_proj, memory=main_deck_context_encoded)
        sb_decoded = self.sb_transformer_output(sb_decoded) # Pass through output transformer
        sb_noise_pred = self.sb_output_proj(sb_decoded)
        return sb_noise_pred


class CardClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CardClassifier, self).__init__()
        self.network = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.network(x)

# --- Global Variables for Models and Data ---
diffusion_model = None
clf_model = None
card_embeddings = None
idx_to_card = None
diffusion_beta = None
diffusion_alpha = None
diffusion_alpha_bar = None
cards = None # Loaded from AtomicCards.json

# --- Model/Data Loading ---
def load_models_and_data():
    global diffusion_model, clf_model, card_embeddings, idx_to_card
    global diffusion_beta, diffusion_alpha, diffusion_alpha_bar
    global cards

    logging.info("Loading models and data...")

    # Load AtomicCards data (needed for legality checks)
    if not os.path.exists(ATOMIC_CARDS_PATH):
        raise FileNotFoundError(f"AtomicCards JSON file not found: {ATOMIC_CARDS_PATH}")
    try:
        with open(ATOMIC_CARDS_PATH, 'r', encoding='utf-8') as f:
            cards_data = json.load(f)
            # Check if the structure is {'data': { 'cardName': [versions...], ...}}
            if isinstance(cards_data.get('data'), dict):
                 cards = cards_data['data']
                 logging.info(f"Loaded {len(cards)} card entries from AtomicCards.json")
            else:
                 raise ValueError("AtomicCards.json does not have the expected {'data': { 'cardName': [...] }} structure.")

    except Exception as e:
        logging.error(f"Error loading or parsing AtomicCards.json: {e}")
        raise


    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, "rb") as f:
        card_embeddings = pickle.load(f)
    logging.info(f"Loaded {len(card_embeddings)} card embeddings.")

    if not os.path.exists(DIFFUSION_MODEL_PATH):
        raise FileNotFoundError(f"Diffusion model checkpoint not found: {DIFFUSION_MODEL_PATH}")
    try:
        logging.info(f"Loading diffusion model from {DIFFUSION_MODEL_PATH}...")
        diff_ckpt = torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE)

        # --- Compatibility Check ---
        if "model" not in diff_ckpt or "config" not in diff_ckpt:
             logging.error("Diffusion checkpoint missing 'model' or 'config' key.")
             # Attempt loading legacy format if necessary, or raise error
             if isinstance(diff_ckpt, dict) and not ("model" in diff_ckpt or "config" in diff_ckpt):
                 logging.warning("Attempting to load diffusion model state dict directly (legacy format?).")
                 model_state_dict = diff_ckpt
                 # Need a default config or load from elsewhere if possible
                 # This is risky - define a default based on known training params
                 diff_cfg = {
                    "model_dim": 256, "heads": 4, "dim_feedforward": 1024,
                    "layers": 6, "sb_layers": 2 # Example defaults - ADJUST AS NEEDED
                 }
                 logging.warning(f"Using default diffusion config: {diff_cfg}")
             else:
                raise KeyError("Checkpoint structure incompatible. Expected keys 'model' and 'config'.")
        else:
            model_state_dict = diff_ckpt["model"]
            diff_cfg = diff_ckpt.get("config")
            if not diff_cfg:
                 raise ValueError("Diffusion checkpoint 'config' is missing or empty.")
            logging.info(f"Loaded diffusion model config: {diff_cfg}")


        diffusion_model = DiffusionModel(diff_cfg).to(DEVICE)

        # Handle potential prefix mismatch (e.g., '_orig_mod.') from torch.compile
        # or DataParallel ('module.')
        model_state_dict = {k.replace('_orig_mod.', ''): v for k, v in model_state_dict.items()}
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}

        missing_keys, unexpected_keys = diffusion_model.load_state_dict(model_state_dict, strict=False) # Use strict=False initially for debugging
        if missing_keys:
             logging.warning(f"Warning: Missing keys in diffusion model state_dict: {missing_keys}")
        if unexpected_keys:
             logging.warning(f"Warning: Unexpected keys in diffusion model state_dict: {unexpected_keys}")
        # If warnings are acceptable, proceed. Otherwise, consider strict=True and debugging.

        diffusion_model.eval()
        diffusion_beta = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
        diffusion_alpha = 1.0 - diffusion_beta
        diffusion_alpha_bar = torch.cumprod(diffusion_alpha, dim=0)
        logging.info("Diffusion model loaded.")
    except KeyError as e:
        logging.error(f"Error: Key missing in diffusion model checkpoint: {e}. Checkpoint structure might be incompatible.")
        raise
    except Exception as e:
        logging.error(f"Error loading diffusion model: {e}")
        raise

    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Classifier model checkpoint not found: {CLASSIFIER_PATH}")
    try:
        logging.info(f"Loading classifier model from {CLASSIFIER_PATH}...")
        clf_ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE)

        # Compatibility check
        required_keys = ["model_state_dict", "embedding_dim", "num_classes", "idx_to_card"]
        if not all(key in clf_ckpt for key in required_keys):
            raise KeyError(f"Classifier checkpoint missing one or more required keys: {required_keys}. Found: {list(clf_ckpt.keys())}")

        clf_model = CardClassifier(clf_ckpt["embedding_dim"], clf_ckpt["num_classes"]).to(DEVICE)

        clf_state_dict = clf_ckpt["model_state_dict"]
        # Handle potential prefix mismatch
        clf_state_dict = {k.replace('_orig_mod.', ''): v for k, v in clf_state_dict.items()}
        clf_state_dict = {k.replace('module.', ''): v for k, v in clf_state_dict.items()}

        clf_model.load_state_dict(clf_state_dict, strict=True)
        clf_model.eval()
        idx_to_card = clf_ckpt["idx_to_card"]
        # Convert keys back to int if they were saved as strings in JSON-like structures
        idx_to_card = {int(k): v for k, v in idx_to_card.items()}
        logging.info("Classifier model loaded.")
    except KeyError as e:
        logging.error(f"Error: Key missing in classifier model checkpoint: {e}. Checkpoint structure might be different.")
        raise
    except Exception as e:
        logging.error(f"Error loading classifier model: {e}")
        raise

    logging.info("Models and data loaded successfully.")

# --- Text Parsing ---
def parse_deck_input(deck_text):
    """Parses the input text area format into a list of card dicts."""
    parsed_cards = []
    lines = deck_text.strip().split('\n')
    pattern = re.compile(r"^\s*(\d+)\s*[xX]?\s*(.+?)\s*$")
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'): # Skip comments/empty
            continue
        match = pattern.match(line)
        if match:
            count = int(match.group(1))
            name = match.group(2).strip()
            if count > 0 and name:
                if name not in card_embeddings:
                     raise ValueError(f"Card not found in embeddings: '{name}'. Please check spelling.")
                parsed_cards.append({"name": name, "count": count})
        else:
            # Handle lines that might just be card names (assume count 1)
            name = line.strip()
            if name:
                 if name not in card_embeddings:
                     raise ValueError(f"Card not found in embeddings: '{name}'. Please check spelling.")
                 parsed_cards.append({"name": name, "count": 1})

    # Combine duplicate entries
    card_counts = Counter()
    for card in parsed_cards:
        card_counts[card['name']] += card['count'] # Use += to aggregate duplicates

    return [{'name': name, 'count': count} for name, count in card_counts.items()]

# --- Scryfall Image Fetching ---
image_cache = {}

def get_card_image_urls(card_names):
    """Fetches image URLs from Scryfall for a list of card names."""
    urls = {}
    names_to_fetch = set()
    name_map = {} # Maps the part sent to Scryfall back to the original full name

    for name in card_names:
        if name not in image_cache:
            # Scryfall fuzzy search works better with just the first part of DFCs
            scryfall_query_name = name.split("//")[0].strip()
            names_to_fetch.add(scryfall_query_name)
            # Map the query name back to the potentially full original name
            if scryfall_query_name not in name_map:
                name_map[scryfall_query_name] = []
            if name not in name_map[scryfall_query_name]: # Avoid duplicate full names if query name is same
                name_map[scryfall_query_name].append(name)
        else:
             # Use original full name for cache lookup consistency
            urls[name] = image_cache[name]

    if not names_to_fetch:
        return urls

    logging.info(f"Fetching {len(names_to_fetch)} unique card names/parts from Scryfall...")

    # Scryfall collection endpoint is efficient for multiple lookups
    identifiers = [{"name": name} for name in names_to_fetch]
    payload = {"identifiers": identifiers}

    try:
        response = requests.post(f"{SCRYFALL_API_BASE}/cards/collection", json=payload, timeout=20) # Added timeout
        response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
        data = response.json()

        # Process results for found cards
        if data and 'data' in data:
            for card_data in data['data']:
                # The name returned by Scryfall might be slightly different, use the name from the request mapping
                # Find which requested query name this result corresponds to (fuzzy match might be needed if names differ significantly)
                # For simplicity, assume Scryfall returns the name used in the identifier if found.
                found_query_name = card_data.get('name')
                if "//" in found_query_name: # Scryfall might return the full name even if we sent part
                    found_query_name = found_query_name.split("//")[0].strip()

                original_full_names = name_map.get(found_query_name)
                if not original_full_names:
                     logging.warning(f"Scryfall returned card '{found_query_name}', but couldn't map it back to an original requested name.")
                     continue # Skip if we can't map it back


                image_info = None
                # Check for multi-face cards first
                if card_data.get('card_faces') and len(card_data['card_faces']) > 1:
                    face1 = card_data['card_faces'][0]
                    face2 = card_data['card_faces'][1]
                    url1 = face1.get('image_uris', {}).get('normal')
                    url2 = face2.get('image_uris', {}).get('normal')
                    if url1 or url2: # Store dict if either face has image
                        image_info = {'front': url1, 'back': url2}
                    # Fallback: check top-level if faces have no images (less common)
                    elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                        image_info = card_data['image_uris']['normal']
                # Single-faced card
                elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                    image_info = card_data['image_uris']['normal']

                # Apply the found image info to all original full names that mapped to this query name
                for original_name in original_full_names:
                    if image_info is not None:
                        image_cache[original_name] = image_info
                        urls[original_name] = image_info
                        logging.debug(f"Cached image for '{original_name}'")
                    else:
                        # Explicitly mark as None if found but no image URL available
                        image_cache[original_name] = None
                        urls[original_name] = None
                        logging.warning(f"Card '{original_name}' found by Scryfall but no suitable image URI.")


        # Handle cards Scryfall explicitly said were not found
        if 'not_found' in data:
            for not_found_identifier in data['not_found']:
                if 'name' in not_found_identifier:
                    missing_query_name = not_found_identifier['name']
                    original_full_names = name_map.get(missing_query_name)
                    if original_full_names:
                        for original_name in original_full_names:
                            # Mark as not found (None) in cache and results if not already found
                            if original_name not in urls:
                                image_cache[original_name] = None
                                urls[original_name] = None
                                logging.warning(f"Card '{original_name}' (queried as '{missing_query_name}') not found by Scryfall.")


    except requests.exceptions.Timeout:
        logging.error("Scryfall API request timed out.")
        # Mark all names in this batch as failed (None)
        for query_name, original_names in name_map.items():
            for original_name in original_names:
                 if original_name not in urls: # Avoid overwriting already cached data
                    image_cache[original_name] = None
                    urls[original_name] = None
    except requests.exceptions.RequestException as e:
        logging.error(f"Scryfall API request failed: {e}")
        for query_name, original_names in name_map.items():
            for original_name in original_names:
                 if original_name not in urls:
                    image_cache[original_name] = None
                    urls[original_name] = None
    except json.JSONDecodeError as e:
         logging.error(f"Failed to decode Scryfall JSON response: {e}")
         for query_name, original_names in name_map.items():
            for original_name in original_names:
                 if original_name not in urls:
                    image_cache[original_name] = None
                    urls[original_name] = None

    # Ensure all initially requested names have an entry in the final dict (even if None)
    for name in card_names:
        if name not in urls:
            urls[name] = image_cache.get(name, None) # Use cache value if exists, else None

    return urls


# --- Inference Functions ---
@torch.no_grad()
def run_inference(known_cards_list, format):
    """Runs diffusion model inference to complete the main deck."""
    logging.info(f"Running main deck inference for known_cards: {len(known_cards_list)} types, format: {format}")

    if diffusion_model is None or clf_model is None or card_embeddings is None or idx_to_card is None:
        raise RuntimeError("Models or data not loaded properly for inference.")

    MAX_REFINEMENT_ITERATIONS = 3

    # 1. Prepare Initial known main deck embeddings and mask
    initial_known_emb = torch.zeros(1, DECK_SIZE, EMB_DIM, device=DEVICE)
    initial_known_mask = torch.zeros(1, DECK_SIZE, 1, device=DEVICE)
    original_known_names = []
    current_idx = 0
    total_known_count = 0

    for card_info in known_cards_list:
        name = card_info["name"]
        count = card_info["count"]
        total_known_count += count
        try:
            vec = torch.tensor(card_embeddings[name], dtype=torch.float32, device=DEVICE)
        except KeyError:
            raise ValueError(f"Card '{name}' embedding not found.")

        for _ in range(count):
            if current_idx < DECK_SIZE:
                initial_known_emb[0, current_idx] = vec
                initial_known_mask[0, current_idx] = 1.0
                original_known_names.append(name)
                current_idx += 1
            else:
                logging.warning(f"Input deck exceeds {DECK_SIZE} cards. Truncating.")
                total_known_count = DECK_SIZE
                break
        if current_idx >= DECK_SIZE:
            break

    num_unknown_initial = DECK_SIZE - total_known_count
    if num_unknown_initial < 0:
         logging.warning(f"Known main deck cards ({total_known_count}) exceeded DECK_SIZE ({DECK_SIZE}). Assuming full deck provided.")
         num_unknown_initial = 0

    logging.info(f"Prepared initial known main deck embeddings for {total_known_count} cards. Initially generating {num_unknown_initial} cards.")

    # --- Iterative Refinement ---
    current_x0_main = None
    current_mask = initial_known_mask.clone()
    current_known_emb = initial_known_emb.clone()

    for refinement_iter in range(MAX_REFINEMENT_ITERATIONS + 1):
        logging.info(f"--- Starting Main Deck Generation/Refinement Iteration {refinement_iter} ---")

        unknown_mask_flat_this_iter = (current_mask[0, :, 0] == 0)
        num_unknown_this_iter = int(unknown_mask_flat_this_iter.sum().item())

        if num_unknown_this_iter == 0 and refinement_iter > 0:
             logging.info(f"Iteration {refinement_iter}: No unknown slots left to refine. Stopping.")
             break
        elif num_unknown_initial == 0 and refinement_iter == 0:
             logging.info("Initial deck was full, skipping refinement loop.")
             current_x0_main = initial_known_emb.clone()
             break

        # 2. Initialize Noise
        x = torch.randn(1, DECK_SIZE, EMB_DIM, device=DEVICE)
        x = current_mask * current_known_emb + (1 - current_mask) * x

        # 3. Run Main Deck Diffusion Sampling Loop
        for t in reversed(range(TIMESTEPS)):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            main_noise_pred = diffusion_model.predict_main_noise(x, t_tensor, current_mask)
            beta_t = diffusion_beta[t] # Already on device
            alpha_t = diffusion_alpha[t]
            alpha_bar_t = diffusion_alpha_bar[t]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()
            mean_main = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * main_noise_pred)
            if t > 0:
                noise_main = torch.randn_like(x)
                x_next = mean_main + noise_main * beta_t.sqrt()
            else:
                x_next = mean_main
            x = current_mask * current_known_emb + (1 - current_mask) * x_next

        current_x0_main = x

        # --- Check Limits and Prepare for Next Iteration ---
        if refinement_iter < MAX_REFINEMENT_ITERATIONS:
            logging.info(f"Iteration {refinement_iter}: Checking 4-copy and format legality...")
            iter_all_logits_main = clf_model(current_x0_main[0])
            iter_all_predicted_indices = torch.argmax(iter_all_logits_main, dim=1).cpu().numpy()
            iter_all_predicted_names = []
            name_idx_map = {}
            for i, idx in enumerate(iter_all_predicted_indices):
                name = idx_to_card.get(int(idx)) # Use int() just in case
                if name is not None:
                    iter_all_predicted_names.append(name)
                    name_idx_map[i] = name

            iter_card_counts = Counter(iter_all_predicted_names)
            indices_to_force_regenerate_4_copy = set()
            indices_to_force_regenerate_format = set()

            # Check 4-copy limit
            for card_name, count in iter_card_counts.items():
                card_data_list = cards.get(card_name)
                if not card_data_list: continue
                card = card_data_list[0] # Use first version for supertypes
                supertypes = card.get("supertypes", [])

                if count > 4 and "Basic" not in supertypes:
                    logging.warning(f"Iteration {refinement_iter}: Card '{card_name}' found {count} times. Marking {count - 4} generated copies for regeneration.")
                    current_indices = [i for i, name in name_idx_map.items() if name == card_name]
                    # Only mark slots that were *not* part of the initial input
                    generated_indices_for_card = {i for i in current_indices if initial_known_mask[0, i, 0] == 0.0}
                    num_to_replace = count - 4
                    indices_to_force_regenerate_4_copy.update(list(generated_indices_for_card)[:num_to_replace])

            # Check Format Legality (only for generated cards)
            for abs_idx, card_name in name_idx_map.items():
                if initial_known_mask[0, abs_idx, 0] == 0.0: # Was this generated?
                    card_data_list = cards.get(card_name)
                    if card_data_list:
                        card_data = card_data_list[0] # Use first version for legality check
                        legalities = card_data.get("legalities", {})
                        supertypes = card_data.get("supertypes", [])
                        # Check legality OR basic land status
                        is_legal = legalities.get(format, "not_legal").lower() in ("legal", "restricted")
                        is_basic = "Basic" in supertypes

                        if not is_legal and not is_basic:
                            logging.warning(f"Iteration {refinement_iter}: Generated card '{card_name}' (Index: {abs_idx}) is not legal in {format} and not basic. Marking for regeneration.")
                            indices_to_force_regenerate_format.add(abs_idx)
                    else:
                         logging.warning(f"Iteration {refinement_iter}: Could not find data for generated card '{card_name}' (Index: {abs_idx}) to check format legality.")

            final_absolute_indices_to_regenerate = indices_to_force_regenerate_4_copy.union(indices_to_force_regenerate_format)
            final_absolute_indices_to_regenerate_list = sorted(list(final_absolute_indices_to_regenerate))

            if not final_absolute_indices_to_regenerate_list and num_unknown_this_iter > 0 :
                logging.info(f"Iteration {refinement_iter}: All generated cards meet constraints. Stopping refinement.")
                break
            elif num_unknown_this_iter == 0:
                pass # Loop should end naturally
            elif not final_absolute_indices_to_regenerate_list:
                 logging.info(f"Iteration {refinement_iter}: No slots marked for regeneration. Stopping refinement.")
                 break
            else:
                 logging.warning(f"Iteration {refinement_iter}: Preparing to regenerate {len(final_absolute_indices_to_regenerate_list)} main deck slots. Indices: {final_absolute_indices_to_regenerate_list}")
                 # Log reasons
                 for abs_idx in final_absolute_indices_to_regenerate_list:
                    temp_logits = clf_model(current_x0_main[0, abs_idx].unsqueeze(0))
                    temp_pred_idx = torch.argmax(temp_logits, dim=1).item()
                    temp_name = idx_to_card.get(temp_pred_idx, "Unknown Index")
                    reason = "Unknown"
                    is_4_copy_violator = abs_idx in indices_to_force_regenerate_4_copy
                    is_format_violator = abs_idx in indices_to_force_regenerate_format
                    if is_4_copy_violator and is_format_violator: reason = "4-Copy & Format"
                    elif is_4_copy_violator: reason = "4-Copy Rule"
                    elif is_format_violator: reason = "Format Legality"
                    logging.warning(f"  - Reconsidering Main Deck Slot {abs_idx}: '{temp_name}' - Reason: {reason}")


                 # Prepare mask and known embeddings for the *next* iteration
                 next_mask = initial_known_mask.clone()
                 next_known_emb = initial_known_emb.clone()
                 all_generated_indices_ever = set(torch.where(initial_known_mask[0, :, 0] == 0)[0].cpu().numpy())
                 valid_copy_generated_indices = list(all_generated_indices_ever - final_absolute_indices_to_regenerate)

                 if valid_copy_generated_indices:
                     next_mask[0, valid_copy_generated_indices, 0] = 1.0
                     next_known_emb[0, valid_copy_generated_indices] = current_x0_main[0, valid_copy_generated_indices]

                 current_mask = next_mask
                 current_known_emb = next_known_emb
                 num_known_next = int(current_mask.sum().item())
                 logging.info(f"Preparing for Iteration {refinement_iter + 1}. Known cards: {num_known_next}, Regenerating: {DECK_SIZE - num_known_next}")

        else:
             logging.info(f"Iteration {refinement_iter}: Refinement loop finished (max iterations reached or stopped early).")

    # --- Final Classification and Formatting ---
    if current_x0_main is None:
        if num_unknown_initial == 0:
            current_x0_main = initial_known_emb.clone()
            logging.info("Using initial known embeddings as final result (deck was full).")
        else:
            logging.error("Error: current_x0_main is None but deck was not initially full.")
            return []

    final_x0_main = current_x0_main
    final_unknown_mask_flat = (initial_known_mask[0, :, 0] == 0)
    generated_main_names = []

    if final_unknown_mask_flat.sum() > 0:
        final_unknown_embeddings = final_x0_main[0][final_unknown_mask_flat]
        logging.info(f"Classifying {final_unknown_embeddings.shape[0]} final generated main deck embeddings...")
        final_logits_main = clf_model(final_unknown_embeddings)
        final_predicted_indices = torch.argmax(final_logits_main, dim=1).cpu().numpy()
        temp_generated_names = []
        for idx in final_predicted_indices:
            name = idx_to_card.get(int(idx))
            if name is not None:
                temp_generated_names.append(name)
            else:
                logging.warning(f"Classifier predicted unknown index {int(idx)} in final main deck. Replacing with 'Error Card Main'.")
                temp_generated_names.append("Error Card Main")
        generated_main_names = temp_generated_names
    else:
        logging.info("No main deck cards were generated (initial deck was full).")

    completed_deck_names = original_known_names + generated_main_names
    if len(completed_deck_names) != DECK_SIZE:
         logging.error(f"Final main deck construction error: Expected {DECK_SIZE} cards, got {len(completed_deck_names)}")
         if len(completed_deck_names) < DECK_SIZE: completed_deck_names.extend(["Error Card Main"] * (DECK_SIZE - len(completed_deck_names)))
         else: completed_deck_names = completed_deck_names[:DECK_SIZE]

    final_main_counts = Counter(completed_deck_names)
    main_unique_names = list(final_main_counts.keys())
    image_urls = get_card_image_urls(main_unique_names)

    completed_deck_list = []
    for name, count in final_main_counts.items():
        if name == "Error Card Main":
             img_url = None
             logging.error("Final deck contains 'Error Card Main' placeholder.")
        else:
             img_url = image_urls.get(name) # Fetch URL, could be string or dict
        completed_deck_list.append({"name": name, "count": count, "image_url": img_url})


    final_count = sum(c['count'] for c in completed_deck_list)
    logging.info(f"Main deck inference complete. Final count: {final_count}.")
    if final_count != DECK_SIZE:
        logging.warning(f"Main deck final count ({final_count}) does not match DECK_SIZE ({DECK_SIZE}).")


    return completed_deck_list


@torch.no_grad()
def complete_sideboard_inference(main_deck_list, current_sideboard_list, format):
    """Completes a sideboard based on a provided main deck and current sideboard cards."""
    logging.info(f"Running sideboard completion for {len(current_sideboard_list)} initial types, format: {format}")

    if diffusion_model is None or clf_model is None or card_embeddings is None or idx_to_card is None:
        raise RuntimeError("Models or data not loaded properly for sideboard inference.")

    if SIDEBOARD_SIZE <= 0:
        logging.info("SIDEBOARD_SIZE is 0 or less, returning empty sideboard.")
        return []

    MAX_REFINEMENT_ITERATIONS = 3

    # 1. Reconstruct main deck embedding tensor and counts
    main_deck_embeddings = torch.zeros(1, DECK_SIZE, EMB_DIM, device=DEVICE)
    main_current_idx = 0
    main_total_cards = 0
    main_deck_counts = Counter()
    for card_info in main_deck_list:
        name = card_info["name"]
        count = card_info["count"]
        main_total_cards += count
        main_deck_counts[name] += count
        try: vec = torch.tensor(card_embeddings[name], dtype=torch.float32, device=DEVICE)
        except KeyError: raise ValueError(f"Card '{name}' from input main deck not found in embeddings.")
        for _ in range(count):
            if main_current_idx < DECK_SIZE: main_deck_embeddings[0, main_current_idx] = vec; main_current_idx += 1
            else: raise ValueError("Provided main deck list exceeds DECK_SIZE.")
    if main_total_cards != DECK_SIZE:
        # Allow slight deviation if main deck had errors, but log warning
        logging.warning(f"Input main deck for sideboard generation has {main_total_cards} cards, expected {DECK_SIZE}.")
        # Pad or truncate main_deck_embeddings if necessary? For now, assume it's close enough.

    logging.info(f"Reconstructed main deck embedding tensor ({main_current_idx} filled) and counts for sideboard context.")

    # 2. Prepare Initial Known Sideboard Embeddings and Mask
    initial_sb_known_emb = torch.zeros(1, SIDEBOARD_SIZE, EMB_DIM, device=DEVICE)
    initial_sb_known_mask = torch.zeros(1, SIDEBOARD_SIZE, 1, device=DEVICE)
    original_known_sb_names = []
    sb_current_idx = 0
    total_known_sb_count = 0

    for card_info in current_sideboard_list:
        name = card_info["name"]
        count = card_info["count"]
        total_known_sb_count += count
        try:
            vec = torch.tensor(card_embeddings[name], dtype=torch.float32, device=DEVICE)
        except KeyError:
            raise ValueError(f"Sideboard card '{name}' embedding not found.")
        for _ in range(count):
            if sb_current_idx < SIDEBOARD_SIZE:
                initial_sb_known_emb[0, sb_current_idx] = vec
                initial_sb_known_mask[0, sb_current_idx] = 1.0
                original_known_sb_names.append(name)
                sb_current_idx += 1
            else:
                logging.warning(f"Input sideboard exceeds {SIDEBOARD_SIZE} cards. Truncating.")
                total_known_sb_count = SIDEBOARD_SIZE
                break
        if sb_current_idx >= SIDEBOARD_SIZE:
            break

    num_unknown_initial = SIDEBOARD_SIZE - total_known_sb_count
    if num_unknown_initial < 0:
        logging.warning(f"Known sideboard cards ({total_known_sb_count}) exceeded SIDEBOARD_SIZE ({SIDEBOARD_SIZE}). Assuming full sideboard provided.")
        num_unknown_initial = 0
    elif num_unknown_initial == 0 and SIDEBOARD_SIZE > 0:
        logging.info("Initial sideboard is already full. Skipping generation.")
        initial_sb_counts = Counter(original_known_sb_names)
        initial_sb_unique_names = list(initial_sb_counts.keys())
        image_urls = get_card_image_urls(initial_sb_unique_names)
        completed_sideboard_list = []
        for name, count in initial_sb_counts.items():
             img_url = image_urls.get(name)
             completed_sideboard_list.append({"name": name, "count": count, "image_url": img_url})
        # Ensure exactly SIDEBOARD_SIZE? The input parsing should handle this.
        return completed_sideboard_list[:SIDEBOARD_SIZE] # Truncate just in case


    logging.info(f"Prepared initial known sideboard embeddings for {total_known_sb_count} cards. Initially generating {num_unknown_initial} cards.")

    # 3. Pre-calculate Main Deck Context Encoding
    sb_context_encoded = diffusion_model.encode_main_deck_context(main_deck_embeddings)
    logging.info("Calculated sideboard context encoding from main deck.")

    # --- Iterative Refinement ---
    current_x0_sb = None
    current_mask = initial_sb_known_mask.clone()
    current_known_emb = initial_sb_known_emb.clone()

    for refinement_iter in range(MAX_REFINEMENT_ITERATIONS + 1):
        logging.info(f"--- Starting Sideboard Generation/Refinement Iteration {refinement_iter} ---")

        unknown_mask_flat_this_iter = (current_mask[0, :, 0] == 0)
        num_unknown_this_iter = int(unknown_mask_flat_this_iter.sum().item())

        if num_unknown_this_iter == 0 and refinement_iter > 0:
             logging.info(f"Iteration {refinement_iter}: No unknown sideboard slots left to refine. Stopping.")
             break
        elif num_unknown_initial == 0 and refinement_iter == 0:
             logging.info("Initial sideboard was full, skipping refinement loop.")
             # This case should have returned earlier, but break defensively
             break


        # 4. Initialize Noise
        sb_x = torch.randn(1, SIDEBOARD_SIZE, EMB_DIM, device=DEVICE)
        sb_x = current_mask * current_known_emb + (1 - current_mask) * sb_x

        # 5. Run Sideboard Diffusion Sampling Loop
        for t in reversed(range(TIMESTEPS)):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)
            sb_noise_pred = diffusion_model.predict_sideboard_noise(sb_x, t_tensor, current_mask, sb_context_encoded)
            beta_t = diffusion_beta[t]
            alpha_t = diffusion_alpha[t]
            alpha_bar_t = diffusion_alpha_bar[t]
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()
            mean_sb = (1.0 / sqrt_alpha_t) * (sb_x - (beta_t / sqrt_one_minus_alpha_bar_t) * sb_noise_pred)
            if t > 0:
                noise_sb = torch.randn_like(sb_x)
                sb_x_next = mean_sb + noise_sb * beta_t.sqrt()
            else:
                sb_x_next = mean_sb
            sb_x = current_mask * current_known_emb + (1 - current_mask) * sb_x_next

        current_x0_sb = sb_x

        # --- Check Limits and Prepare for Next Iteration ---
        if refinement_iter < MAX_REFINEMENT_ITERATIONS:
            logging.info(f"Iteration {refinement_iter}: Checking 4-copy and format legality across main+sideboard...")
            iter_all_sb_logits = clf_model(current_x0_sb[0])
            iter_all_sb_predicted_indices = torch.argmax(iter_all_sb_logits, dim=1).cpu().numpy()
            iter_all_sb_predicted_names = []
            sb_name_idx_map = {}
            for i, idx in enumerate(iter_all_sb_predicted_indices):
                name = idx_to_card.get(int(idx))
                if name is not None:
                    iter_all_sb_predicted_names.append(name)
                    sb_name_idx_map[i] = name

            iter_sb_counts = Counter(iter_all_sb_predicted_names)
            combined_counts = main_deck_counts.copy()
            combined_counts.update(iter_sb_counts)

            indices_to_force_regenerate_4_copy = set()
            indices_to_force_regenerate_format = set()

            # Check 4-copy limit
            for card_name, combined_count in combined_counts.items():
                card_data_list = cards.get(card_name)
                if not card_data_list: continue
                card = card_data_list[0]
                supertypes = card.get("supertypes", [])

                if combined_count > 4 and "Basic" not in supertypes:
                    sb_count_for_card = iter_sb_counts.get(card_name, 0)
                    num_to_remove_total = combined_count - 4
                    if sb_count_for_card > 0 and num_to_remove_total > 0:
                        logging.warning(f"Iteration {refinement_iter}: Card '{card_name}' found {combined_count} times (Main: {main_deck_counts.get(card_name, 0)}, SB: {sb_count_for_card}). Marking {min(sb_count_for_card, num_to_remove_total)} generated SB copies for regeneration.")
                        current_sb_indices = [i for i, name in sb_name_idx_map.items() if name == card_name]
                        # Only mark slots that were *not* part of the initial SB input
                        generated_sb_indices_for_card = {i for i in current_sb_indices if initial_sb_known_mask[0, i, 0] == 0.0}
                        num_to_regenerate = min(len(generated_sb_indices_for_card), num_to_remove_total)
                        indices_to_force_regenerate_4_copy.update(list(generated_sb_indices_for_card)[:num_to_regenerate])

            # Check Format Legality (only for generated sideboard cards)
            for abs_idx, card_name in sb_name_idx_map.items():
                if initial_sb_known_mask[0, abs_idx, 0] == 0.0: # Was this generated?
                    card_data_list = cards.get(card_name)
                    if card_data_list:
                        card_data = card_data_list[0]
                        legalities = card_data.get("legalities", {})
                        supertypes = card_data.get("supertypes", [])
                        is_legal = legalities.get(format, "not_legal").lower() in ("legal", "restricted")
                        is_basic = "Basic" in supertypes
                        if not is_legal and not is_basic:
                            logging.warning(f"Iteration {refinement_iter}: Generated sideboard card '{card_name}' (Index: {abs_idx}) is not legal in {format} and not basic. Marking for regeneration.")
                            indices_to_force_regenerate_format.add(abs_idx)
                    else:
                         logging.warning(f"Iteration {refinement_iter}: Could not find data for generated sideboard card '{card_name}' (Index: {abs_idx}) to check format legality.")


            final_absolute_indices_to_regenerate = indices_to_force_regenerate_4_copy.union(indices_to_force_regenerate_format)
            absolute_indices_to_regenerate_list = sorted(list(final_absolute_indices_to_regenerate))

            if not absolute_indices_to_regenerate_list and num_unknown_this_iter > 0:
                logging.info(f"Iteration {refinement_iter}: All generated SB cards meet constraints. Stopping refinement.")
                break
            elif num_unknown_this_iter == 0:
                 pass
            elif not absolute_indices_to_regenerate_list:
                 logging.info(f"Iteration {refinement_iter}: No SB slots marked for regeneration. Stopping refinement.")
                 break
            else:
                 logging.warning(f"Iteration {refinement_iter}: Preparing to regenerate {len(absolute_indices_to_regenerate_list)} SB slots. Indices: {absolute_indices_to_regenerate_list}")
                 # Log reasons
                 for abs_idx in absolute_indices_to_regenerate_list:
                    temp_logits = clf_model(current_x0_sb[0, abs_idx].unsqueeze(0))
                    temp_pred_idx = torch.argmax(temp_logits, dim=1).item()
                    temp_name = idx_to_card.get(temp_pred_idx, "Unknown Index")
                    reason = "Unknown"
                    is_4_copy_violator = abs_idx in indices_to_force_regenerate_4_copy
                    is_format_violator = abs_idx in indices_to_force_regenerate_format
                    if is_4_copy_violator and is_format_violator: reason = "4-Copy & Format"
                    elif is_4_copy_violator: reason = "4-Copy Rule"
                    elif is_format_violator: reason = "Format Legality"
                    logging.warning(f"  - Reconsidering SB Slot {abs_idx}: '{temp_name}' - Reason: {reason}")

                 # Prepare mask and known embeddings for the *next* iteration
                 next_mask = initial_sb_known_mask.clone()
                 next_known_emb = initial_sb_known_emb.clone()
                 all_generated_indices_ever = set(torch.where(initial_sb_known_mask[0, :, 0] == 0)[0].cpu().numpy())
                 # Use the combined set of indices to regenerate
                 valid_copy_generated_indices = list(all_generated_indices_ever - final_absolute_indices_to_regenerate)

                 if valid_copy_generated_indices:
                     next_mask[0, valid_copy_generated_indices, 0] = 1.0
                     next_known_emb[0, valid_copy_generated_indices] = current_x0_sb[0, valid_copy_generated_indices]

                 current_mask = next_mask
                 current_known_emb = next_known_emb
                 num_known_next = int(current_mask.sum().item())
                 logging.info(f"Preparing for Iteration {refinement_iter + 1}. Known SB cards: {num_known_next}, Regenerating: {SIDEBOARD_SIZE - num_known_next}")

        else:
            logging.info(f"Iteration {refinement_iter}: Refinement loop finished (max iterations reached or stopped early).")

    # --- Final Classification and Formatting ---
    final_x0_sb = current_x0_sb

    if final_x0_sb is None:
        # This case should only be hit if sideboard was full initially and returned early.
        # If SIDEBOARD_SIZE is 0, return empty list.
        if SIDEBOARD_SIZE == 0:
             return []
        # If it wasn't full initially, this is an error.
        if num_unknown_initial > 0:
             logging.error("Error: final_x0_sb is None but sideboard was not initially full and generation didn't run.")
             return []
        # Otherwise, the pre-formatted full sideboard should have been returned already.
        # We might need to re-format it here if the logic flow reaches this point unexpectedly.
        logging.warning("Reached end of sideboard function with final_x0_sb=None unexpectedly. Re-formatting initial list.")
        initial_sb_counts = Counter(original_known_sb_names)
        initial_sb_unique_names = list(initial_sb_counts.keys())
        image_urls = get_card_image_urls(initial_sb_unique_names)
        completed_sideboard_list = []
        for name, count in initial_sb_counts.items():
             img_url = image_urls.get(name)
             completed_sideboard_list.append({"name": name, "count": count, "image_url": img_url})
        return completed_sideboard_list[:SIDEBOARD_SIZE]


    final_unknown_mask_flat = (initial_sb_known_mask[0, :, 0] == 0)
    generated_sb_names = []

    if final_unknown_mask_flat.sum() > 0:
        final_unknown_embeddings = final_x0_sb[0][final_unknown_mask_flat]
        logging.info(f"Classifying {final_unknown_embeddings.shape[0]} final generated sideboard embeddings...")
        final_logits_sb = clf_model(final_unknown_embeddings)
        final_predicted_indices = torch.argmax(final_logits_sb, dim=1).cpu().numpy()
        temp_generated_names = []
        for idx in final_predicted_indices:
            name = idx_to_card.get(int(idx))
            if name is not None:
                temp_generated_names.append(name)
            else:
                logging.warning(f"Classifier predicted unknown index {int(idx)} in final sideboard. Replacing with 'Error Card SB'.")
                temp_generated_names.append("Error Card SB")
        generated_sb_names = temp_generated_names
    else:
         logging.info("No sideboard cards were generated (initial sideboard was full or SIDEBOARD_SIZE=0).")

    completed_sb_names = original_known_sb_names + generated_sb_names

    if len(completed_sb_names) != SIDEBOARD_SIZE and SIDEBOARD_SIZE > 0:
         logging.warning(f"Final sideboard construction resulted in {len(completed_sb_names)} cards, expected {SIDEBOARD_SIZE}. Padding/Truncating.")
         if len(completed_sb_names) < SIDEBOARD_SIZE: completed_sb_names.extend(["Error Card SB"] * (SIDEBOARD_SIZE - len(completed_sb_names)))
         else: completed_sb_names = completed_sb_names[:SIDEBOARD_SIZE]

    final_sb_counts = Counter(completed_sb_names)
    sb_unique_names = list(final_sb_counts.keys())
    image_urls = get_card_image_urls(sb_unique_names)

    completed_sideboard_list = []
    for name, count in final_sb_counts.items():
        if name == "Error Card SB":
            img_url = None
            logging.error("Final sideboard contains 'Error Card SB' placeholder.")
        else:
            img_url = image_urls.get(name)
        completed_sideboard_list.append({"name": name, "count": count, "image_url": img_url})


    final_count = sum(c['count'] for c in completed_sideboard_list)
    logging.info(f"Sideboard completion complete. Final count: {final_count}.")
    if final_count != SIDEBOARD_SIZE and SIDEBOARD_SIZE > 0:
        logging.warning(f"Sideboard final count ({final_count}) does not match SIDEBOARD_SIZE ({SIDEBOARD_SIZE}).")


    return completed_sideboard_list


# --- RunPod Handler ---
def handler(event):
    """
    Processes incoming requests for deck or sideboard completion.
    Input event structure depends on the calling method.
    Simulate endpoint routing based on expected keys.
    """
    logging.info(f"Received event: {event}")
    job_input = event.get('input', {})

    if not job_input:
        return {"error": "Missing 'input' field in the event."}

    # Determine task: complete main deck or sideboard
    task = job_input.get('task') # Expect 'complete_deck' or 'complete_sideboard'
    format_input = job_input.get('format', DEFAULT_FORMAT).lower()
    if format_input not in ALLOWED_FORMATS:
        return {"error": f"Invalid format '{format_input}'. Allowed: {', '.join(ALLOWED_FORMATS)}"}

    start_time = time.time()

    try:
        if task == 'complete_deck':
            logging.info("Task: Complete Deck")
            deck_text = job_input.get('deck_list')
            if not deck_text:
                return {"error": "Missing 'deck_list' for complete_deck task."}

            known_cards = parse_deck_input(deck_text)
            total_known = sum(c['count'] for c in known_cards)
            if total_known > DECK_SIZE:
                return {"error": f"Input deck has {total_known} cards (max {DECK_SIZE})."}
            if total_known == 0:
                return {"error": "Input deck cannot be empty."}

            completed_deck = run_inference(known_cards, format_input)
            if not completed_deck:
                 return {"error": "Main deck inference failed."}

            end_time = time.time()
            logging.info(f"Deck completion took {end_time - start_time:.2f} seconds.")
            return {"completed_deck": completed_deck}

        elif task == 'complete_sideboard':
            logging.info("Task: Complete Sideboard")
            main_deck_list = job_input.get('completed_deck')
            current_sideboard_text = job_input.get('current_sideboard', "") # Allow empty string for empty SB

            if not main_deck_list:
                return {"error": "Missing 'completed_deck' for complete_sideboard task."}
            if not isinstance(main_deck_list, list):
                 return {"error": "'completed_deck' must be a list."}
            if sum(c.get('count', 0) for c in main_deck_list) != DECK_SIZE:
                 # Allow flexibility if main deck had errors, but log
                 logging.warning(f"Sideboard completion received a main deck with {sum(c.get('count', 0) for c in main_deck_list)} cards, expected {DECK_SIZE}.")
                 # return {"error": f"'completed_deck' must contain exactly {DECK_SIZE} cards."}


            # Parse the current sideboard text (can be empty)
            current_sideboard_list = parse_deck_input(current_sideboard_text) if current_sideboard_text else []
            current_sb_count = sum(c.get('count', 0) for c in current_sideboard_list)
            if current_sb_count > SIDEBOARD_SIZE:
                return {"error": f"Input sideboard has {current_sb_count} cards (max {SIDEBOARD_SIZE})."}


            completed_sideboard = complete_sideboard_inference(main_deck_list, current_sideboard_list, format_input)

            end_time = time.time()
            logging.info(f"Sideboard completion took {end_time - start_time:.2f} seconds.")
            return {"completed_sideboard": completed_sideboard}

        else:
            return {"error": f"Invalid task specified: '{task}'. Use 'complete_deck' or 'complete_sideboard'."}

    except ValueError as e:
        logging.error(f"Value Error during processing: {e}")
        return {"error": f"Input Error: {str(e)}"}
    except RuntimeError as e:
        logging.error(f"Runtime Error during processing: {e}")
        # This might indicate model loading issues or CUDA errors
        return {"error": f"Server Error: {str(e)}"}
    except FileNotFoundError as e:
         logging.error(f"File Not Found Error: {e}")
         return {"error": f"Server Configuration Error: Missing required file {str(e)}"}
    except Exception as e:
        logging.exception("An unexpected error occurred during handler execution:") # Log full traceback
        return {"error": f"An internal server error occurred: {str(e)}"}


# --- Main Execution ---
if __name__ == '__main__':
    print("Starting RunPod Serverless Worker...")
    try:
        load_models_and_data()
        print("Models loaded. Starting serverless handler...")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"FATAL: Failed to initialize worker: {e}")