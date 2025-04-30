import os
import re
import json
import requests
import time
import pickle
import math
import numpy as np
from flask import Flask, request, jsonify, render_template, current_app
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim
from scipy.spatial.distance import cosine
import nltk
from nltk.corpus import stopwords
import logging

# --- NLTK Stopwords Download ---
try:
    stopwords.words('english')
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")

# --- Configuration ---
# Adjust these paths if your files are located elsewhere
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_MODEL_PATH = os.path.join(MODEL_DIR, "models/diffusion_model.pth") # Adjusted path assuming 'models' subdir
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "models/card_classifier.pt") # Adjusted path
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "data/card_embeddings.pkl") # Adjusted path
DOC2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "models/embedding_model")

SCRYFALL_API_BASE = "https://api.scryfall.com"
SCRYFALL_REQUEST_DELAY = 0.1 # Scryfall asks for 50-100ms delay between requests

# Model & Inference Constants
EMB_DIM = 128
DECK_SIZE = 60
SIDEBOARD_SIZE = 15 # Added constant
TIMESTEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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

class DiffusionModel(nn.Module):
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
        self.sb_time_mlp = nn.Sequential( # Renamed from sb_shared_time_mlp, only for SB Decoder
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
            num_layers=1
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
            num_layers=1
        )
        self.sb_transformer_output = nn.TransformerEncoder(
            sb_context_encoder_layer,
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
        sb_decoded = self.sb_transformer_output(sb_decoded)
        sb_noise_pred = self.sb_output_proj(sb_decoded)
        return sb_noise_pred

# --- Card Classifier --- (Keep as is)
class CardClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CardClassifier, self).__init__()
        self.network = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.network(x)

# --- Global Variables --- (Keep as is)
diffusion_model = None
clf_model = None
card_embeddings = None
idx_to_card = None
diffusion_beta = None
diffusion_alpha = None
diffusion_alpha_bar = None
doc2vec_model = None

def load_models_and_data():
    global diffusion_model, clf_model, card_embeddings, idx_to_card
    global diffusion_beta, diffusion_alpha, diffusion_alpha_bar
    global doc2vec_model

    print("Loading models and data...")
    # Load Embeddings (Keep as is, maybe adjust path if needed)
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, "rb") as f:
        card_embeddings = pickle.load(f)
    print(f"Loaded {len(card_embeddings)} card embeddings.")

    # Load Doc2Vec Model (Keep as is)
    if not os.path.exists(DOC2VEC_MODEL_PATH):
        raise FileNotFoundError(f"Doc2Vec model file not found: {DOC2VEC_MODEL_PATH}")
    try:
        print(f"Loading Doc2Vec model from {DOC2VEC_MODEL_PATH}...")
        doc2vec_model = gensim.models.Doc2Vec.load(DOC2VEC_MODEL_PATH)
        print("Doc2Vec model loaded.")
    except Exception as e:
        print(f"Error loading Doc2Vec model: {e}")
        raise

    # Load Diffusion Model (Needs Update)
    if not os.path.exists(DIFFUSION_MODEL_PATH):
         raise FileNotFoundError(f"Diffusion model checkpoint not found: {DIFFUSION_MODEL_PATH}")
    try:
        print(f"Loading diffusion model from {DIFFUSION_MODEL_PATH}...")
        diff_ckpt = torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE)

        # Check if the checkpoint has a 'config' key, otherwise assume it's just the state_dict
        if "model" in diff_ckpt and isinstance(diff_ckpt["model"], dict):
            model_state_dict = diff_ckpt["model"]
             # Try to get config from checkpoint, otherwise use command-line defaults from training script
            diff_cfg = diff_ckpt.get("config", {})
            diff_cfg.setdefault("layers", 6) # Default from training script
            diff_cfg.setdefault("sb_layers", 6) # Default from training script
            diff_cfg.setdefault("heads", 8)   # Default from training script
            diff_cfg.setdefault("dim_feedforward", 2048) # Default from training script
            diff_cfg.setdefault("model_dim", 256) # Default internal projection dimension
            print(f"Using diffusion model config: {diff_cfg}")
        else:
            # Assume older checkpoint format just contains the state dict
            model_state_dict = diff_ckpt
            # Use default config since it wasn't saved in the checkpoint
            diff_cfg = {
                "layers": 6, "sb_layers": 1, "heads": 8,
                "dim_feedforward": 2048, "model_dim": 256
            }
            print("Warning: Diffusion checkpoint missing 'config' key. Using default config.")
            print(f"Default diffusion config: {diff_cfg}")

        diffusion_model = DiffusionModel(diff_cfg).to(DEVICE)
        # Load state dict - use strict=False if architecture changed significantly
        # between training and inference (e.g., added/removed layers)
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(model_state_dict, strict=False)
        if missing_keys:
             print(f"Warning: Missing keys in diffusion model state_dict: {missing_keys}")
        if unexpected_keys:
             print(f"Warning: Unexpected keys in diffusion model state_dict: {unexpected_keys}")

        diffusion_model.eval()
        diffusion_beta = cosine_beta_schedule(TIMESTEPS).to(DEVICE)
        diffusion_alpha = 1.0 - diffusion_beta
        diffusion_alpha_bar = torch.cumprod(diffusion_alpha, dim=0)
        print("Diffusion model loaded.")
    except KeyError as e:
        print(f"Error: Key missing in diffusion model checkpoint: {e}. Checkpoint structure might be incompatible.")
        raise
    except Exception as e:
        print(f"Error loading diffusion model: {e}")
        raise

    # Load Classifier Model (Keep as is, maybe check path)
    if not os.path.exists(CLASSIFIER_PATH):
         raise FileNotFoundError(f"Classifier model checkpoint not found: {CLASSIFIER_PATH}")
    try:
        print(f"Loading classifier model from {CLASSIFIER_PATH}...")
        clf_ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
        clf_model = CardClassifier(clf_ckpt["embedding_dim"], clf_ckpt["num_classes"]).to(DEVICE)
        clf_model.load_state_dict(clf_ckpt["model_state_dict"])
        clf_model.eval()
        idx_to_card = clf_ckpt["idx_to_card"]
        print("Classifier model loaded.")
    except KeyError as e:
        print(f"Error: Key missing in classifier model checkpoint: {e}. Checkpoint structure might be different.")
        raise
    except Exception as e:
        print(f"Error loading classifier model: {e}")
        raise

    print("Models and data loaded successfully.")

# --- Flask App Initialization ---
app = Flask(__name__)
app.logger.setLevel(logging.INFO) # Use INFO for more details during dev/debug

load_models_and_data() # Load models when the app starts

# --- Helper Functions --- (Keep clean_search_text and parse_deck_input as is)

# Text cleaning logic adapted from create_text_encodings.py
reminder_remover = re.compile(r'\(.*?\)') # Match parentheses and content
stop_words = set(stopwords.words('english'))
# Allowed characters are not strictly enforced here as user input might be more varied,
# but basic cleaning and stopword removal are applied.

def clean_search_text(text):
    """Cleans the user's search description similarly to card text preprocessing."""
    if not text:
        return []

    # Basic cleaning: lowercase, remove reminders, standard replacements
    text = text.lower()
    text = re.sub(reminder_remover, '', text.replace('}{', '} {'))
    text = text.replace('&', 'and').replace('\n', ' ').replace(';', ' ').replace(':', ' :')
    text = text.replace('−', '-').replace('—', '-') # Handle different dash types
    text = text.replace('’', "'").replace('`', "'") # Handle apostrophes
    text = text.replace(',', '').replace('.', '').replace('\'', '').replace('"', '') # Remove punctuation

    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word and word not in stop_words]
    return filtered_words


def parse_deck_input(deck_text):
    """Parses the input text area format into a list of card dicts."""
    cards = []
    lines = deck_text.strip().split('\n')
    # Regex to capture "(Number)x [Card Name]", handling optional whitespace
    pattern = re.compile(r"^\s*(\d+)\s*[xX]?\s*(.+?)\s*$")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            count = int(match.group(1))
            name = match.group(2).strip() # Remove leading/trailing spaces from name
            if count > 0 and name:
                # Basic validation: check if card name exists in embeddings
                if name not in card_embeddings:
                     raise ValueError(f"Card not found in embeddings: '{name}'. Please check spelling.")
                cards.append({"name": name, "count": count})
        else:
            # Handle lines that might just be card names (assume count 1)
            if line:
                 if line not in card_embeddings:
                     raise ValueError(f"Card not found in embeddings: '{line}'. Please check spelling.")
                 cards.append({"name": line, "count": 1})

    # Combine duplicate entries
    card_counts = Counter()
    for card in cards:
        card_counts[card['name']] = card['count']

    return [{'name': name, 'count': count} for name, count in card_counts.items()]


# --- Inference Function (Main Deck Completion Only) ---
@torch.no_grad()
def run_inference(known_cards_list):
    """
    Runs diffusion model inference to complete the main deck, preserving known cards.

    Args:
        known_cards_list (list): [{'name': '...', 'count': ...}, ...] for main deck.

    Returns:
        list: Completed main deck list: [{'name': ..., 'count': ..., 'image_url': ...}]
    """
    current_app.logger.info(f"Running main deck inference for known_cards: {known_cards_list}")

    if diffusion_model is None or clf_model is None or card_embeddings is None or idx_to_card is None:
         raise RuntimeError("Models or data not loaded properly for inference.")

    # 1. Prepare known main deck embeddings and mask
    known_emb = torch.zeros(1, DECK_SIZE, EMB_DIM, device=DEVICE)
    known_mask = torch.zeros(1, DECK_SIZE, 1, device=DEVICE)
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
                known_emb[0, current_idx] = vec
                known_mask[0, current_idx] = 1.0
                original_known_names.append(name)
                current_idx += 1
            else:
                current_app.logger.warning(f"Input deck exceeds {DECK_SIZE} cards. Truncating.")
                break
        if current_idx >= DECK_SIZE:
            break

    num_unknown_main = DECK_SIZE - total_known_count
    if num_unknown_main < 0:
         current_app.logger.error(f"Error: total_known_count ({total_known_count}) > DECK_SIZE ({DECK_SIZE})")
         raise ValueError("Input deck size exceeds maximum allowed.")

    current_app.logger.info(f"Prepared known main deck embeddings for {total_known_count} cards. Generating {num_unknown_main} main deck cards.")

    # 2. Initialize Noise for Main Deck
    x = torch.randn(1, DECK_SIZE, EMB_DIM, device=DEVICE)
    x = known_mask * known_emb + (1 - known_mask) * x # Initialize main deck (noise in unknown slots)

    # 3. Run Main Deck Diffusion Sampling Loop
    for t in reversed(range(TIMESTEPS)):
        t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)

        # Predict noise for main deck using the dedicated method
        main_noise_pred = diffusion_model.predict_main_noise(x, t_tensor, known_mask)

        # Get diffusion parameters for timestep t
        beta_t = diffusion_beta[t].to(DEVICE)
        alpha_t = diffusion_alpha[t].to(DEVICE)
        alpha_bar_t = diffusion_alpha_bar[t].to(DEVICE)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()

        # Calculate mean for main deck
        mean_main = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * main_noise_pred)

        # Add noise for next step (if not t=0)
        if t > 0:
            noise_main = torch.randn_like(x)
            x_next = mean_main + noise_main * beta_t.sqrt()
        else:
            x_next = mean_main # Final step uses mean

        # Re-apply known mask to main deck
        x = known_mask * known_emb + (1 - known_mask) * x_next

    x0_final_main = x # Resulting main deck tensor [1, DECK_SIZE, EMB_DIM]

    # 4. Classify Generated Embeddings (Main Deck Unknowns)
    generated_main_names = []
    if num_unknown_main > 0:
        unknown_main_mask = (known_mask[0, :, 0] == 0)
        unknown_main_embeddings = x0_final_main[0][unknown_main_mask]
        if unknown_main_embeddings.shape[0] != num_unknown_main:
             current_app.logger.error(f"Main deck shape mismatch: Expected {num_unknown_main} unknown embeddings, found {unknown_main_embeddings.shape[0]}")
             raise RuntimeError("Failed to correctly isolate unknown main deck embeddings.")

        current_app.logger.info(f"Classifying {num_unknown_main} generated main deck embeddings...")
        logits_main = clf_model(unknown_main_embeddings)
        predicted_indices_main = torch.argmax(logits_main, dim=1).cpu().numpy()
        generated_main_names = [idx_to_card[int(idx)] for idx in predicted_indices_main]
        current_app.logger.info(f"Generated main deck card names: {Counter(generated_main_names)}")

    # 5. Combine and Format Main Deck Results
    completed_deck_names = original_known_names + generated_main_names
    if len(completed_deck_names) != DECK_SIZE:
         current_app.logger.error(f"Final main deck construction error: Expected {DECK_SIZE} cards, got {len(completed_deck_names)}")
         # Simple padding/truncation fallback
         if len(completed_deck_names) < DECK_SIZE:
             completed_deck_names.extend(["Error Card"] * (DECK_SIZE - len(completed_deck_names)))
         else:
             completed_deck_names = completed_deck_names[:DECK_SIZE]

    final_main_counts = Counter(completed_deck_names)

    # Fetch images for unique main deck names
    main_unique_names = list(final_main_counts.keys())
    image_urls = get_card_image_urls(main_unique_names)

    # Structure main deck results
    completed_deck_list = []
    for name, count in final_main_counts.items():
        completed_deck_list.append({
            "name": name, "count": count, "image_url": image_urls.get(name)
        })

    current_app.logger.info(f"Main deck inference complete. Final count: {sum(c['count'] for c in completed_deck_list)}.")

    return completed_deck_list

# --- Inference Function (Sideboard Completion) ---
@torch.no_grad()
def complete_sideboard_inference(main_deck_list, current_sideboard_list):
    """
    Completes a sideboard based on a provided main deck and current sideboard cards.

    Args:
        main_deck_list (list): Completed 60-card main deck list: 
                               [{'name': ..., 'count': ..., 'image_url': ...}, ...]
        current_sideboard_list (list): Current cards in the sideboard: 
                                     [{'name': ..., 'count': ..., 'image_url': ...}, ...]

    Returns:
        list: Completed sideboard list: [{'name': ..., 'count': ..., 'image_url': ...}]
    """
    current_app.logger.info(f"Running sideboard completion based on main deck and current sideboard: {current_sideboard_list}")

    if diffusion_model is None or clf_model is None or card_embeddings is None or idx_to_card is None:
         raise RuntimeError("Models or data not loaded properly for sideboard inference.")

    if SIDEBOARD_SIZE <= 0:
        current_app.logger.info("SIDEBOARD_SIZE is 0 or less, returning empty sideboard.")
        return []
        
    # 1. Reconstruct main deck embedding tensor (context)
    main_deck_embeddings = torch.zeros(1, DECK_SIZE, EMB_DIM, device=DEVICE)
    main_current_idx = 0
    main_total_cards = 0
    for card_info in main_deck_list:
        name = card_info["name"]
        count = card_info["count"]
        main_total_cards += count
        try: vec = torch.tensor(card_embeddings[name], dtype=torch.float32, device=DEVICE)
        except KeyError: raise ValueError(f"Card '{name}' from input main deck not found in embeddings.")
        for _ in range(count):
            if main_current_idx < DECK_SIZE: main_deck_embeddings[0, main_current_idx] = vec; main_current_idx += 1
            else: raise ValueError("Provided main deck list exceeds DECK_SIZE.")
    if main_total_cards != DECK_SIZE or main_current_idx != DECK_SIZE:
        raise ValueError(f"Provided main deck list does not contain exactly {DECK_SIZE} cards.")
    current_app.logger.info(f"Reconstructed main deck embedding tensor for sideboard context.")
    
    # 2. Prepare Known Sideboard Embeddings and Mask
    sb_known_emb = torch.zeros(1, SIDEBOARD_SIZE, EMB_DIM, device=DEVICE)
    sb_known_mask = torch.zeros(1, SIDEBOARD_SIZE, 1, device=DEVICE)
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
                sb_known_emb[0, sb_current_idx] = vec
                sb_known_mask[0, sb_current_idx] = 1.0
                original_known_sb_names.append(name)
                sb_current_idx += 1
            else:
                current_app.logger.warning(f"Input sideboard exceeds {SIDEBOARD_SIZE} cards. Truncating.")
                total_known_sb_count = SIDEBOARD_SIZE # Adjust count if truncated
                break 
        if sb_current_idx >= SIDEBOARD_SIZE:
            break
    
    num_unknown_sb = SIDEBOARD_SIZE - total_known_sb_count
    if num_unknown_sb < 0:
        # This case implies input validation failed or truncation occurred.
        # If truncated, num_unknown_sb should be 0.
        current_app.logger.warning(f"Known sideboard cards ({total_known_sb_count}) exceeded SIDEBOARD_SIZE ({SIDEBOARD_SIZE}). Assuming full sideboard provided.")
        num_unknown_sb = 0 # Cannot generate more cards
        
    current_app.logger.info(f"Prepared known sideboard embeddings for {total_known_sb_count} cards. Generating {num_unknown_sb} cards.")

    # 3. Initialize Sideboard Noise (only in unknown slots)
    sb_x = torch.randn(1, SIDEBOARD_SIZE, EMB_DIM, device=DEVICE)
    sb_x = sb_known_mask * sb_known_emb + (1 - sb_known_mask) * sb_x 

    # 4. Pre-calculate Main Deck Context Encoding 
    sb_context_encoded = diffusion_model.encode_main_deck_context(main_deck_embeddings)
    current_app.logger.info("Calculated sideboard context encoding from main deck.")

    # 5. Run Sideboard Diffusion Sampling Loop
    for t in reversed(range(TIMESTEPS)):
        t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)

        # Predict noise for sideboard using the dedicated method and pre-computed context
        # Pass the *current* sb_x, timestep, the known mask, and context
        sb_noise_pred = diffusion_model.predict_sideboard_noise(sb_x, t_tensor, sb_known_mask, sb_context_encoded)

        # Diffusion update steps for sb_x
        beta_t = diffusion_beta[t].to(DEVICE)
        alpha_t = diffusion_alpha[t].to(DEVICE)
        alpha_bar_t = diffusion_alpha_bar[t].to(DEVICE)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()
        mean_sb = (1.0 / sqrt_alpha_t) * (sb_x - (beta_t / sqrt_one_minus_alpha_bar_t) * sb_noise_pred)
        if t > 0:
            noise_sb = torch.randn_like(sb_x)
            sb_x_next = mean_sb + noise_sb * beta_t.sqrt()
        else:
            sb_x_next = mean_sb
            
        # IMPORTANT: Apply mask to preserve known cards during update
        sb_x = sb_known_mask * sb_known_emb + (1 - sb_known_mask) * sb_x_next

    x0_final_sb = sb_x # Resulting completed sideboard tensor [1, SIDEBOARD_SIZE, EMB_DIM]

    # 6. Classify *Only Generated* Sideboard Embeddings
    generated_sb_names = []
    if num_unknown_sb > 0:
        unknown_sb_mask_flat = (sb_known_mask[0, :, 0] == 0)
        unknown_sb_embeddings = x0_final_sb[0][unknown_sb_mask_flat]
        
        # Sanity check shape
        if unknown_sb_embeddings.shape[0] != num_unknown_sb:
             current_app.logger.error(f"Sideboard shape mismatch: Expected {num_unknown_sb} unknown embeddings, found {unknown_sb_embeddings.shape[0]}")
             # Attempt to continue, but log error
             if unknown_sb_embeddings.shape[0] < num_unknown_sb:
                 num_unknown_sb = unknown_sb_embeddings.shape[0] # Adjust count
             else: # Too many? Take the expected number
                 unknown_sb_embeddings = unknown_sb_embeddings[:num_unknown_sb]
                 
        if num_unknown_sb > 0: # Check again after potential adjustment
             current_app.logger.info(f"Classifying {num_unknown_sb} generated sideboard embeddings...")
             logits_sb = clf_model(unknown_sb_embeddings)
             predicted_indices_sb = torch.argmax(logits_sb, dim=1).cpu().numpy()
             generated_sb_names = [idx_to_card[int(idx)] for idx in predicted_indices_sb]
             current_app.logger.info(f"Generated sideboard card names: {Counter(generated_sb_names)}")
        else:
             current_app.logger.info("No unknown sideboard slots to classify after shape check.")

    # 7. Combine Original Known SB Names and Generated SB Names
    completed_sb_names = original_known_sb_names + generated_sb_names
    # Ensure final count is exactly SIDEBOARD_SIZE (handle potential errors/truncation)
    if len(completed_sb_names) != SIDEBOARD_SIZE:
         current_app.logger.warning(f"Final sideboard construction resulted in {len(completed_sb_names)} cards, expected {SIDEBOARD_SIZE}. Padding/Truncating.")
         if len(completed_sb_names) < SIDEBOARD_SIZE:
             # This case is less likely now, maybe pad with a placeholder?
             completed_sb_names.extend(["Unknown SB Card"] * (SIDEBOARD_SIZE - len(completed_sb_names)))
         else:
             completed_sb_names = completed_sb_names[:SIDEBOARD_SIZE]

    # 8. Format Sideboard Results
    final_sb_counts = Counter(completed_sb_names)
    sb_unique_names = list(final_sb_counts.keys())
    image_urls = get_card_image_urls(sb_unique_names)

    completed_sideboard_list = []
    for name, count in final_sb_counts.items():
        completed_sideboard_list.append({
            "name": name, "count": count, "image_url": image_urls.get(name)
        })

    current_app.logger.info(f"Sideboard completion complete. Final count: {sum(c['count'] for c in completed_sideboard_list)}.")
    return completed_sideboard_list

# --- Scryfall Image Fetching ---
# Cache for image URLs to avoid repeated Scryfall lookups
image_cache = {}

def get_card_image_urls(card_names):
    """Fetches image URLs from Scryfall for a list of card names.
       Sends the full name (including // for split/DFC cards) to the API.
       Returns a dictionary mapping original full card names to either a string URL 
       or a dictionary {'front': url1, 'back': url2} for multi-face cards.
    """
    urls = {}
    # Map from the lowercase version of the name sent to Scryfall back to the original full name(s)
    # (Using lowercase for robust matching against Scryfall's potential case variations in response)
    scryfall_name_map_lower = {}
    names_to_fetch = set() # Use the original names for fetching

    for name in card_names:
        if name not in image_cache:
            names_to_fetch.add(name) 
            # Store mapping from lowercase name back to original name(s)
            lower_name = name.lower()
            if lower_name not in scryfall_name_map_lower:
                scryfall_name_map_lower[lower_name] = []
            # Ensure we don't add duplicate original names for the same query name
            if name not in scryfall_name_map_lower[lower_name]:
                scryfall_name_map_lower[lower_name].append(name)
        else:
            urls[name] = image_cache[name]

    if not names_to_fetch:
        # All images were already cached
        return urls

    current_app.logger.info(f"Fetching {len(names_to_fetch)} card names from Scryfall...")

    # Construct the identifiers payload using the full original names
    identifiers = [{"name": name} for name in names_to_fetch]
    payload = {"identifiers": identifiers}

    try:
        response = requests.post(f"{SCRYFALL_API_BASE}/cards/collection", json=payload)
        response.raise_for_status() 
        data = response.json()

        # Map from lowercase Scryfall result name to the image info (URL string or dict)
        found_map_scryfall_lower = {}
        # Process results for found cards
        if data and 'data' in data:
            for card_data in data['data']:
                scryfall_result_name = card_data.get('name')
                image_info = None 

                # Check for multi-face cards (includes split, flip, transform, modal_dfc, etc.)
                if card_data.get('card_faces') and len(card_data['card_faces']) > 1:
                    # Assume first two faces are most relevant for images
                    face1 = card_data['card_faces'][0]
                    face2 = card_data['card_faces'][1]
                    url1 = face1.get('image_uris', {}).get('normal')
                    url2 = face2.get('image_uris', {}).get('normal')
                    if url1 or url2:
                        image_info = {'front': url1, 'back': url2}
                
                # If not multi-face (or faces lacked images), check top-level image_uris
                elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                    image_info = card_data['image_uris']['normal']

                if scryfall_result_name:
                    # Store result keyed by the lowercase name Scryfall returned
                    found_map_scryfall_lower[scryfall_result_name.lower()] = image_info 

        # Match fetched data back to the *original* requested names and update cache/results
        # Iterate through the lowercase map we created earlier
        for lower_name_key, original_names in scryfall_name_map_lower.items():
             # Check if this lowercase name was found in Scryfall's results
             image_data = found_map_scryfall_lower.get(lower_name_key)
             
             # Add to cache and results for all original names that mapped to this lowercase name
             for original_name in original_names:
                image_cache[original_name] = image_data # Cache result (string, dict, or None)
                urls[original_name] = image_data

        # Log cards not found by Scryfall API call (based on the original names we queried)
        if 'not_found' in data:
            for not_found_identifier in data['not_found']:
                # The 'not_found' array contains the identifier objects we sent
                if 'name' in not_found_identifier:
                    missing_original_name = not_found_identifier['name']
                    # Mark this original name as None in cache and results if not already set
                    if missing_original_name not in image_cache or image_cache[missing_original_name] is None:
                        image_cache[missing_original_name] = None
                        urls[missing_original_name] = None

    except requests.exceptions.RequestException as e:
        current_app.logger.error(f"Scryfall API request failed: {e}")
        # Mark all original names associated with this batch request as failed (None)
        for original_name in names_to_fetch:
            if original_name not in urls: # Avoid overwriting already cached data
                image_cache[original_name] = None
                urls[original_name] = None
    except json.JSONDecodeError as e:
         current_app.logger.error(f"Failed to decode Scryfall JSON response: {e}")
         for original_name in names_to_fetch:
             if original_name not in urls:
                image_cache[original_name] = None
                urls[original_name] = None

    # Introduce delay ONLY if we made a request
    if names_to_fetch:
        time.sleep(SCRYFALL_REQUEST_DELAY)

    return urls

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/search-cards', methods=['POST'])
def search_cards():
    """Handles card search requests based on text description."""
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' in request"}), 400

        description = data['description'].strip()
        top_n = data.get('top_n', 10) # Get top N, default 10

        if not description:
            return jsonify({"error": "Description cannot be empty"}), 400

        if doc2vec_model is None or card_embeddings is None:
            raise RuntimeError("Models or data not loaded properly for search.")

        # --- Determine Query Vector ---
        query_vector = None

        # Check if the input description matches a known card name (case-insensitive check might be useful)
        # For simplicity, let's do a direct check first, maybe add case-insensitivity later if needed.
        # Trim whitespace for robustness.
        trimmed_description = description.strip()
        if trimmed_description in card_embeddings:
            current_app.logger.info(f"Input '{trimmed_description}' matches a known card. Using its embedding.")
            query_vector = card_embeddings[trimmed_description]
        else:
            current_app.logger.info(f"Input '{description}' does not match a known card. Inferring vector from text.")
            # Clean the input description
            cleaned_tokens = clean_search_text(description)
            if not cleaned_tokens:
                return jsonify({"error": "Description contained no searchable words after cleaning."}), 400
            # Infer embedding for the cleaned description
            query_vector = doc2vec_model.infer_vector(cleaned_tokens)

        # --- Calculate Similarities ---
        if query_vector is None:
            # This case should ideally not happen if checks above are correct
            return jsonify({"error": "Failed to determine query vector."}), 500

        similarities = []
        for card_name, embedding in card_embeddings.items():
            # Using scipy.spatial.distance.cosine (1 - similarity)
            # Ensure embeddings are numpy arrays for cosine calculation
            similarity_score = 1 - cosine(np.array(query_vector), np.array(embedding))
            similarities.append((card_name, similarity_score))

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top N results
        top_cards = similarities[:top_n]

        # Prepare results and fetch images
        result_cards = []
        top_card_names = [name for name, score in top_cards]
        image_urls = get_card_image_urls(top_card_names)

        for name, score in top_cards:
            result_cards.append({
                "name": name,
                "similarity": float(score), # Ensure score is JSON serializable
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

@app.route('/complete-deck', methods=['POST'])
def complete_deck():
    """Handles the main deck completion request."""
    try:
        data = request.get_json()
        if not data or 'deck_list' not in data:
            return jsonify({"error": "Missing 'deck_list' in request"}), 400

        deck_text = data['deck_list']
        try:
            known_cards = parse_deck_input(deck_text)
        except ValueError as e: # Catch card name errors from parser
            return jsonify({"error": str(e)}), 400

        total_known = sum(c['count'] for c in known_cards)
        if total_known > DECK_SIZE:
            return jsonify({"error": f"Input deck has more than {DECK_SIZE} cards ({total_known})."}), 400
        if total_known == 0:
            return jsonify({"error": "Input deck cannot be empty."}), 400

        # --- Run Main Deck Inference Only ---
        completed_deck_list = run_inference(known_cards)

        if not completed_deck_list: # Check if main deck generation failed
             return jsonify({"error": "Inference failed to generate main deck."}), 500

        # Verify main deck count
        final_main_count = sum(item.get('count', 0) for item in completed_deck_list)
        if final_main_count != DECK_SIZE:
            current_app.logger.warning(f"Main deck count mismatch after inference: Expected {DECK_SIZE}, got {final_main_count}. Returning result anyway.")

        current_app.logger.info(f"/complete-deck: Returning main deck ({final_main_count} cards).")

        # --- Format Output (Only Main Deck) ---
        return jsonify({
            "completed_deck": completed_deck_list
        })

    except ValueError as e:
        current_app.logger.error(f"Value Error during deck completion: {e}")
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e:
         current_app.logger.error(f"Runtime Error during deck completion: {e}")
         return jsonify({"error": "Server configuration error during inference."}), 500
    except Exception as e:
        current_app.logger.exception("An unexpected error occurred during deck completion:")
        return jsonify({"error": "An internal server error occurred."}), 500

@app.route('/complete-sideboard', methods=['POST']) # Renamed endpoint
def complete_sideboard_route():
    """Handles the sideboard completion request based on main deck and current SB."""
    try:
        data = request.get_json()
        if not data or 'completed_deck' not in data:
            return jsonify({"error": "Missing 'completed_deck' in request body"}), 400
        # Expect current sideboard, can be empty list
        if 'current_sideboard' not in data:
             return jsonify({"error": "Missing 'current_sideboard' in request body"}), 400

        main_deck_list = data['completed_deck']
        current_sideboard_list = data['current_sideboard']

        # Basic validation of main deck list
        if not isinstance(main_deck_list, list):
             return jsonify({"error": "'completed_deck' must be a list."}), 400
        if not main_deck_list or sum(c.get('count', 0) for c in main_deck_list) != DECK_SIZE:
             return jsonify({"error": f"'completed_deck' must be a list containing exactly {DECK_SIZE} cards."}), 400
             
        # Basic validation of current sideboard list
        if not isinstance(current_sideboard_list, list):
             return jsonify({"error": "'current_sideboard' must be a list."}), 400
        current_sb_count = sum(c.get('count', 0) for c in current_sideboard_list)
        if current_sb_count > SIDEBOARD_SIZE:
             return jsonify({"error": f"'current_sideboard' cannot contain more than {SIDEBOARD_SIZE} cards."}), 400

        # --- Run Sideboard Completion Inference ---
        completed_sideboard_list = complete_sideboard_inference(main_deck_list, current_sideboard_list)

        # Verify final sideboard count
        final_sb_count = sum(item.get('count', 0) for item in completed_sideboard_list)
        if final_sb_count != SIDEBOARD_SIZE and SIDEBOARD_SIZE > 0:
             current_app.logger.warning(f"Sideboard count mismatch after completion: Expected {SIDEBOARD_SIZE}, got {final_sb_count}. Returning result anyway.")

        current_app.logger.info(f"/complete-sideboard: Returning completed sideboard ({final_sb_count} cards).")

        # --- Format Output --- 
        return jsonify({
            "completed_sideboard": completed_sideboard_list # Key name matches frontend expectation
        })

    except ValueError as e: # Catch errors from validation or sideboard completion
        current_app.logger.error(f"Value Error during sideboard completion: {e}")
        return jsonify({"error": str(e)}), 400
    except RuntimeError as e: # Catch model loading errors etc.
         current_app.logger.error(f"Runtime Error during sideboard completion: {e}")
         return jsonify({"error": "Server configuration error during sideboard inference."}), 500
    except Exception as e:
        current_app.logger.exception("An unexpected error occurred during sideboard completion:")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # Use waitress or gunicorn in production instead of Flask's dev server
    # Example: waitress-serve --host 0.0.0.0 --port 5000 app:app
    # For development:
    app.run(debug=True, host='0.0.0.0') # Turn debug True for easier development/reloading