import os
import re
import json
import requests
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
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DIFFUSION_MODEL_PATH = os.path.join(MODEL_DIR, "models/diffusion_model.pth") 
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "models/card_classifier.pt")
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "data/card_embeddings.pkl")
DOC2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "models/embedding_model")

SCRYFALL_API_BASE = "https://api.scryfall.com"

# Model & Inference Constants
EMB_DIM = 128
DECK_SIZE = 60
SIDEBOARD_SIZE = 15 # Added constant
TIMESTEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ALLOWED_FORMATS = {'standard', 'pioneer', 'modern', 'legacy', 'vintage', 'pauper'}
DEFAULT_FORMAT = 'modern'

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

class CardClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CardClassifier, self).__init__()
        self.network = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.network(x)

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
    global doc2vec_model, cards

    print("Loading models and data...")

    with open('data/AtomicCards.json', 'r', encoding='utf-8') as f:
        cards = json.load(f)['data']

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_PATH}")
    with open(EMBEDDINGS_PATH, "rb") as f:
        card_embeddings = pickle.load(f)
    print(f"Loaded {len(card_embeddings)} card embeddings.")

    if not os.path.exists(DOC2VEC_MODEL_PATH):
        raise FileNotFoundError(f"Doc2Vec model file not found: {DOC2VEC_MODEL_PATH}")
    try:
        print(f"Loading Doc2Vec model from {DOC2VEC_MODEL_PATH}...")
        doc2vec_model = gensim.models.Doc2Vec.load(DOC2VEC_MODEL_PATH)
        print("Doc2Vec model loaded.")
    except Exception as e:
        print(f"Error loading Doc2Vec model: {e}")
        raise

    if not os.path.exists(DIFFUSION_MODEL_PATH):
        raise FileNotFoundError(f"Diffusion model checkpoint not found: {DIFFUSION_MODEL_PATH}")
    try:
        print(f"Loading diffusion model from {DIFFUSION_MODEL_PATH}...")
        diff_ckpt = torch.load(DIFFUSION_MODEL_PATH, map_location=DEVICE)

        model_state_dict = diff_ckpt["model"]
        diff_cfg = diff_ckpt.get("config", {})

        diffusion_model = DiffusionModel(diff_cfg).to(DEVICE)
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(model_state_dict, strict=True)
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
def run_inference(known_cards_list, format):
    """
    Runs diffusion model inference to complete the main deck and preserving known cards.

    Args:
        known_cards_list (list): [{'name': '...', 'count': ...}, ...] for main deck.
        format (str): The selected game format (e.g., 'modern').

    Returns:
        list: Completed main deck list: [{'name': ..., 'count': ..., 'image_url': ...}]
    """
    current_app.logger.info(f"Running main deck inference for known_cards: {known_cards_list}, format: {format}")

    if diffusion_model is None or clf_model is None or card_embeddings is None or idx_to_card is None:
         raise RuntimeError("Models or data not loaded properly for inference.")

    # --- Constants for Refinement ---
    MAX_REFINEMENT_ITERATIONS = 3

    # 1. Prepare *Initial* known main deck embeddings and mask
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
                current_app.logger.warning(f"Input deck exceeds {DECK_SIZE} cards. Truncating.")
                total_known_count = DECK_SIZE # Adjust count if truncated
                break
        if current_idx >= DECK_SIZE:
            break

    num_unknown_initial = DECK_SIZE - total_known_count
    if num_unknown_initial < 0:
         # This case implies truncation or error
         current_app.logger.warning(f"Known main deck cards ({total_known_count}) exceeded DECK_SIZE ({DECK_SIZE}). Assuming full deck provided.")
         num_unknown_initial = 0

    current_app.logger.info(f"Prepared initial known main deck embeddings for {total_known_count} cards. Initially generating {num_unknown_initial} cards.")

    # --- Iterative Refinement ---
    current_x0_main = None
    current_mask = initial_known_mask.clone()
    current_known_emb = initial_known_emb.clone()

    for refinement_iter in range(MAX_REFINEMENT_ITERATIONS + 1):
        current_app.logger.info(f"--- Starting Main Deck Generation/Refinement Iteration {refinement_iter} ---")

        # Slots unknown at the start of *this iteration*
        unknown_mask_flat_this_iter = (current_mask[0, :, 0] == 0)
        num_unknown_this_iter = int(unknown_mask_flat_this_iter.sum().item())

        # If no slots are unknown in this iteration, stop early
        if num_unknown_this_iter == 0 and refinement_iter > 0: # Check needed only after first iter
             current_app.logger.info(f"Iteration {refinement_iter}: No unknown slots left to refine. Stopping.")
             break
        elif num_unknown_initial == 0 and refinement_iter == 0: # Handle case where deck was full initially
             current_app.logger.info("Initial deck was full, skipping refinement loop.")
             # Need to set current_x0_main to the initial state if loop is skipped
             current_x0_main = initial_known_emb.clone()
             break

        # 2. Initialize Noise (only in unknown slots for this iteration)
        x = torch.randn(1, DECK_SIZE, EMB_DIM, device=DEVICE)
        x = current_mask * current_known_emb + (1 - current_mask) * x # Initialize (noise in unknown slots)

        # 3. Run Main Deck Diffusion Sampling Loop for this iteration
        for t in reversed(range(TIMESTEPS)):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)

            # Predict noise using the *current mask*
            main_noise_pred = diffusion_model.predict_main_noise(x, t_tensor, current_mask)

            # Get diffusion parameters
            beta_t = diffusion_beta[t].to(DEVICE)
            alpha_t = diffusion_alpha[t].to(DEVICE)
            alpha_bar_t = diffusion_alpha_bar[t].to(DEVICE)
            sqrt_alpha_t = alpha_t.sqrt()
            sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt()

            # Calculate mean
            mean_main = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * main_noise_pred)

            # Add noise for next step (if not t=0)
            if t > 0:
                noise_main = torch.randn_like(x)
                x_next = mean_main + noise_main * beta_t.sqrt()
            else:
                x_next = mean_main # Final step uses mean

            # Re-apply *current known mask* and *current known embeddings*
            x = current_mask * current_known_emb + (1 - current_mask) * x_next

        current_x0_main = x # Resulting main deck tensor for this iteration

        # --- Check 4-Copy Limit and Prepare for Next Iteration
        if refinement_iter < MAX_REFINEMENT_ITERATIONS:
            current_app.logger.info(f"Iteration {refinement_iter}: Checking 4-copy limit...")
            # Classify ALL slots from the current result to get counts
            iter_all_logits_main = clf_model(current_x0_main[0])
            iter_all_predicted_indices = torch.argmax(iter_all_logits_main, dim=1).cpu().numpy()
            # Get names, filtering out potential None results from idx_to_card safely
            iter_all_predicted_names = []
            name_idx_map = {}
            for i, idx in enumerate(iter_all_predicted_indices):
                name = idx_to_card.get(int(idx))
                if name is not None:
                    iter_all_predicted_names.append(name)
                    name_idx_map[i] = name # Map index to valid name
                # else: log warning or handle index? For now, just skip.

            iter_card_counts = Counter(iter_all_predicted_names)
            indices_to_force_regenerate_4_copy = set()
            indices_to_force_regenerate_format = set()

            for card_name, count in iter_card_counts.items():
                # Safely get card data
                card_data_list = cards.get(card_name)
                if not card_data_list: continue
                card = card_data_list[0]
                supertypes = card.get("supertypes", [])

                if count > 4 and "Basic" not in supertypes:
                    current_app.logger.warning(f"Iteration {refinement_iter}: Card '{card_name}' found {count} times. Marking {count - 4} generated copies for regeneration.")
                    # Find all absolute indices for this card *that have a valid name mapped*
                    current_indices = [i for i, name in name_idx_map.items() if name == card_name]
                    # Identify which of these were *generated* in this or previous refinement steps
                    generated_indices_for_card = {i for i in current_indices if initial_known_mask[0, i, 0] == 0.0}

                    num_to_replace = count - 4
                    # Add the required number of generated indices to the force regenerate set
                    indices_to_force_regenerate_4_copy.update(list(generated_indices_for_card)[:num_to_replace])

            # Format Legality Check (only for generated cards)
            for abs_idx, card_name in name_idx_map.items(): # Iterate using the map of index -> valid name
                # Check if this slot was *generated* (not part of initial input)
                if initial_known_mask[0, abs_idx, 0] == 0.0:
                    card_data_list = cards.get(card_name)
                    if card_data_list:
                        card_data = card_data_list[0]
                        legalities = card_data.get("legalities", {})
                        supertypes = card_data.get("supertypes", [])
                        is_legal = legalities.get(format, "not_legal") == "Legal" or legalities.get(format, "not_legal") == "Restricted"
                        is_basic = "Basic" in supertypes
                        
                        if not is_legal and not is_basic:
                            current_app.logger.warning(f"Iteration {refinement_iter}: Generated card '{card_name}' (Index: {abs_idx}) is not legal in {format}. Marking for regeneration.")
                            indices_to_force_regenerate_format.add(abs_idx)
                    else:
                         current_app.logger.warning(f"Iteration {refinement_iter}: Could not find data for generated card '{card_name}' (Index: {abs_idx}) to check format legality.")

            # Combine regeneration sets
            final_absolute_indices_to_regenerate = indices_to_force_regenerate_4_copy.union(indices_to_force_regenerate_format)

            if indices_to_force_regenerate_4_copy:
                current_app.logger.info(f"Iteration {refinement_iter}: Marking {len(indices_to_force_regenerate_4_copy)} slots for regeneration due to 4-copy rule: {sorted(list(indices_to_force_regenerate_4_copy))}")
            if indices_to_force_regenerate_format:
                 current_app.logger.info(f"Iteration {refinement_iter}: Marking {len(indices_to_force_regenerate_format)} additional slots for regeneration due to format ({format}) legality: {sorted(list(indices_to_force_regenerate_format))}")
            #else:
            #     current_app.logger.info(f"Iteration {refinement_iter}: No cards exceeded 4-copy limit requiring forced regeneration.")

            # 3. Determine Regeneration Targets
            #final_absolute_indices_to_regenerate = indices_to_force_regenerate_4_copy
            final_absolute_indices_to_regenerate_list = sorted(list(final_absolute_indices_to_regenerate))

            # 4. Check if Regeneration is Needed and Prepare for Next Iteration
            if not final_absolute_indices_to_regenerate_list and num_unknown_this_iter > 0 :
                current_app.logger.info(f"All {num_unknown_this_iter} generated cards meet 4-copy and format legality rules. Stopping refinement.")
                break # Stop refinement if all generated cards are good
            elif num_unknown_this_iter == 0:
                # This case means no cards were generated, loop should terminate naturally or break earlier
                pass
            elif not final_absolute_indices_to_regenerate_list:
                 # This case should not be reachable due to the break above, but good for safety
                 current_app.logger.info(f"Iteration {refinement_iter}: No slots marked for regeneration. Stopping refinement.")
                 break
            else:
                 current_app.logger.warning(f"Iteration {refinement_iter}: Preparing to regenerate {len(final_absolute_indices_to_regenerate_list)} slots (due to 4-copy rule). Indices: {final_absolute_indices_to_regenerate_list}")

                 # Log details for slots being reconsidered
                 for abs_idx in final_absolute_indices_to_regenerate_list:
                     temp_logits = clf_model(current_x0_main[0, abs_idx].unsqueeze(0))
                     temp_pred_idx = torch.argmax(temp_logits, dim=1).item()
                     temp_name = idx_to_card.get(temp_pred_idx, "Unknown Index")
                     # Determine reason
                     reason = "Unknown"
                     if abs_idx in indices_to_force_regenerate_4_copy and abs_idx in indices_to_force_regenerate_format:
                         reason = "4-Copy & Format"
                     elif abs_idx in indices_to_force_regenerate_4_copy:
                         reason = "4-Copy Rule"
                     elif abs_idx in indices_to_force_regenerate_format:
                         reason = "Format Legality"
                     current_app.logger.warning(f"  - Reconsidering Main Deck Slot {abs_idx}: '{temp_name}' - Reason: {reason}")

                 # --- Prepare mask and known embeddings for the *next* iteration ---
                 next_mask = initial_known_mask.clone()
                 next_known_emb = initial_known_emb.clone()

                 # Identify generated slots that are *not* being regenerated
                 all_generated_indices_ever = set(torch.where(initial_known_mask[0, :, 0] == 0)[0].cpu().numpy()) # All indices NOT initially known
                 valid_copy_generated_indices = list(all_generated_indices_ever - final_absolute_indices_to_regenerate)

                 if valid_copy_generated_indices:
                     next_mask[0, valid_copy_generated_indices, 0] = 1.0
                     # Use embeddings from the original diffusion result for knowns
                     next_known_emb[0, valid_copy_generated_indices] = current_x0_main[0, valid_copy_generated_indices]

                 current_mask = next_mask
                 current_known_emb = next_known_emb
                 num_unknown_next = DECK_SIZE - int(current_mask.sum().item())
                 current_app.logger.info(f"Preparing for Iteration {refinement_iter + 1}. Known cards: {int(current_mask.sum().item())}, Regenerating: {num_unknown_next}")

        else:
             current_app.logger.info(f"Iteration {refinement_iter}: Refinement loop finished (max iterations reached or stopped early).")

    # --- Final Classification and Formatting ---
    # Handle case where loop was skipped because deck was initially full
    if current_x0_main is None:
        if num_unknown_initial == 0:
            current_x0_main = initial_known_emb.clone()
            current_app.logger.info("Using initial known embeddings as final result (deck was full).")
        else:
            # This indicates an error state
            current_app.logger.error("Error: current_x0_main is None but deck was not initially full.")
            return [] # Return empty on error

    final_x0_main = current_x0_main # Use the result from the last iteration's diffusion sampling or initial state

    # Identify slots that were *ultimately* generated (not in the initial mask)
    final_unknown_mask_flat = (initial_known_mask[0, :, 0] == 0)

    generated_main_names = []
    if final_unknown_mask_flat.sum() > 0:
        final_unknown_embeddings = final_x0_main[0][final_unknown_mask_flat]

        current_app.logger.info(f"Classifying {final_unknown_embeddings.shape[0]} final generated main deck embeddings...")
        final_logits_main = clf_model(final_unknown_embeddings)
        final_predicted_indices = torch.argmax(final_logits_main, dim=1).cpu().numpy()
        # Safely get names
        temp_generated_names = []
        for idx in final_predicted_indices:
            name = idx_to_card.get(int(idx))
            if name is not None:
                temp_generated_names.append(name)
            else:
                current_app.logger.warning(f"Classifier predicted unknown index {int(idx)} in final main deck. Replacing with 'Error Card Main'.")
                temp_generated_names.append("Error Card Main")
        generated_main_names = temp_generated_names

    else:
        current_app.logger.info("No main deck cards were generated (initial deck was full).")

    # 5. Combine and Format Main Deck Results
    completed_deck_names = original_known_names + generated_main_names
    if len(completed_deck_names) != DECK_SIZE:
         current_app.logger.error(f"Final main deck construction error: Expected {DECK_SIZE} cards, got {len(completed_deck_names)}")
         # Simple padding/truncation fallback
         if len(completed_deck_names) < DECK_SIZE:
             completed_deck_names.extend(["Error Card Main"] * (DECK_SIZE - len(completed_deck_names)))
         else:
             completed_deck_names = completed_deck_names[:DECK_SIZE]

    final_main_counts = Counter(completed_deck_names)

    # Fetch images for unique main deck names
    main_unique_names = list(final_main_counts.keys())
    image_urls = get_card_image_urls(main_unique_names)

    # Structure main deck results
    completed_deck_list = []
    for name, count in final_main_counts.items():
        img_url = image_urls.get(name) if name != "Error Card Main" else None
        completed_deck_list.append({
            "name": name, "count": count, "image_url": img_url
        })

    current_app.logger.info(f"Main deck inference complete after refinement. Final count: {sum(c['count'] for c in completed_deck_list)}.")

    return completed_deck_list

# --- Inference Function (Sideboard Completion) ---
@torch.no_grad()
def complete_sideboard_inference(main_deck_list, current_sideboard_list, format):
    """
    Completes a sideboard based on a provided main deck and current sideboard cards.

    Args:
        main_deck_list (list): Completed 60-card main deck list:
                               [{'name': ..., 'count': ..., 'image_url': ...}, ...]
        current_sideboard_list (list): Current cards in the sideboard:
                                     [{'name': ..., 'count': ..., 'image_url': ...}, ...]
        format (str): The selected game format (e.g., 'modern').

    Returns:
        list: Completed sideboard list: [{'name': ..., 'count': ..., 'image_url': ...}]
    """
    current_app.logger.info(f"Running sideboard completion based on main deck and current sideboard: {current_sideboard_list}, format: {format}")

    if diffusion_model is None or clf_model is None or card_embeddings is None or idx_to_card is None:
         raise RuntimeError("Models or data not loaded properly for sideboard inference.")

    if SIDEBOARD_SIZE <= 0:
        current_app.logger.info("SIDEBOARD_SIZE is 0 or less, returning empty sideboard.")
        return []

    # --- Constants for Refinement ---
    MAX_REFINEMENT_ITERATIONS = 3

    # 1. Reconstruct main deck embedding tensor (context) and get main deck counts
    main_deck_embeddings = torch.zeros(1, DECK_SIZE, EMB_DIM, device=DEVICE)
    main_current_idx = 0
    main_total_cards = 0
    main_deck_counts = Counter() # Count cards in the main deck
    for card_info in main_deck_list:
        name = card_info["name"]
        count = card_info["count"]
        main_total_cards += count
        main_deck_counts[name] += count # Add to main deck counts
        try: vec = torch.tensor(card_embeddings[name], dtype=torch.float32, device=DEVICE)
        except KeyError: raise ValueError(f"Card '{name}' from input main deck not found in embeddings.")
        for _ in range(count):
            if main_current_idx < DECK_SIZE: main_deck_embeddings[0, main_current_idx] = vec; main_current_idx += 1
            else: raise ValueError("Provided main deck list exceeds DECK_SIZE.") # Should not happen if validation passed
    if main_total_cards != DECK_SIZE or main_current_idx != DECK_SIZE:
        raise ValueError(f"Provided main deck list does not contain exactly {DECK_SIZE} cards.")
    current_app.logger.info(f"Reconstructed main deck embedding tensor and counts for sideboard context.")

    # 2. Prepare *Initial* Known Sideboard Embeddings and Mask
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
                current_app.logger.warning(f"Input sideboard exceeds {SIDEBOARD_SIZE} cards. Truncating.")
                total_known_sb_count = SIDEBOARD_SIZE # Adjust count if truncated
                break
        if sb_current_idx >= SIDEBOARD_SIZE:
            break

    num_unknown_initial = SIDEBOARD_SIZE - total_known_sb_count
    if num_unknown_initial < 0:
        current_app.logger.warning(f"Known sideboard cards ({total_known_sb_count}) exceeded SIDEBOARD_SIZE ({SIDEBOARD_SIZE}). Assuming full sideboard provided.")
        num_unknown_initial = 0 # Cannot generate more cards
    # If the initial sideboard is already full, we skip generation/refinement
    elif num_unknown_initial == 0 and SIDEBOARD_SIZE > 0:
        current_app.logger.info("Initial sideboard is already full. Skipping generation.")
        # Directly format and return the initial sideboard
        initial_sb_counts = Counter(original_known_sb_names)
        initial_sb_unique_names = list(initial_sb_counts.keys())
        image_urls = get_card_image_urls(initial_sb_unique_names)
        completed_sideboard_list = []
        for name, count in initial_sb_counts.items():
            img_url = image_urls.get(name)
            completed_sideboard_list.append({"name": name, "count": count, "image_url": img_url})
        return completed_sideboard_list

    current_app.logger.info(f"Prepared initial known sideboard embeddings for {total_known_sb_count} cards. Initially generating {num_unknown_initial} cards.")

    # 3. Pre-calculate Main Deck Context Encoding (Done once)
    sb_context_encoded = diffusion_model.encode_main_deck_context(main_deck_embeddings)
    current_app.logger.info("Calculated sideboard context encoding from main deck.")

    # --- Iterative Refinement (Now only for 4-copy limit) ---
    current_x0_sb = None # Will hold the result of each diffusion run
    current_mask = initial_sb_known_mask.clone()
    current_known_emb = initial_sb_known_emb.clone()

    for refinement_iter in range(MAX_REFINEMENT_ITERATIONS + 1): # +1 because iter 0 is the initial run
        current_app.logger.info(f"--- Starting Sideboard Generation/Refinement Iteration {refinement_iter} ---")

        # Slots unknown at the start of *this iteration*
        unknown_mask_flat_this_iter = (current_mask[0, :, 0] == 0)
        num_unknown_this_iter = int(unknown_mask_flat_this_iter.sum().item())

        # If no slots are unknown in this iteration, stop early
        if num_unknown_this_iter == 0 and refinement_iter > 0: # Check needed only after first iter
             current_app.logger.info(f"Iteration {refinement_iter}: No unknown slots left to refine. Stopping.")
             break
        elif num_unknown_initial == 0 and refinement_iter == 0: # Handled above, but safe check
             current_app.logger.info("Initial sideboard was full, skipping refinement loop.")
             break # Should already have returned if SB was initially full


        # 4. Initialize Noise (only in unknown slots for this iteration)
        sb_x = torch.randn(1, SIDEBOARD_SIZE, EMB_DIM, device=DEVICE)
        # Apply the *current* mask and known embeddings
        sb_x = current_mask * current_known_emb + (1 - current_mask) * sb_x

        # 5. Run Sideboard Diffusion Sampling Loop for this iteration
        for t in reversed(range(TIMESTEPS)):
            t_tensor = torch.full((1,), t, device=DEVICE, dtype=torch.long)

            # Predict noise using the dedicated method, current mask, and pre-computed context
            sb_noise_pred = diffusion_model.predict_sideboard_noise(sb_x, t_tensor, current_mask, sb_context_encoded)

            # Diffusion update steps
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

            # IMPORTANT: Apply mask to preserve known cards for *this iteration* during update
            sb_x = current_mask * current_known_emb + (1 - current_mask) * sb_x_next

        current_x0_sb = sb_x # Resulting sideboard tensor for this iteration

        # --- Check 4-Copy Limit and Prepare for Next Iteration (if not last) ---
        if refinement_iter < MAX_REFINEMENT_ITERATIONS:
            current_app.logger.info(f"Iteration {refinement_iter}: Checking 4-copy limit across main deck and current sideboard result...")

            # Classify ALL slots from the current sideboard result
            iter_all_sb_logits = clf_model(current_x0_sb[0])
            iter_all_sb_predicted_indices = torch.argmax(iter_all_sb_logits, dim=1).cpu().numpy()
            # Get names, filtering out potential None results from idx_to_card safely
            iter_all_sb_predicted_names = []
            sb_name_idx_map = {} # Map absolute sideboard index to valid name
            for i, idx in enumerate(iter_all_sb_predicted_indices):
                name = idx_to_card.get(int(idx))
                if name is not None:
                    iter_all_sb_predicted_names.append(name)
                    sb_name_idx_map[i] = name
                # else: Skip index if classification failed

            iter_sb_counts = Counter(iter_all_sb_predicted_names)

            # Combine counts from main deck and current sideboard iteration
            combined_counts = main_deck_counts.copy() # Start with main deck counts
            combined_counts.update(iter_sb_counts) # Add sideboard counts

            indices_to_force_regenerate_4_copy = set()
            indices_to_force_regenerate_format = set()

            for card_name, combined_count in combined_counts.items():
                 # Safely get card data from the preloaded 'cards' dict
                card_data_list = cards.get(card_name)
                if not card_data_list: continue # Skip if card data not found
                card = card_data_list[0]
                supertypes = card.get("supertypes", [])

                if combined_count > 4 and "Basic" not in supertypes:
                    # Calculate how many copies are *in the sideboard* for this card
                    sb_count_for_card = iter_sb_counts.get(card_name, 0)
                    # Calculate how many *total* copies need to be removed (from SB)
                    num_to_remove_total = combined_count - 4

                    if sb_count_for_card > 0 and num_to_remove_total > 0:
                        current_app.logger.warning(f"Iteration {refinement_iter}: Card '{card_name}' found {combined_count} times (Main: {main_deck_counts.get(card_name, 0)}, SB: {sb_count_for_card}). Max 4 allowed (non-basic). Marking {min(sb_count_for_card, num_to_remove_total)} generated SB copies for regeneration.")

                        # Find all absolute *sideboard* indices for this card
                        current_sb_indices = [i for i, name in sb_name_idx_map.items() if name == card_name]

                        # Identify which of these SB indices were *generated* (not in initial SB mask)
                        # Use initial_sb_known_mask to know which were originally provided vs generated at any point
                        generated_sb_indices_for_card = {i for i in current_sb_indices if initial_sb_known_mask[0, i, 0] == 0.0}

                        # We only need to regenerate up to num_to_remove_total copies,
                        # and we can only regenerate copies that were actually generated (not user-provided)
                        num_to_regenerate = min(len(generated_sb_indices_for_card), num_to_remove_total)

                        # Add the required number of generated indices to the force regenerate set
                        indices_to_force_regenerate_4_copy.update(list(generated_sb_indices_for_card)[:num_to_regenerate])

            # Format Legality Check (for generated sideboard cards)
            for abs_idx, card_name in sb_name_idx_map.items(): # Iterate using the map of index -> valid name
                # Check if this slot was *generated* (not part of initial sideboard)
                if initial_sb_known_mask[0, abs_idx, 0] == 0.0:
                    card_data_list = cards.get(card_name)
                    if card_data_list:
                        card_data = card_data_list[0]
                        legalities = card_data.get("legalities", {})
                        supertypes = card_data.get("supertypes", [])
                        is_legal = legalities.get(format, "not_legal") == "Legal" or legalities.get(format, "not_legal") == "Restricted"
                        is_basic = "Basic" in supertypes
                        
                        if not is_legal and not is_basic:
                            current_app.logger.warning(f"Iteration {refinement_iter}: Generated sideboard card '{card_name}' (Index: {abs_idx}) is not legal in {format}. Marking for regeneration.")
                            indices_to_force_regenerate_format.add(abs_idx)
                    else:
                         current_app.logger.warning(f"Iteration {refinement_iter}: Could not find data for generated sideboard card '{card_name}' (Index: {abs_idx}) to check format legality.")

            # Combine regeneration sets
            final_absolute_indices_to_regenerate = indices_to_force_regenerate_4_copy.union(indices_to_force_regenerate_format)

            if indices_to_force_regenerate_4_copy:
                current_app.logger.info(f"Iteration {refinement_iter}: Marking {len(indices_to_force_regenerate_4_copy)} SB slots for regeneration due to 4-copy rule: {sorted(list(indices_to_force_regenerate_4_copy))}")
            if indices_to_force_regenerate_format:
                 current_app.logger.info(f"Iteration {refinement_iter}: Marking {len(indices_to_force_regenerate_format)} additional SB slots for regeneration due to format ({format}) legality: {sorted(list(indices_to_force_regenerate_format))}")
            #else:
            #     current_app.logger.info(f"Iteration {refinement_iter}: No cards exceeded 4-copy limit requiring forced regeneration in the sideboard.")


            # --- Check if Regeneration is Needed and Prepare for Next Iteration ---
            #absolute_indices_to_regenerate_list = sorted(list(indices_to_force_regenerate_4_copy))
            absolute_indices_to_regenerate_list = sorted(list(final_absolute_indices_to_regenerate))

            if not absolute_indices_to_regenerate_list and num_unknown_this_iter > 0:
                current_app.logger.info(f"Iteration {refinement_iter}: All {num_unknown_this_iter} generated SB cards satisfy 4-copy and format legality rules. Stopping refinement.")
                break # Stop refinement if all generated cards are good
            elif num_unknown_this_iter == 0:
                 # This case means no cards were generated, loop should terminate naturally or break earlier
                 pass
            elif not absolute_indices_to_regenerate_list:
                 # This case should not be reachable due to the break above, but good for safety
                 current_app.logger.info(f"Iteration {refinement_iter}: No SB slots marked for regeneration. Stopping refinement.")
                 break
            else:
                 current_app.logger.warning(f"Iteration {refinement_iter}: Preparing to regenerate {len(absolute_indices_to_regenerate_list)} SB slots due to 4-copy rule. Indices: {absolute_indices_to_regenerate_list}")

                 # Log details for slots being reconsidered
                 for abs_idx in absolute_indices_to_regenerate_list:
                     temp_logits = clf_model(current_x0_sb[0, abs_idx].unsqueeze(0))
                     temp_pred_idx = torch.argmax(temp_logits, dim=1).item()
                     temp_name = idx_to_card.get(temp_pred_idx, "Unknown Index")
                     # Determine reason
                     reason = "Unknown"
                     if abs_idx in indices_to_force_regenerate_4_copy and abs_idx in indices_to_force_regenerate_format:
                         reason = "4-Copy & Format"
                     elif abs_idx in indices_to_force_regenerate_4_copy:
                         reason = "4-Copy Rule"
                     elif abs_idx in indices_to_force_regenerate_format:
                         reason = "Format Legality"
                     current_app.logger.warning(f"  - Reconsidering SB Slot {abs_idx}: '{temp_name}' - Reason: {reason}")


                 # --- Prepare mask and known embeddings for the *next* iteration ---
                 next_mask = initial_sb_known_mask.clone() # Start with initial knowns
                 next_known_emb = initial_sb_known_emb.clone()

                 # Identify generated slots (absolute indices) that are *not* being regenerated
                 all_generated_indices_ever = set(torch.where(initial_sb_known_mask[0, :, 0] == 0)[0].cpu().numpy()) # All indices NOT initially known
                 valid_copy_generated_indices = list(all_generated_indices_ever - indices_to_force_regenerate_4_copy)

                 # Mark valid-copy generated cards as known for the next iteration
                 if valid_copy_generated_indices:
                     next_mask[0, valid_copy_generated_indices, 0] = 1.0
                     # Use their embeddings from the *current* result
                     next_known_emb[0, valid_copy_generated_indices] = current_x0_sb[0, valid_copy_generated_indices]

                 # Update current mask and known embeddings for the next loop
                 current_mask = next_mask
                 current_known_emb = next_known_emb
                 num_unknown_next = SIDEBOARD_SIZE - int(current_mask.sum().item())
                 current_app.logger.info(f"Preparing for Iteration {refinement_iter + 1}. Known SB cards: {int(current_mask.sum().item())}, Regenerating: {num_unknown_next}")

        else:
            current_app.logger.info(f"Iteration {refinement_iter}: Refinement loop finished (max iterations reached or stopped early).")


    # --- Final Classification and Formatting ---
    # Use the final state after all iterations
    final_x0_sb = current_x0_sb

    # If current_x0_sb is still None (e.g., SB was initially full and loop didn't run)
    # we need to construct the final list from the original input
    if final_x0_sb is None:
        if num_unknown_initial == 0 and SIDEBOARD_SIZE > 0:
            # This case was handled earlier, return the pre-formatted list
            # Re-calculate just in case (should match the earlier return)
            initial_sb_counts = Counter(original_known_sb_names)
            initial_sb_unique_names = list(initial_sb_counts.keys())
            image_urls = get_card_image_urls(initial_sb_unique_names)
            completed_sideboard_list = []
            for name, count in initial_sb_counts.items():
                img_url = image_urls.get(name)
                completed_sideboard_list.append({"name": name, "count": count, "image_url": img_url})
            current_app.logger.info("Returning initially full sideboard.")
            return completed_sideboard_list
        elif SIDEBOARD_SIZE == 0:
             return [] # Handled at the start
        else:
             # This indicates an unexpected state
             current_app.logger.error("Error: final_x0_sb is None but sideboard was not initially full.")
             return [] # Return empty on error


    # Identify slots that were *ultimately* generated (i.e., not in the initial mask)
    final_unknown_mask_flat = (initial_sb_known_mask[0, :, 0] == 0)

    generated_sb_names = []
    if final_unknown_mask_flat.sum() > 0:
        final_unknown_embeddings = final_x0_sb[0][final_unknown_mask_flat]

        current_app.logger.info(f"Classifying {final_unknown_embeddings.shape[0]} final generated sideboard embeddings...")
        final_logits_sb = clf_model(final_unknown_embeddings)
        final_predicted_indices = torch.argmax(final_logits_sb, dim=1).cpu().numpy()
        # Safely get names, handling potential index errors
        temp_generated_names = []
        for idx in final_predicted_indices:
            name = idx_to_card.get(int(idx))
            if name is not None:
                temp_generated_names.append(name)
            else:
                current_app.logger.warning(f"Classifier predicted unknown index {int(idx)} in final sideboard. Replacing with 'Error Card SB'.")
                temp_generated_names.append("Error Card SB") # Placeholder for unknown index
        generated_sb_names = temp_generated_names

    else:
         current_app.logger.info("No sideboard cards were generated (initial sideboard was full or SIDEBOARD_SIZE=0).")


    # 7. Combine Original Known SB Names and Final Generated SB Names
    completed_sb_names = original_known_sb_names + generated_sb_names

    # Ensure final count is exactly SIDEBOARD_SIZE (handle potential errors/truncation)
    if len(completed_sb_names) != SIDEBOARD_SIZE:
         current_app.logger.warning(f"Final sideboard construction resulted in {len(completed_sb_names)} cards, expected {SIDEBOARD_SIZE}. Padding/Truncating.")
         # Simple padding/truncation fallback
         if len(completed_sb_names) < SIDEBOARD_SIZE:
             completed_sb_names.extend(["Error Card SB"] * (SIDEBOARD_SIZE - len(completed_sb_names)))
         else:
             completed_sb_names = completed_sb_names[:SIDEBOARD_SIZE]


    # 8. Format Sideboard Results
    final_sb_counts = Counter(completed_sb_names)
    sb_unique_names = list(final_sb_counts.keys())
    image_urls = get_card_image_urls(sb_unique_names)

    completed_sideboard_list = []
    for name, count in final_sb_counts.items():
        # Handle potential "Error Card SB" placeholder
        img_url = image_urls.get(name) if name != "Error Card SB" else None
        completed_sideboard_list.append({
            "name": name, "count": count, "image_url": img_url
        })

    current_app.logger.info(f"Sideboard completion complete after refinement. Final count: {sum(c['count'] for c in completed_sideboard_list)}.")
    return completed_sideboard_list

# --- Scryfall Image Fetching ---
# Cache for image URLs to avoid repeated Scryfall lookups
image_cache = {}

def get_card_image_urls(card_names):
    """Fetches image URLs from Scryfall for a list of card names.
       Returns a dictionary mapping original full card names to either a string URL 
       or a dictionary {'front': url1, 'back': url2} for multi-face cards.
    """
    urls = {}
    names_to_fetch = set()
    name_map = {}

    for name in card_names:
        if name not in image_cache:
            names_to_fetch.add(name.split("//")[0])
            name_map[name.split("//")[0]] = name
        else:
            urls[name] = image_cache[name]

    if not names_to_fetch:
        return urls

    current_app.logger.info(f"Fetching {len(names_to_fetch)} card names from Scryfall...")

    # Construct the identifiers payload using the full original names
    identifiers = [{"name": name} for name in names_to_fetch]
    payload = {"identifiers": identifiers}

    try:
        response = requests.post(f"{SCRYFALL_API_BASE}/cards/collection", json=payload)
        response.raise_for_status() 
        data = response.json()

        found_map = {}
        # Process results for found cards
        if data and 'data' in data:
            for card_data in data['data']:
                scryfall_result_name = name_map.get(card_data.get('name'), card_data.get('name'))
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
                    elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                        image_info = card_data['image_uris']['normal']
                
                # If not multi-face (or faces lacked images), check top-level image_uris
                elif card_data.get('image_uris') and card_data['image_uris'].get('normal'):
                    image_info = card_data['image_uris']['normal']

                if scryfall_result_name:
                    found_map[scryfall_result_name] = image_info 

        for name, image_data in found_map.items():
            image_cache[name] = image_data
            urls[name] = image_data

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

        # Check if the input description matches a known card name
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

        # Get and validate format
        format_input = data.get('format', DEFAULT_FORMAT).lower()
        if format_input not in ALLOWED_FORMATS:
            return jsonify({"error": f"Invalid format specified. Allowed formats: {', '.join(ALLOWED_FORMATS)}"}), 400
        selected_format = format_input

        total_known = sum(c['count'] for c in known_cards)
        if total_known > DECK_SIZE:
            return jsonify({"error": f"Input deck has more than {DECK_SIZE} cards ({total_known})."}), 400
        if total_known == 0:
            return jsonify({"error": "Input deck cannot be empty."}), 400

        # --- Run Main Deck Inference Only ---
        completed_deck_list = run_inference(known_cards, selected_format)

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
             
        # Get and validate format
        format_input = data.get('format', DEFAULT_FORMAT).lower()
        if format_input not in ALLOWED_FORMATS:
            return jsonify({"error": f"Invalid format specified. Allowed formats: {', '.join(ALLOWED_FORMATS)}"}), 400
        selected_format = format_input
             
        # Basic validation of current sideboard list
        if not isinstance(current_sideboard_list, list):
             return jsonify({"error": "'current_sideboard' must be a list."}), 400
        current_sb_count = sum(c.get('count', 0) for c in current_sideboard_list)
        if current_sb_count > SIDEBOARD_SIZE:
             return jsonify({"error": f"'current_sideboard' cannot contain more than {SIDEBOARD_SIZE} cards."}), 400

        # --- Run Sideboard Completion Inference ---
        completed_sideboard_list = complete_sideboard_inference(main_deck_list, current_sideboard_list, selected_format)

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
    app.run(debug=True, host='0.0.0.0')