![Manamorphosis Logo](static/logo.png)

A diffusion model to complete Magic: The Gathering (MTG) decklists based on partially provided main decks and sideboards. It also includes a semantic search for cards using text embeddings.

## Features

*   **Main Deck Completion:** Given a partial list of main deck cards, the AI generates the remaining cards to complete a 60-card deck.
*   **Sideboard Completion:** Given a completed 60-card main deck and an optional partial sideboard, the AI generates the remaining cards to complete a 15-card sideboard.
*   **Card Search:** Find cards based on a natural language description (uses Doc2Vec similarity).
*   **Web Interface:** A Flask-based web application provides an easy-to-use interface for deck completion and card search.

## Model Details

*   **Embeddings:** Card text (including mana cost, type, power/toughness, and rules text) is preprocessed and embedded into a 128-dimension vector space using a Doc2Vec model (`train_embedding_model.py`). This captures semantic similarities between cards based on their text.
*   **Classifier:** A simple linear layer (`train_embedding_classifier.py`) is trained to map these 128-dimension embeddings back to unique card indices. This is crucial during the reverse diffusion process to identify the specific card corresponding to a generated embedding.
*   **Diffusion Model:** A transformer-based architecture (`diffusion_model.py`) is trained to perform denoising diffusion on sets of card embeddings.
    *   **Forward Process (Training):** Starting with a real deck (represented by card embeddings `x0`), noise is gradually added over `T` timesteps to produce noisy versions `xt`. The model learns to predict the noise added at each timestep `t`.
    *   **Reverse Process (Inference):** Starting from pure noise (`xT`), the trained model iteratively predicts the noise and subtracts it, gradually denoising the embeddings back towards a coherent deck (`x0`).
    *   **Transformer Architecture:** The model uses transformer encoder/decoder layers to process the sequence of card embeddings. Time embeddings (sinusoidal) and mask embeddings are added to the input embeddings to inform the model about the current timestep and which cards are known vs. unknown. There are no positional embeddings, allowing the model to process decks as unordered sets.
    *   **Conditioning:** During training and inference, a binary mask indicates which card slots are "known" (provided by the user) and which should be generated. The model's loss function focuses on predicting noise only for the unknown slots, and during inference, the known card embeddings are reapplied at each step to guide the generation.
    *   **Main Deck & Sideboard:** The model has distinct paths:
        *   The main deck is processed by a transformer encoder to predict noise.
        *   The final, denoised main deck embedding (`x0_main`) is then encoded to create a context vector.
        *   The sideboard embeddings are processed by a transformer decoder, conditioned on the main deck context, to predict sideboard noise.

## Requirements

*   Python 3.8+
*   PyTorch (with CUDA support recommended for faster training/inference)
*   Flask
*   Gensim
*   NLTK
*   Requests
*   Scipy
*   Numpy
*   BeautifulSoup4 (for scraping)
*   aiohttp (for scraping)
*   tqdm (for progress bars)

You can install the required Python packages using pip:

```bash
pip install torch flask gensim nltk requests scipy numpy beautifulsoup4 aiohttp tqdm
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/JakeBoggs/Manamorphosis.git
    cd Manamorphosis
    ```

2.  **Download AtomicCards.json:**
    *   Download the `AtomicCards.json` file from [MTGJSON](https://mtgjson.com/downloads/all-files/#atomiccards).
    *   Place it in a `data/` directory within the project root (`./data/AtomicCards.json`).

3.  **Prepare Data and Models:**
    *   Follow the steps in the "Data Preparation" section below.

4.  **Run the Application:**
    *   See the "Running the Application" section.

## Data Preparation

NOTE: Pre-trained models (embedding, classifier, and diffusion) for 60 card constructed formats are available for download to skip the training steps. The model was trained on approximately 47,000 decklists scraped from MTGTop8. You can find the models here: [Google Drive Folder](https://drive.google.com/drive/folders/1ZvVbUGXa8FGzL97lplQGea2Ech7yfR-0?usp=sharing)

The following scripts need to be run *if you are not using pre-trained models* in order to prepare the necessary data and train the models.

1.  **Train Text Embeddings:**
    *   This script uses `AtomicCards.json` to train a Doc2Vec model on card text and saves card name -> embedding mappings.
    *   Ensure `data/AtomicCards.json` exists.
    *   Run: `python train_embedding_model.py`
    *   This will create:
        *   `models/embedding_model`: The Doc2Vec model.
        *   `data/card_embeddings.pkl`: A dictionary mapping card names to their vector embeddings.
        *   `data/cards.txt`: A text file containing the processed card text corpus (optional).

2.  **Train Card Classifier:**
    *   This script trains a simple linear classifier to predict a card's index based on its embedding. This is used during diffusion inference to map generated embeddings back to card names.
    *   Requires `data/card_embeddings.pkl` from the previous step.
    *   Run: `python train_embedding_classifier.py`
    *   This will create:
        *   `models/card_classifier.pt`: The trained classifier model and mappings (card_to_idx, idx_to_card).

3.  **Scrape Decklists:**
    *   This script scrapes decklists from MTGTop8.com to create a dataset for training the diffusion model.
    *   Requires `data/card_embeddings.pkl` to validate card names.
    *   Run: `python scrape_decks.py`
    *   This will create JSON files for each scraped deck in the `./data/decks/` directory (default).

4.  **Train Diffusion Model:**
    *   This script trains the main diffusion transformer model.
    *   Requires:
        *   `data/card_embeddings.pkl`
        *   `models/card_classifier.pt` (for card-to-index mapping)
        *   A directory of deck data (e.g., `./data/decks/` from the scraping step). The default path in the script is `./data/new_decks`, adjust the `--deck_dir` argument if needed.
    *   Run: `python diffusion_model.py --deck_dir ./data/decks --epochs 500 --batch_size 16 --lr 2e-5` (adjust parameters as needed, especially `--deck_dir`).
    *   This will create (or update):
        *   `models/diffusion_model.pth`: The trained diffusion model checkpoint.

## Diffusion Model Architecture (`diffusion_model.py:DiffusionModel`)

The model is composed of several interconnected transformer blocks and MLPs. Despite its complexity, the architecture is relatively efficient, containing approximately 56 million parameters, and was successfully trained on a consumer-grade laptop GPU (NVIDIA GeForce RTX 3050 with 4GB VRAM).

1.  **Inputs:**
    *   `x_t` / `sb_x_t`: Noisy card embeddings for the main deck / sideboard at timestep `t` (Shape: `[Batch, Deck/SB Size, EMB_DIM]`). The embedding dimension (`EMB_DIM`) is 128.
    *   `t`: Current diffusion timestep (Shape: `[Batch]`).
    *   `mask` / `sb_mask`: Binary masks indicating known card positions (1.0 for known, 0.0 for unknown) (Shape: `[Batch, Deck/SB Size, 1]`).
    *   `x0` (Training only): Original main deck embeddings used for sideboard context (Shape: `[Batch, Deck Size, EMB_DIM=128]`).
    *   `main_deck_context_encoded` (Inference only): Pre-calculated context from the main deck.

2.  **Time Embeddings:**
    *   `sinusoidal_embedding`: Generates a fixed sinusoidal embedding for the timestep `t` (dimension `EMB_DIM=128`).
    *   `main_time_mlp` / `sb_time_mlp`: Separate MLPs (Linear -> SiLU -> Linear) process the sinusoidal embedding to create time-specific bias vectors for the main deck and sideboard paths, respectively. Output shape: `[Batch, EMB_DIM=128]`. These are expanded to match the deck/sideboard size.

3.  **Mask Embeddings:**
    *   `main_mask_mlp` / `sb_mask_mlp`: Separate MLPs (Linear -> SiLU -> Linear) process the binary mask input to create embeddings representing known/unknown positions. Output shape: `[Batch, Deck/SB Size, EMB_DIM=128]`.

4.  **Main Deck Path (Encoder):**
    *   The input `x_t` is combined with `main_t_emb` and `main_mask_emb` via addition.
    *   `main_input_proj`: A Linear layer projects the combined embeddings from `EMB_DIM=128` to the model's internal `model_dim=384`.
    *   `main_transformer_encoder`: A standard `nn.TransformerEncoder` (using `nn.TransformerEncoderLayer`) with `layers=8` processes the projected sequence.
    *   `main_output_proj`: A Linear layer projects the result back from `model_dim=384` to `EMB_DIM=128`, predicting the noise (`main_noise_pred`).

5.  **Sideboard Context Path (Encoder):**
    *   Takes the *original* main deck embeddings `x0` (during training) or the *final denoised* main deck embeddings (during inference).
    *   `sb_context_input_proj`: Linear layer projects from `EMB_DIM=128` to `model_dim=384`.
    *   `sideboard_context_encoder`: A standard `nn.TransformerEncoder` with **1 layer** processes the projected main deck sequence to create the context (`sb_context_encoded`).

6.  **Sideboard Decoder Path:**
    *   The input `sb_x_t` is combined with `sb_decoder_t_emb` and `sb_decoder_mask_emb` via addition.
    *   `sb_input_proj`: Linear layer projects combined sideboard embeddings from `EMB_DIM=128` to `model_dim=384`.
    *   `sb_transformer_decoder`: A standard `nn.TransformerDecoder` (using `nn.TransformerDecoderLayer`) with **1 layer**. It receives the projected sideboard sequence as `tgt` and the `sb_context_encoded` as `memory`.
    *   `sb_transformer_output`: The output sequence from the decoder is passed through *another* standard `nn.TransformerEncoder` with `sb_layers=8` layers for further processing.
    *   `sb_output_proj`: Linear layer projects the result back from `model_dim=384` to `EMB_DIM=128`, predicting the sideboard noise (`sb_noise_pred`).

### Training Process Details

*   **Data Filtration (`diffusion_model.py:DeckDataset`):**
    *   When loading decks for training, only decks with exactly 60 main deck cards and 15 sideboard cards are included.
    *   Decks containing card names not found in the pre-computed embeddings (`card_embeddings.pkl`) or the classifier mapping (`card_classifier.pt`) are skipped to ensure data consistency.
*   **Mask Generation (`diffusion_model.py:DiffusionTrainer`):**
    *   During each training step, for every deck in the batch, multiple masks (`masks_per_deck` parameter) are generated for both the main deck and the sideboard.
    *   **Main Deck Masking:**
        *   The total number of possible known cards `k` (from 1 to 59) is partitioned across the `masks_per_deck`. Each generated mask samples a `k` value from one of these partitions. This ensures the model sees a diverse range of `k` values.
        *   For a chosen `k`, the specific card positions to keep known (marked as 1.0 in the mask) are selected based on card popularity and a probabilistic approach:
            *   Unique cards available to be masked are identified.
            *   These unique cards are sampled *without replacement* based on weights derived from their pre-calculated popularity (less popular cards are slightly favored).
            *   The algorithm iterates through the weighted, shuffled unique cards.
            *   For each card, there's an 85% chance it attempts to mask *all* available copies of that card (up to the remaining `k` needed) and a 15% chance it attempts to mask a *random number* of available copies (between 1 and the number available, up to the remaining `k` needed).
            *   This process continues until exactly `k` positions are marked as known.
    *   **Sideboard Masking:**
        *   A random target `k` value is chosen between 1 and 14 (`SIDEBOARD_SIZE - 1`).
        *   There's a 50% chance this target `k` is set to 0.
        *   The same card selection logic (using popularity weighting and the 85%/15% split) as the main deck is then used to select `k` sideboard card positions to keep known. If `k` is 0, no sideboard cards are marked as known.

## Running the Application

Once the data preparation steps (at least embeddings and classifier training) are complete and the diffusion model is trained (or you have a pre-trained one), you can run the Flask web application:

```bash
python app.py
```

This will start a development server (usually at `http://127.0.0.1:5000` or `http://localhost:5000`). Open this URL in your web browser.

For production, use a proper WSGI server like Gunicorn or Waitress:

```bash
# Example using Waitress
pip install waitress
waitress-serve --host 0.0.0.0 --port 5000 app:app
```

## Utilities

### Finding Similar Cards (Testing Embeddings)

You can test the quality of the learned card embeddings or simply find cards similar to a given card using the `search_similar_cards.py` script. This script calculates the cosine similarity between the embedding of a specified card and all other cards in the vocabulary.

**Requirements:**

*   `data/card_embeddings.pkl` (generated by `train_embedding_model.py`)

**Usage:**

```bash
python search_similar_cards.py "[Card Name]" [--top N] [--embeddings PATH]
```

*   **`"[Card Name]"`:** The exact name of the card you want to find similar cards for (must be present in the embeddings).
*   **`--top N` (Optional):** The number of similar cards to display (default: 10).
*   **`--embeddings PATH` (Optional):** Path to the `card_embeddings.pkl` file (default: `data/card_embeddings.pkl`).

**Example:**

```bash
python search_similar_cards.py "Lightning Bolt" --top 5
```

This will load the embeddings and print the top 5 cards most similar to "Lightning Bolt" based on their text embeddings.

## Folder Structure

```
.
├── data/
│   ├── AtomicCards.json          # (Needs download) Raw MTGJSON card data
│   ├── card_embeddings.pkl       # Generated by train_embedding_model.py
│   ├── cards.txt                 # Generated by train_embedding_model.py (optional)
│   └── decks/                    # Generated by scrape_decks.py (or provide your own)
│       └── *.json
├── models/
│   ├── embedding_model           # Generated by train_embedding_model.py
│   ├── card_classifier.pt        # Generated by train_embedding_classifier.py
│   └── diffusion_model.pth       # Generated by diffusion_model.py
├── static/
│   ├── style.css                 # CSS for the web interface
│   ├── icon.png                  # Small icon
│   └── logo.png                  # Logo image
├── templates/
│   └── index.html                # HTML for the web interface
├── app.py                        # Flask web application
├── diffusion_model.py            # Diffusion model definition and training script
├── scrape_decks.py               # Script to scrape decklists
├── search_similar_cards.py       # Utility script to find similar cards via embeddings
├── train_embedding_model.py      # Script to train Doc2Vec embeddings
├── train_embedding_classifier.py # Script to train the embedding->card classifier
└── README.md                     # This file
``` 