![Manamorphosis Logo](static/logo.png)

A diffusion model to complete Magic: The Gathering (MTG) decklists based on partially provided main decks and sideboards. It also includes a fully-featured deck editor and semantic search for cards using text embeddings.

Technical deep dive in the [announcement blog post](https://boggs.tech/posts/manamorphosis)

## Features

*   **Main Deck Completion:** Given a partial list of main deck cards, the AI generates the remaining cards to complete a 60-card deck.
*   **Sideboard Completion:** Given a completed 60-card main deck and an optional partial sideboard, the AI generates the remaining cards to complete a 15-card sideboard.
*   **Card Search:** Find cards based on a natural language description.
*   **Web Interface:** A Flask-based web application (`app.py`, `templates/index.html`, `static/script.js`) provides an easy-to-use interface for:
    *   Building main decks and sideboards.
    *   Searching for cards by name or description.
    *   Main deck completion and sideboard generation.
    *   Viewing card images fetched from Scryfall.
    *   Exporting decklists.

    *Main Deck & Sideboard View:*
    ![Main Interface Screenshot](static/editor.png)

    *Card Search Results:*
    ![Search Results Screenshot](static/search.png)

## Demo Video

https://github.com/user-attachments/assets/c7a1d50a-eb2f-442d-acff-7c1ea031357b

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
*   BeautifulSoup4
*   aiohttp
*   tqdm

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

## Running the Application

Once the data preparation steps (at least embeddings and classifier training) are complete and the diffusion model is trained (or you have a pre-trained one), you can run the Flask web application:

```bash
python app.py
```

This will start a development server (usually at `http://127.0.0.1:5000` or `http://localhost:5000`). Open this URL in your web browser.

## Deployment

The demo is deployed with the web interface and the inference model running separately to prevent excessive billing from idle GPUs.

*   **Web Application (DigitalOcean App Platform):**
    *   The Flask application (`app.py`) and its associated frontend (`templates/`, `static/`) are deployed as a web service on the DigitalOcean App Platform.
    *   It handles user interactions, deck building, card search, and makes requests to the inference server.
    *   Configuration details can be found in the `deployment/app/` directory.

*   **Inference Server (Runpod Serverless):**
    *   The computationally intensive diffusion model inference (main deck and sideboard completion) runs on Runpod Serverless.
    *   This provides a cost-effective GPU-powered endpoint that scales on demand.
    *   The necessary handler (`handler.py`) and Docker setup for Runpod can be found in the `deployment/inference/` directory.
    *   The web application needs to be configured with the URL of the deployed Runpod endpoint.

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
