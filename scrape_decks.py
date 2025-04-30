import os
import re
import json
import pickle
import random
import asyncio
import aiohttp
import difflib
from bs4 import BeautifulSoup

# Configuration
BASE_URL = "https://www.mtgtop8.com"
SEARCH_URL = f"{BASE_URL}/search"
DECK_URL = f"{BASE_URL}/mtgo"  # Endpoint that returns plain text
DECKS_FOLDER = "./data/decks"

# Format codes used by MTGTop8
FORMAT_CODES = {
    "Standard": "ST",
    "Modern": "MO",
    "Legacy": "LE",
    "Vintage": "VI",
    "Pioneer": "PI",
    "Pauper": "PAU",
    "Commander": "cEDH"
}

def normalize_card_name(name):
    """Normalize a card name for better matching."""
    # First decode any unicode escape sequences
    try:
        name = name.encode('latin-1').decode('unicode-escape')
    except (UnicodeError, AttributeError):
        pass  # If this fails, just use the original name
    
    # Remove any text in parentheses (often set codes or variations)
    name = re.sub(r'\s*\([^)]*\)', '', name)
    
    # Remove punctuation and special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Convert to lowercase and strip whitespace
    name = name.lower().strip()
    
    # Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name)
    
    return name

def find_double_sided_card(card_name, card_vocab):
    """Check if this card is the first side of a double-sided card in the vocabulary."""
    normalized_name = normalize_card_name(card_name)
    
    # Look for cards that start with this name followed by " // "
    for vocab_name in card_vocab.keys():
        if '//' in vocab_name:
            first_side = vocab_name.split('//')[0].strip()
            if normalize_card_name(first_side) == normalized_name:
                return vocab_name
    
    return None

def find_closest_match(card_name, card_vocab, threshold=0.85):
    """Find the closest matching card name in the vocabulary."""
    normalized_name = normalize_card_name(card_name)
    
    # First try direct match after normalization
    for vocab_name in card_vocab.keys():
        if normalize_card_name(vocab_name) == normalized_name:
            return vocab_name
    
    # Check if this is a double-sided card
    double_sided_match = find_double_sided_card(card_name, card_vocab)
    if double_sided_match:
        return double_sided_match
    
    # If no exact match, try fuzzy matching
    matches = difflib.get_close_matches(
        normalized_name,
        [normalize_card_name(name) for name in card_vocab.keys()],
        n=1,
        cutoff=threshold
    )
    
    if matches:
        # Find the original card name that matched
        for vocab_name in card_vocab.keys():
            if normalize_card_name(vocab_name) == matches[0]:
                return vocab_name
    
    return None

def load_card_vocab(embeddings_path="card_embeddings.pkl"):
    """Load the card data (including names as keys) from the embeddings pickle file."""
    try:
        with open(embeddings_path, "rb") as f:
            return pickle.load(f) # Return the full dictionary
    except Exception as e:
        print(f"Error loading card data from embeddings: {e}")
        return {}

def create_search_form_data(format_code, page=1):
    """Create the form data for the search request."""
    form_data = {
        "current_page": str(page),
        "event_titre": "",
        "deck_titre": "",
        "player": "",
        "format": format_code,
        "compet_check[P]": "1",
        "compet_check[M]": "1",
        "compet_check[C]": "1",
        "compet_check[R]": "1",
        "MD_check": "1",
        "SB_check": "1" if format_code != "cEDH" else "0",
        "cards": "",
        "date_start": "",
        "date_end": "",
    }
    
    # Add archetype_sel for all formats
    for fmt_code in FORMAT_CODES.values():
        form_data[f"archetype_sel[{fmt_code}]"] = ""
    
    return form_data

async def get_deck_links_from_page(format_code, page=1, session=None, semaphore=None):
    """Get deck links from a search results page using BeautifulSoup."""
    form_data = create_search_form_data(format_code, page)
    
    try:
        async with semaphore:
            async with session.post(SEARCH_URL, data=form_data) as response:
                response.raise_for_status()
                html_content = await response.text()
        
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all links in elements with class 'Stable' where href contains 'event?' and 'd='
        deck_links = []
        for link in soup.select('.Stable a'):
            href = link.get('href')
            if href and 'event?' in href and 'd=' in href:
                deck_links.append(href)
        
        print(f"    Found {len(deck_links)} deck links on page {page}")
        return deck_links
    except Exception as e:
        print(f"Error fetching search page {page} for format {format_code}: {e}")
        return []

def extract_deck_id(deck_url):
    """Extract the deck ID from the URL."""
    match = re.search(r"d=(\d+)", deck_url)
    if match:
        return match.group(1)
    return None

async def get_deck_data(deck_url, card_vocab, format_name, session=None, semaphore=None):
    """Fetch and parse a deck directly from the text version."""
    deck_id = extract_deck_id(deck_url)
    if not deck_id:
        return None
        
    # Use the direct MTGO format URL that gives text
    text_url = f"{DECK_URL}?d={deck_id}"
    
    try:
        async with semaphore:
            async with session.get(text_url) as response:
                response.raise_for_status()
                raw_text = await response.text()
        
        raw_text = raw_text.strip()
        if not raw_text:
            return None
            
        # Process line by line
        lines = raw_text.split("\n")
        main_deck = []
        sideboard = []
        in_sideboard = False
        has_unfixable_cards = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.lower() == "sideboard":
                in_sideboard = True
                continue
                
            # Parse card count and name
            match = re.match(r"^(\d+)\s+(.+)$", line)
            if match:
                count = int(match.group(1))
                card_name = match.group(2).strip()
                
                # Try to decode unicode escape sequences
                try:
                    decoded_name = card_name.encode('latin-1').decode('unicode-escape')
                    if decoded_name != card_name:
                        print(f"  Decoded Unicode: {card_name} -> {decoded_name}")
                        card_name = decoded_name
                except (UnicodeError, AttributeError):
                    pass  # If decoding fails, keep the original name
                
                # Check if card name exists in vocabulary
                if card_name in card_vocab:
                    card_entry = {"name": card_name, "count": count}
                    if in_sideboard:
                        sideboard.append(card_entry)
                    else:
                        main_deck.append(card_entry)
                else:
                    # Try to find a close match
                    fixed_name = find_closest_match(card_name, card_vocab)
                    if fixed_name:
                        card_entry = {"name": fixed_name, "count": count}
                        if in_sideboard:
                            sideboard.append(card_entry)
                        else:
                            main_deck.append(card_entry)
                    else:
                        # Can't fix this card name, so the deck is invalid
                        has_unfixable_cards = True
                        print(f"  Cannot fix card name: {card_name}")
                        break
        
        if has_unfixable_cards:
            return None
        
        # Create deck JSON object with the format we already know
        deck_json = {
            "format": format_name.lower(),
            "cards": main_deck,
            "sideboard": sideboard
        }
        
        return deck_json
        
    except Exception as e:
        print(f"Error fetching deck data from {text_url}: {e}")
        return None

async def scrape_format(format_name, card_vocab, session, semaphore, max_pages=10, 
                  max_decks_per_format=100, delay_between_requests=1):
    """Scrape decks for a specific format."""
    format_code = FORMAT_CODES.get(format_name)
    if not format_code:
        print(f"Unknown format: {format_name}")
        return 0
        
    print(f"Scraping {format_name} (code: {format_code})...")
    decks_saved = 0
    
    for page in range(1, max_pages + 1):
        if decks_saved >= max_decks_per_format:
            break
            
        print(f"  Processing page {page}...")
        deck_links = await get_deck_links_from_page(format_code, page, session, semaphore)
        
        if not deck_links:
            print(f"  No deck links found on page {page}. Moving to next format.")
            break
            
        for deck_url in deck_links:
            if decks_saved >= max_decks_per_format:
                break
                
            deck_id = extract_deck_id(deck_url)
            if not deck_id:
                continue
                
            # Add a delay to avoid overwhelming the server
            await asyncio.sleep(delay_between_requests)
            
            print(f"    Fetching deck {deck_id}...")
            deck_json = await get_deck_data(deck_url, card_vocab, format_name, session, semaphore)
            
            if deck_json:
                os.makedirs(DECKS_FOLDER, exist_ok=True)
                    
                # Save the deck as JSON with ensure_ascii=False to preserve Unicode characters
                output_file = os.path.join(DECKS_FOLDER, f"{format_name.lower()}_{deck_id}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(deck_json, f, indent=2, ensure_ascii=False)
                
                decks_saved += 1
                print(f"    Saved deck {deck_id} ({decks_saved}/{max_decks_per_format})")
            else:
                print(f"    Skipping deck {deck_id} - unfixable card names or failed to fetch")
            
            # Random additional delay to simulate human browsing
            await asyncio.sleep(random.uniform(0.5, 1.5))
    
    return decks_saved

async def main_async():
    """Async main function to scrape decks from MTGTop8."""
    print("MTGTop8 Deck Scraper (Async Version)")
    print("====================================")
    
    # Load card vocabulary
    card_vocab = load_card_vocab()
    if not card_vocab:
        print("Failed to load card data from embeddings. Cannot continue.")
        return
    
    print(f"Loaded vocabulary with {len(card_vocab)} cards")
    
    # Create the decks folder if it doesn't exist
    os.makedirs(DECKS_FOLDER, exist_ok=True)
    
    # Create a semaphore to limit concurrent requests to 3
    semaphore = asyncio.Semaphore(3)
    
    # You can customize which formats to scrape and how many decks per format
    formats_to_scrape = {
        "Standard": 10000,
        "Modern": 10000,
        "Legacy": 10000,
        "Vintage": 5000,
        "Pioneer": 10000,
        "Pauper": 5000
    }
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for each format
        tasks = []
        for format_name, max_decks in formats_to_scrape.items():
            task = scrape_format(
                format_name, 
                card_vocab,
                session,
                semaphore,
                max_pages=2000,
                max_decks_per_format=max_decks,
                delay_between_requests=1
            )
            tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks)
        
        # Process results
        total_decks = sum(results)
        print(f"\nTotal decks scraped: {total_decks}")
        print(f"All decks saved to: {os.path.abspath(DECKS_FOLDER)}")

def main():
    """Main function that runs the async event loop."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 