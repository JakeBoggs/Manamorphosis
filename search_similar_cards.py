import pickle
import argparse
from scipy.spatial.distance import cosine

def parse_args():
    parser = argparse.ArgumentParser(description="Search for MTG cards similar to a given card")
    parser.add_argument("card_name", help="Name of the card to find similar matches for")
    parser.add_argument("--top", type=int, default=10, help="Number of similar cards to return (default: 10)")
    parser.add_argument("--embeddings", default="data/card_embeddings.pkl", help="Path to card embeddings file (default: card_embeddings.pkl)")
    return parser.parse_args()

def load_embeddings(embeddings_path):
    print(f"Loading card embeddings from {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        return pickle.load(f)

def find_similar_cards(query_card, embeddings_map, top_n=10):
    if query_card not in embeddings_map:
        print(f"Error: Card '{query_card}' not found in embeddings.")
        print("Use --list-cards to see all available card names.")
        return []
    
    query_embedding = embeddings_map[query_card]
    similarities = []
    
    # Calculate cosine similarity between query card and all other cards
    for card_name, embedding in embeddings_map.items():
        if card_name == query_card:
            continue
        
        # Lower value = more similar for cosine distance
        similarity = 1 - cosine(query_embedding, embedding)
        similarities.append((card_name, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N similar cards
    return similarities[:top_n]

def main():
    args = parse_args()
    embeddings_map = load_embeddings(args.embeddings)

    query_card = args.card_name
    top_n = args.top
    
    print(f"Finding cards similar to: {query_card}")
    similar_cards = find_similar_cards(query_card, embeddings_map, top_n)
    
    if similar_cards:
        print(f"\nTop {len(similar_cards)} similar cards:")
        for i, (card_name, similarity) in enumerate(similar_cards, 1):
            print(f"{card_name} (similarity: {similarity:.4f})")

if __name__ == "__main__":
    main() 