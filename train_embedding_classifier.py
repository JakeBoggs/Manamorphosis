import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CardClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CardClassifier, self).__init__()
        self.network = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        return self.network(x)

def prepare_data(embeddings_map):
    print("Preparing training data...")
    # Create a mapping from card names to indices
    card_names = sorted(embeddings_map.keys())
    card_to_idx = {card: idx for idx, card in enumerate(card_names)}
    idx_to_card = {idx: card for idx, card in enumerate(card_names)}
    
    # Create X (embeddings) and y (indices)
    X = []
    y = []
    
    for card_name, embedding in embeddings_map.items():
        X.append(embedding)
        y.append(card_to_idx[card_name])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    
    return X_tensor, y_tensor, card_to_idx, idx_to_card

def train_model(model, X, y, batch_size, epochs, lr, weight_decay):
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Set up data loader
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate batch accuracy for progress tracking
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            pbar.set_postfix({"loss": running_loss / (pbar.n + 1), "acc": correct / total})
    
    print("Training complete!")
    return model

def main():    
    # Load card embeddings
    print(f"Loading card embeddings...")
    with open('data/card_embeddings.pkl', 'rb') as f:
        embeddings_map = pickle.load(f)
    
    # Prepare data
    X, y, card_to_idx, idx_to_card = prepare_data(embeddings_map)
    
    # Initialize model
    embedding_dim = X.shape[1]
    num_classes = len(card_to_idx)
    print(f"Creating model with {embedding_dim} input dimensions and {num_classes} output classes...")
    model = CardClassifier(embedding_dim, num_classes).to(device)
    
    # Train model
    print("Training model...")
    model = train_model(
        model, X, y, 128, 20, 0.001, 1e-4
    )
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'card_to_idx': card_to_idx,
        'idx_to_card': idx_to_card,
        'embedding_dim': embedding_dim,
        'num_classes': num_classes
    }, 'models/card_classifier.pt')
    
    print("Done!")

if __name__ == "__main__":
    main() 