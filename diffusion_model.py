import argparse
import json
import math
import tqdm
import pickle
import random
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

EMB_DIM = 128
DECK_SIZE = 60
SIDEBOARD_SIZE = 15
TIMESTEPS = 1000

class DeckDataset(Dataset):
    def __init__(self, deck_dir: str, embeddings_path: str, classifier_path: str):
        with open(embeddings_path, "rb") as f:
            raw = pickle.load(f)
        self.card2vec = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}

        if not os.path.exists(classifier_path):
            raise FileNotFoundError(f"Classifier checkpoint not found at {classifier_path}. Please train the classifier first.")
        checkpoint = torch.load(classifier_path, map_location='cpu') # Load to CPU initially
        self.card_to_idx = checkpoint['card_to_idx']

        self.decks_embeds = []
        self.decks_indices = []
        self.sideboard_embeds = []
        self.sideboard_indices = []

        card_counts = {}

        for path in Path(deck_dir).glob("*.json"):
            try:
                deck = json.load(open(path, encoding="utf-8"))
                cards_in_this_deck = set() # Track unique cards per deck file

                # --- Main Deck Processing ---
                main_deck_cards = deck.get("cards", [])
                total_main = sum(e.get("count", 0) for e in main_deck_cards)
                if total_main != DECK_SIZE:
                    print(f"Warning: Deck {path.name} main deck size is {total_main}, expected {DECK_SIZE}. Skipping.")
                    continue

                main_cards_embeds = []
                main_cards_indices = []
                for e in main_deck_cards:
                    card_name = e["name"]
                    cards_in_this_deck.add(card_name) # Add to set for this deck
                    vec = self.card2vec.get(card_name)
                    if vec is None:
                        raise ValueError(f"Main deck card '{card_name}' missing in embeddings")
                    card_idx = self.card_to_idx.get(card_name)
                    if card_idx is None:
                        raise ValueError(f"Main deck card '{card_name}' missing in card_to_idx mapping")

                    count = e["count"]
                    main_cards_embeds.extend([vec] * count)
                    main_cards_indices.extend([card_idx] * count)

                if len(main_cards_embeds) != DECK_SIZE: # Double check
                    print(f"Warning: Deck {path.name} ended with {len(main_cards_embeds)} main deck cards after processing. Expected {DECK_SIZE}. Skipping.")
                    continue

                # --- Sideboard Processing ---
                sideboard_cards = deck.get("sideboard", [])
                total_sb = sum(e.get("count", 0) for e in sideboard_cards)
                if total_sb != SIDEBOARD_SIZE:
                    print(f"Warning: Deck {path.name} sideboard size is {total_sb}, expected {SIDEBOARD_SIZE}. Skipping.")
                    continue

                sb_cards_embeds = []
                sb_cards_indices = []
                for e in sideboard_cards:
                    card_name = e["name"]
                    cards_in_this_deck.add(card_name) # Add to set for this deck
                    vec = self.card2vec.get(card_name)
                    if vec is None:
                        raise ValueError(f"Sideboard card '{card_name}' missing in embeddings")
                    card_idx = self.card_to_idx.get(card_name)
                    if card_idx is None:
                        raise ValueError(f"Sideboard card '{card_name}' missing in card_to_idx mapping")

                    count = e["count"]
                    sb_cards_embeds.extend([vec] * count)
                    sb_cards_indices.extend([card_idx] * count)

                if len(sb_cards_embeds) != SIDEBOARD_SIZE: # Double check
                    print(f"Warning: Deck {path.name} ended with {len(sb_cards_embeds)} sideboard cards after processing. Expected {SIDEBOARD_SIZE}. Skipping.")
                    continue

                # Append successful deck
                self.decks_embeds.append(torch.stack(main_cards_embeds))
                self.decks_indices.append(torch.tensor(main_cards_indices, dtype=torch.long))
                self.sideboard_embeds.append(torch.stack(sb_cards_embeds))
                self.sideboard_indices.append(torch.tensor(sb_cards_indices, dtype=torch.long))

                # Increment counts for unique cards found in this deck/sideboard
                for card_name in cards_in_this_deck:
                    card_counts[card_name] = card_counts.get(card_name, 0) + 1

            except ValueError as e:
                print(f"Skipping deck {path.name}: {e}")
            except Exception as e:
                print(f"Error processing deck {path.name}: {e}")

        # --- Calculate Card Popularity ---
        if not card_counts:
            print("Warning: No cards found to calculate popularity.")
            self.card_popularity = {}
        else:
            highest_sum = max(card_counts.values()) if card_counts else 1 # Avoid division by zero
            self.card_popularity = {
                self.card_to_idx[name]: 1.0 - ((count - 1) / highest_sum)
                for name, count in card_counts.items()
                if name in self.card_to_idx
            }
            print(f"Calculated popularity for {len(self.card_popularity)} unique card indices.")

    def __len__(self):
        return len(self.decks_embeds) # Length based on main decks

    def __getitem__(self, idx):
        # Return main deck and sideboard data
        return (
            self.decks_embeds[idx],
            self.decks_indices[idx],
            self.sideboard_embeds[idx],
            self.sideboard_indices[idx]
        )

def sinusoidal_embedding(t: torch.Tensor, dim: int = EMB_DIM):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None] * freqs[None]
    emb = torch.cat((args.sin(), args.cos()), dim=-1)
    if dim % 2:
        emb = nn.functional.pad(emb, (0, 1))
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

    def forward(self, x_t, x0, sb_x_t, t, mask, sb_mask):
        # Calculate base sinusoidal embedding once
        sin_emb = sinusoidal_embedding(t, EMB_DIM)

        # --- Path 1: Main Deck Noise Prediction ---
        main_t_emb_flat = self.main_time_mlp(sin_emb)
        main_t_emb = main_t_emb_flat[:, None, :].expand(-1, DECK_SIZE, -1)
        main_mask_emb = self.main_mask_mlp(mask)
        h_main = x_t + main_t_emb + main_mask_emb
        h_main_proj = self.main_input_proj(h_main)
        main_encoded = self.main_transformer_encoder(h_main_proj)
        main_noise_pred = self.main_output_proj(main_encoded)

        # --- Path 2: Sideboard Context Generation ---
        h_sb_context = x0
        h_sb_context_proj = self.sb_context_input_proj(h_sb_context)
        sb_context_encoded = self.sideboard_context_encoder(h_sb_context_proj)

        # --- Path 3: Sideboard Noise Prediction ---
        sb_decoder_t_emb_flat = self.sb_time_mlp(sin_emb)
        sb_decoder_t_emb = sb_decoder_t_emb_flat[:, None, :].expand(-1, SIDEBOARD_SIZE, -1)
        sb_decoder_mask_emb = self.sb_mask_mlp(sb_mask)
        h_sb = sb_x_t + sb_decoder_t_emb + sb_decoder_mask_emb
        h_sb_proj = self.sb_input_proj(h_sb)
        sb_decoded = self.sb_transformer_decoder(tgt=h_sb_proj, memory=sb_context_encoded)
        sb_decoded = self.sb_transformer_output(sb_decoded)
        sb_noise_pred = self.sb_output_proj(sb_decoded)

        return main_noise_pred, sb_noise_pred

def cosine_beta_schedule(T, s=0.008):
    steps = torch.linspace(0, T, T + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clip(betas, 0, 0.999).float()

class DiffusionTrainer:
    def __init__(self, model, device, lr, weight_decay, total_epochs, card_popularity, T=TIMESTEPS, masks_per_deck=1):
        self.model = model.to(device)
        self.device = device
        self.T = T
        self.masks_per_deck = masks_per_deck
        self.total_epochs = total_epochs
        self.card_popularity = card_popularity # Now maps card_idx -> popularity score

        beta = cosine_beta_schedule(T).to(device)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register("beta", beta)
        self.register("alpha", alpha)
        self.register("alpha_bar", alpha_bar)
        self.register("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def register(self, name, tensor):
        setattr(self, name, tensor)

    @torch.no_grad()
    def _extract(self, a, t, shape, target_device):
        batch_size = t.shape[0]
        out = a.gather(0, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(target_device)

    def q_sample(self, x0, t, mask):
        mask_expanded = mask.expand_as(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bar, t, x0.shape, x0.device)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bar, t, x0.shape, x0.device)
        noise = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        return mask_expanded * x0 + (1 - mask_expanded) * x_t, noise

    def _create_mask_row(self, k_target, deck_size, current_deck_indices, device):
        """Creates a single mask row [deck_size, 1] for a given k_target,
           tracking available indices across passes."""
        mask_row = torch.zeros(deck_size, 1, device=device)
        k_target = min(k_target, deck_size) # Allow masking up to the full deck size
        if k_target <= 0: # Handle edge case
            return mask_row

        available_mask = torch.ones(deck_size, dtype=torch.bool, device=device)

        masked_indices_list = [] # Use list to collect tensors
        current_masked_count = 0
        max_iterations = 10 # Safeguard
        current_iteration = 0

        while current_masked_count < k_target and current_iteration < max_iterations and available_mask.any():
            # --- Get unique cards available in *this specific pass* ---
            # This needs to be done inside the loop as available_mask changes
            current_available_indices = available_mask.nonzero(as_tuple=True)[0]
            if len(current_available_indices) == 0: break # Should not happen based on while condition, but safety

            unique_cards_in_pass, pass_inverse = torch.unique(current_deck_indices[current_available_indices], return_inverse=True)
            if len(unique_cards_in_pass) == 0: break # No cards left to potentially mask

            # --- Calculate weights for unique cards in this pass ---
            weights = torch.tensor([
                # Use the id->score map directly
                0.5 + self.card_popularity.get(card_idx.item(), 1.0)
                for card_idx in unique_cards_in_pass
            ], device=device, dtype=torch.float)

            num_to_sample = len(unique_cards_in_pass)
            # Use multinomial sampling *without replacement* to get a weighted permutation
            perm_indices = torch.multinomial(weights, num_samples=num_to_sample, replacement=False)
            # perm_indices now holds indices into unique_cards_in_pass, ordered by weighted sample
            weighted_shuffled_unique_cards = unique_cards_in_pass[perm_indices]

            for card_idx in weighted_shuffled_unique_cards: # Use the new weighted order
                if current_masked_count >= k_target:
                    break

                # Find positions of this card ONLY among AVAILABLE indices
                potential_positions = (current_deck_indices == card_idx).nonzero(as_tuple=True)[0]
                available_positions_mask = available_mask[potential_positions]
                available_positions = potential_positions[available_positions_mask]
                available_count = len(available_positions)

                if available_count == 0:
                    continue # No available copies of this card left

                needed = k_target - current_masked_count

                if random.random() < 0.85: # Path 1: Mask all AVAILABLE copies
                    num_to_mask_this_card = min(available_count, needed)
                    # Select the required number from available positions
                    perm_select = torch.randperm(available_count, device=device)[:num_to_mask_this_card]
                    selected_positions = available_positions[perm_select]

                else: # Path 2: Mask a random number of AVAILABLE copies
                    max_can_mask = min(available_count, needed)
                    if max_can_mask <= 0: # Should already be handled by needed check, but safety
                        continue
                    # Randomly choose how many to mask (1 to max_can_mask)
                    num_to_mask_this_card = random.randint(1, max(1, max_can_mask))
                    # Select random available positions
                    perm_select = torch.randperm(available_count, device=device)[:num_to_mask_this_card]
                    selected_positions = available_positions[perm_select]

                if len(selected_positions) > 0:
                    # Mark these positions as unavailable for future iterations/cards
                    available_mask[selected_positions] = False
                    # Add to our list of masked indices
                    masked_indices_list.append(selected_positions)
                    # Update the count
                    current_masked_count += len(selected_positions)

            current_iteration += 1

        if current_iteration >= max_iterations and current_masked_count < k_target:
            print(f"Warning: Reached max iterations ({max_iterations}) in _create_mask_row but only masked {current_masked_count}/{k_target} items. Proceeding with current mask.")

        if masked_indices_list:
            final_masked_indices = torch.cat(masked_indices_list)
            mask_row[final_masked_indices] = 1.0

        return mask_row

    def _generate_mask_and_k(self, B, N, deck_size, x0_indices_repeated, device):
        """Generates masks using partitioning logic for k values."""
        BN = B * N
        total_k_range = deck_size - 1
        if total_k_range < N:
            print(f"Warning: deck_size-1 ({total_k_range}) < masks_per_deck ({N}). Sampling k from [1, {deck_size-1}] for each item.")
            ks = torch.randint(1, deck_size, (BN,), device=device, dtype=torch.long)
        else:
            base_size = total_k_range // N
            remainder = total_k_range % N
            ks = torch.zeros(BN, dtype=torch.long, device=device)
            current_k_start = 1
            for j in range(N):
                size = base_size + (1 if j < remainder else 0)
                part_start = current_k_start
                part_end = current_k_start + size
                if part_end <= part_start:
                    print(f"Warning: Partition {j} for deck_size {deck_size} is empty (size={size}). Sampling k=1.")
                    k_sample = 1
                else:
                    k_sample = torch.randint(part_start, part_end, (B,), device=device)
                indices = torch.arange(j, BN, N, device=device)
                ks[indices] = k_sample
                current_k_start = part_end

        # Create mask tensor by calling _create_mask_row for each k
        mask = torch.zeros(BN, deck_size, 1, device=device)
        for i in range(BN):
            mask[i] = self._create_mask_row(ks[i].item(), deck_size, x0_indices_repeated[i], device)

        return mask, ks

    def p_losses(self, x0_embeddings, x0_indices, sb_x0_embeddings, sb_x0_indices):
        B = x0_embeddings.size(0)
        N = self.masks_per_deck
        BN = B * N
        device = self.device

        # Repeat inputs
        x0_embeddings_repeated = x0_embeddings.repeat_interleave(N, dim=0)
        x0_indices_repeated = x0_indices.repeat_interleave(N, dim=0)
        sb_x0_embeddings_repeated = sb_x0_embeddings.repeat_interleave(N, dim=0)
        sb_x0_indices_repeated = sb_x0_indices.repeat_interleave(N, dim=0)

        t = torch.randint(0, self.T, (BN,), device=device, dtype=torch.long)

        # --- Generate Masks ---
        mask, main_ks = self._generate_mask_and_k(B, N, DECK_SIZE, x0_indices_repeated, device)
        # Generate random k values for sideboard (1 to SIDEBOARD_SIZE - 1)
        sb_k_values = torch.randint(1, SIDEBOARD_SIZE, (BN,), device=device, dtype=torch.long)
        # Create a 50/50 mask to set some k values to 0
        sb_zero_mask = torch.randint(0, 2, (BN,), device=device, dtype=torch.long)
        # Apply mask: ~50% become 0, ~50% keep their value (1 to SIDEBOARD_SIZE - 1)
        sb_ks = sb_k_values * sb_zero_mask

        sb_mask = torch.zeros(BN, SIDEBOARD_SIZE, 1, device=device)
        for i in range(BN):
            sb_mask[i] = self._create_mask_row(sb_ks[i].item(), SIDEBOARD_SIZE, sb_x0_indices_repeated[i], device)

        # --- Diffusion Process (q_sample) ---
        x_t, noise = self.q_sample(x0_embeddings_repeated, t, mask)
        sb_x_t, sb_noise = self.q_sample(sb_x0_embeddings_repeated, t, sb_mask)

        # --- Get Model Predictions ---
        # Pass x0_embeddings_repeated as the x0 argument to the model
        main_noise_pred, sb_noise_pred = self.model(x_t, x0_embeddings_repeated, sb_x_t, t, mask, sb_mask)

        # --- Calculate Losses ---
        main_loss = ((noise - main_noise_pred) * (1 - mask.expand_as(noise))).pow(2).mean()
        sb_loss = ((sb_noise - sb_noise_pred) * (1 - sb_mask.expand_as(sb_noise))).pow(2).mean()

        total_loss = main_loss + sb_loss

        return total_loss, main_loss, sb_loss

    def train(self, train_loader, epochs, start_epoch, save_path):
        self.model.train()

        for epoch in range(epochs):
            epoch += start_epoch
            pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            train_loss_accum = 0.0
            main_loss_accum = 0.0
            sb_loss_accum = 0.0

            # Unpack main deck and sideboard data
            for batch_embeds, batch_indices, sideboard_embeds, sideboard_indices in pbar:
                batch_embeds = batch_embeds.to(self.device)
                batch_indices = batch_indices.to(self.device)
                sideboard_embeds = sideboard_embeds.to(self.device)
                sideboard_indices = sideboard_indices.to(self.device)

                # --- Diffusion Model Update ---
                self.opt.zero_grad()

                # Calculate diffusion loss, now returns total, main, and sideboard loss
                total_loss, main_loss, sb_loss = self.p_losses(
                    batch_embeds, batch_indices, sideboard_embeds, sideboard_indices
                )

                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                # --- Accumulate losses --- 
                train_loss_accum += total_loss.item()
                main_loss_accum += main_loss.item()
                sb_loss_accum += sb_loss.item()

                pbar.set_postfix({
                    "Total Loss": total_loss.item(),
                    "Main Loss": main_loss.item(),
                    "SB Loss": sb_loss.item(),
                    "GradNorm": grad_norm.item()
                })

            avg_train_loss = train_loss_accum / len(train_loader)
            avg_main_loss = main_loss_accum / len(train_loader)
            avg_sb_loss = sb_loss_accum / len(train_loader)
            print(f"Epoch {epoch+1}: Avg Loss: {avg_train_loss:.4f} (Main: {avg_main_loss:.4f}, SB: {avg_sb_loss:.4f})")

            save_dict = {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "config": self.model.cfg
            }
            torch.save(save_dict, save_path)

def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Model")
    parser.add_argument("--deck_dir", default="./data/new_decks")
    parser.add_argument("--embeddings", default="data/card_embeddings.pkl")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", default="models/diffusion_model.pth")
    parser.add_argument("--masks_per_deck", type=int, default=5, help="Number of different masks (k values) to generate per deck per epoch")
    parser.add_argument("--classifier_path", default="models/card_classifier.pt", help="Path to the trained card classifier for mappings")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers for Main Deck Transformer Encoder")
    parser.add_argument("--sb_num_layers", type=int, default=8, help="Number of layers for Sideboard Decoder")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads for Transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=3072, help="Feedforward dimension for Transformer layers")
    parser.add_argument("--model_dim", type=int, default=384, help="Internal projection dimension for the diffusion model")
    parser.add_argument("--diff_weight_decay", type=float, default=1e-3, help="Weight decay for the diffusion model optimizer")

    args = parser.parse_args()

    print(f"Using device: {args.device}")
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("Loading dataset...")
    dataset = DeckDataset(args.deck_dir, args.embeddings, args.classifier_path)
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check deck directory and preprocessing.")
        return
    print(f"Loaded {len(dataset)} decks.")

    train_dataset = dataset
    print(f"Using full dataset ({len(train_dataset)} samples) for training.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True)

    model_cfg = {
        "layers": args.num_layers,
        "sb_layers": args.sb_num_layers,
        "heads": args.num_heads,
        "dim_feedforward": args.dim_feedforward,
        "model_dim": args.model_dim
    }
    model = DiffusionModel(model_cfg)
    print(f"Diffusion Model - Main Layers: {args.num_layers}, SB Layers: {args.sb_num_layers}, Heads: {args.num_heads}, FF Dim: {args.dim_feedforward}, Model Dim: {args.model_dim}")

    start_epoch = 0

    # Initialize trainer first, then potentially load state
    trainer = DiffusionTrainer(
        model, args.device, args.lr,
        args.diff_weight_decay,
        args.epochs,
        dataset.card_popularity,
        T=TIMESTEPS, masks_per_deck=args.masks_per_deck
    )

    if os.path.exists(args.save):
        print(f"Loading existing checkpoint from {args.save}")
        try:
            # Load to the same device the model is on
            checkpoint = torch.load(args.save, map_location=args.device)

            # Load model state - use strict=False initially if keys might mismatch due to refactoring
            model.load_state_dict(checkpoint['model'], strict=True)
            print("Model state loaded.")

            # Resume from the next epoch
            start_epoch = checkpoint.get('epoch', -1) + 1
            del checkpoint

        except Exception as e:
            print(f"Error loading checkpoint: {e}. Training from scratch.")
            # Reset start epoch if loading failed
            start_epoch = 0
        
        trainer.model.to(args.device)
    else:
        print(f"No existing checkpoint found at {args.save}, starting training from scratch.")
        start_epoch = 0
        # Ensure model is on the correct device
        trainer.model.to(args.device)

    print(f"Starting training from epoch {start_epoch} for {args.epochs} total epochs...")
    if args.epochs - start_epoch > 0:
        # Pass the number of remaining epochs
        trainer.train(train_loader, args.epochs, start_epoch, args.save)
    else:
        print(f"Training already completed or start_epoch ({start_epoch}) >= total epochs ({args.epochs}).")

if __name__ == "__main__":
    main()