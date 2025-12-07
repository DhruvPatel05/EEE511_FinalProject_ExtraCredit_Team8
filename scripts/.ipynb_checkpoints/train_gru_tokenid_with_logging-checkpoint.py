# train_gru_tokenid_with_logging.py
"""
GRU training on token-id pairs with BOS/EOS, with CSV logging.

Usage (if this file is in the same folder as models.py & dataset.py):
    python train_gru_tokenid_with_logging.py

If you keep a `scripts/` package and this file is inside it:
    python -m scripts.train_gru_tokenid_with_logging
"""
import os
import csv
import json
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# If your project uses a "scripts" package, keep these imports:
from dataset import NeuralTextDataset, collate_fn
from models import Seq2SeqGRU
# If everything is in one flat folder, change to:
# from dataset import NeuralTextDataset, collate_fn
# from models import Seq2SeqGRU


# -------- CONFIG --------
PAIRS_PKL = "day_pairs_tokenids_bos_small.pkl"   # same data as before
TOKEN_MAP = "tokenid_map.json"
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-4
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "gru_tokenid_checkpoint_logged.pth"
LOG_CSV = "logs/gru_tokenid_train.csv"
# ------------------------


def main():
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True) if os.path.dirname(LOG_CSV) else None

    print("Device:", DEVICE)
    print("Loading pairs from:", PAIRS_PKL)
    pairs = pickle.load(open(PAIRS_PKL, "rb"))
    print("Total pairs:", len(pairs))

    # We only use token IDs (third element in tuple is mapped string or None)
    pairs = [(n, t, None) for (n, t, _) in pairs]

    # Load token-id meta (PAD/BOS/EOS, vocab size)
    tm = json.load(open(TOKEN_MAP))
    PAD_ID = int(tm["PAD"])
    BOS_ID = int(tm["BOS"])
    EOS_ID = int(tm["EOS"])
    vocab_size = int(tm["vocab_size"])
    print("PAD,BOS,EOS:", PAD_ID, BOS_ID, EOS_ID, "vocab_size:", vocab_size)

    # Dataset + loader
    dataset = NeuralTextDataset(pairs, char2idx=None)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Get input dimension from one batch
    X0, x_lens0, Y0, y_lens0 = next(iter(loader))
    in_dim = X0.shape[2]
    print("Input feature dim:", in_dim)

    model = Seq2SeqGRU(
        in_dim=in_dim,
        enc_hidden=HIDDEN,
        dec_hidden=HIDDEN,
        vocab_size=vocab_size,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # CSV logging
    csv_file = open(LOG_CSV, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss"])
    print("Logging training loss to:", LOG_CSV)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for X, x_lens, Y, y_lens in loader:
            X = X.to(DEVICE)                # (B, T, C)
            Y = Y.to(torch.long)            # (B, L), contains token IDs or -100

            B, L = Y.size()

            # Build decoder input y_in:
            #  - BOS at position 0
            #  - previous tokens shifted right, PAD where Y == -100
            y_in = torch.full_like(Y, fill_value=PAD_ID, device=DEVICE)
            y_in[:, 0] = BOS_ID
            if L > 1:
                prev = Y[:, :-1].clone()
                prev = torch.where(prev == -100,
                                   torch.full_like(prev, fill_value=PAD_ID),
                                   prev)
                y_in[:, 1:] = prev.to(DEVICE)

            y_tgt = Y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X, y_in)          # (B, L, vocab_size)
            loss = criterion(
                logits.view(-1, vocab_size),
                y_tgt.view(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        print(f"[GRU] Epoch {epoch}/{EPOCHS} - train_loss: {avg_loss:.4f}")
        writer.writerow([epoch, avg_loss])
        csv_file.flush()

        # Save checkpoint every epoch (you can keep only last if you want)
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    csv_file.close()
    print("Training complete. Checkpoint saved to:", CHECKPOINT_PATH)
    print("Training log saved to:", LOG_CSV)


if __name__ == "__main__":
    main()
