# scripts/train_gru.py
import pickle, torch
from torch.utils.data import DataLoader
import torch.nn as nn
from scripts.dataset import NeuralTextDataset, build_vocab_from_pairs, collate_fn
from scripts.models import Seq2SeqGRU

# CONFIG
PAIRS_PKL = "day_pairs_small.pkl"   # small for quick runs
BATCH_SIZE = 8
EPOCHS = 6
LR = 1e-3
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "gru_checkpoint.pth"

def main():
    pairs = pickle.load(open(PAIRS_PKL, "rb"))

    # Build char vocab if mapped strings present
    tokens, char2idx, idx2char = build_vocab_from_pairs(pairs)
    use_char_vocab = (char2idx is not None)
    if use_char_vocab:
        print("Using character vocab of size", len(char2idx))
        # FILTER: keep only pairs that have a mapped string
        pairs = [p for p in pairs if p[2] is not None]
        if len(pairs) == 0:
            raise RuntimeError("No pairs with mapped strings found.")
        print("After filtering, examples with strings:", len(pairs))
    else:
        print("No mapped strings found; using token ids directly.")

    ds = NeuralTextDataset(pairs, char2idx=char2idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # vocab size
    if use_char_vocab:
        vocab_size = len(char2idx)
    else:
        sample = next(iter(loader))
        Y_sample = sample[2]
        vocab_size = int(torch.max(Y_sample[Y_sample != -100]).item() + 1)

    print("Using vocab_size =", vocab_size)
    in_dim = next(iter(loader))[0].shape[2]
    model = Seq2SeqGRU(in_dim, enc_hidden=HIDDEN, dec_hidden=HIDDEN, vocab_size=vocab_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0
        for X, x_lens, Y, y_lens in loader:
            X = X.to(DEVICE)
            if Y.size(1) < 2:
                continue
            y_in = Y[:, :-1].to(DEVICE)
            y_tgt = Y[:, 1:].to(DEVICE)
            logits = model(X, y_in)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = y_tgt.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{EPOCHS} avg loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), SAVE_PATH)
    print("Training finished. Saved:", SAVE_PATH)

if __name__ == "__main__":
    main()
