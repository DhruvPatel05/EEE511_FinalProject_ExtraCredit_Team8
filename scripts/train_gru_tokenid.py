# scripts/train_gru_tokenid.py
"""
GRU training on token-id pairs with BOS/EOS.
Run: python -m scripts.train_gru_tokenid
"""
import json, pickle, torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.dataset import NeuralTextDataset, collate_fn
from scripts.models import Seq2SeqGRU

# CONFIG
PAIRS_PKL = "day_pairs_tokenids_bos_small.pkl"   # small set for dev; change to full file if desired
TOKEN_MAP = "tokenid_map.json"
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-3
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "gru_tokenid_checkpoint.pth"
PRINT_SAMPLE = True

def infer_vocab_size_from_map(map_path):
    m = json.load(open(map_path))
    return int(m["vocab_size"])

def load_token_map(map_path):
    return json.load(open(map_path))

def main():
    print("Loading pairs:", PAIRS_PKL)
    pairs = pickle.load(open(PAIRS_PKL, "rb"))
    print("Pairs loaded:", len(pairs))
    # ensure format (neural, tokenids_with_eos, None)
    pairs = [(n, t, None) for (n,t,s) in pairs]

    tm = load_token_map(TOKEN_MAP)
    PAD_ID = int(tm["PAD"])
    BOS_ID = int(tm["BOS"])
    EOS_ID = int(tm["EOS"])
    vocab_size = infer_vocab_size_from_map(TOKEN_MAP)
    print("PAD,BOS,EOS:", PAD_ID, BOS_ID, EOS_ID, "vocab_size:", vocab_size)

    ds = NeuralTextDataset(pairs, char2idx=None)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    sample = next(iter(loader))
    in_dim = sample[0].shape[2]
    model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=HIDDEN, dec_hidden=HIDDEN, vocab_size=vocab_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        steps = 0
        for X, x_lens, Y, y_lens in loader:
            X = X.to(DEVICE)
            # Build decoder input y_in (insert BOS and shift right), keep y_tgt with -100 padding
            B, L = Y.size()
            Y = Y.to(torch.long)
            # y_in default PAD_ID then set BOS at pos 0 and previous tokens in subsequent positions
            y_in = torch.full_like(Y, fill_value=PAD_ID).to(DEVICE)
            y_in[:,0] = BOS_ID
            if L > 1:
                prev = Y[:, :-1].clone()
                prev = torch.where(prev == -100, torch.full_like(prev, fill_value=PAD_ID), prev)
                y_in[:, 1:] = prev.to(DEVICE)
            y_tgt = Y.to(DEVICE)  # contains EOS and -100 for padded positions

            logits = model(X.to(DEVICE), y_in)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = y_tgt.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"[GRU] Epoch {epoch}/{EPOCHS} avg_loss: {avg:.4f}")
        torch.save(model.state_dict(), SAVE_PATH)

        if PRINT_SAMPLE:
            model.eval()
            with torch.no_grad():
                Xs, x_lens_s, Ys, y_lens_s = next(iter(loader))
                # prepare y_in from batch[0] to inspect teacher-forced logits
                B0 = Ys.size(0)
                y_in_sample = torch.full_like(Ys, fill_value=PAD_ID).to(DEVICE)
                y_in_sample[:,0] = BOS_ID
                if Ys.size(1) > 1:
                    prev = Ys[:, :-1].clone()
                    prev = torch.where(prev == -100, torch.full_like(prev, fill_value=PAD_ID), prev)
                    y_in_sample[:, 1:] = prev.to(DEVICE)
                logits = model(Xs.to(DEVICE)[:1], y_in_sample[:1])
                pred_ids = logits.argmax(dim=-1).cpu().numpy()[0]
                true_ids = Ys[0].cpu().numpy()
                print(" sample true (first 40):", true_ids[:40])
                print(" sample pred (first 40):", pred_ids[:40])

    print("Training complete. Checkpoint saved:", SAVE_PATH)

if __name__ == "__main__":
    main()
