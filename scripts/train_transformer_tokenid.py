# scripts/train_transformer_tokenid.py
"""
Transformer training on token-id pairs with BOS/EOS.
Run: python -m scripts.train_transformer_tokenid
"""
import json, pickle, torch, math
import torch.nn as nn
from torch.utils.data import DataLoader
from scripts.dataset import NeuralTextDataset, collate_fn
from scripts.models import TransformerSeq2Seq, make_tgt_mask

# CONFIG
PAIRS_PKL = "day_pairs_tokenids_bos_small.pkl"
TOKEN_MAP = "tokenid_map.json"
BATCH_SIZE = 8
EPOCHS = 12
LR = 1e-4
D_MODEL = 128
NHEAD = 4
ENC_LAYERS = 2
DEC_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "transformer_tokenid_checkpoint.pth"
PRINT_SAMPLE = True

def load_token_map(map_path):
    return json.load(open(map_path))

def infer_vocab_size_from_map(map_path):
    m = json.load(open(map_path))
    return int(m["vocab_size"])

def main():
    print("Loading pairs:", PAIRS_PKL)
    pairs = pickle.load(open(PAIRS_PKL, "rb"))
    print("Pairs loaded:", len(pairs))
    pairs = [(n,t,None) for (n,t,s) in pairs]

    tm = load_token_map(TOKEN_MAP)
    PAD_ID = int(tm["PAD"]); BOS_ID = int(tm["BOS"]); EOS_ID = int(tm["EOS"])
    vocab_size = infer_vocab_size_from_map(TOKEN_MAP)
    print("PAD,BOS,EOS:", PAD_ID, BOS_ID, EOS_ID, "vocab_size:", vocab_size)

    ds = NeuralTextDataset(pairs, char2idx=None)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    sample = next(iter(loader))
    in_dim = sample[0].shape[2]
    model = TransformerSeq2Seq(in_dim=in_dim, d_model=D_MODEL, nhead=NHEAD, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, vocab_size=vocab_size).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def make_local_tgt_mask(sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask.to(DEVICE)

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        steps = 0
        for X, x_lens, Y, y_lens in loader:
            X = X.to(DEVICE)
            # Build y_in as before (insert BOS and shift)
            B, L = Y.size()
            Y = Y.to(torch.long)
            y_in = torch.full_like(Y, fill_value=PAD_ID).to(DEVICE)
            y_in[:,0] = BOS_ID
            if L > 1:
                prev = Y[:, :-1].clone()
                prev = torch.where(prev == -100, torch.full_like(prev, fill_value=PAD_ID), prev)
                y_in[:, 1:] = prev.to(DEVICE)
            y_tgt = Y.to(DEVICE)

            tgt_mask = make_local_tgt_mask(y_in.size(1))
            src_padding_mask = (torch.arange(X.size(1))[None, :].to(DEVICE) >= x_lens[:, None].to(DEVICE))
            tgt_padding_mask = (y_in == -100)  # but y_in uses PAD_ID for empty -> no -100 expected
            logits = model(X, y_in, src_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = y_tgt.reshape(-1)
            loss = criterion(logits_flat, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg = total_loss / max(1, steps)
        print(f"[Transformer] Epoch {epoch}/{EPOCHS} avg_loss: {avg:.4f}")
        torch.save(model.state_dict(), SAVE_PATH)

        if PRINT_SAMPLE:
            model.eval()
            with torch.no_grad():
                Xs, x_lens_s, Ys, y_lens_s = next(iter(loader))
                # build y_in sample
                y_in_s = torch.full_like(Ys, fill_value=PAD_ID).to(DEVICE)
                y_in_s[:,0] = BOS_ID
                if Ys.size(1) > 1:
                    prev = Ys[:, :-1].clone()
                    prev = torch.where(prev == -100, torch.full_like(prev, fill_value=PAD_ID), prev)
                    y_in_s[:,1:] = prev.to(DEVICE)
                logits = model(Xs.to(DEVICE)[:1], y_in_s[:1], src_key_padding_mask=(torch.arange(Xs.size(1))[None,:].to(DEVICE) >= x_lens_s[:1, None].to(DEVICE)), tgt_mask=make_local_tgt_mask(y_in_s.size(1)), tgt_key_padding_mask=(y_in_s[:1] == -100))
                pred_ids = logits.argmax(dim=-1).cpu().numpy()[0]
                true_ids = Ys[0].cpu().numpy()
                print(" sample true (first 50):", true_ids[:50])
                print(" sample pred (first 50):", pred_ids[:50])

    print("Training complete. Checkpoint saved:", SAVE_PATH)

if __name__ == "__main__":
    main()
