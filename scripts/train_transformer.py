# scripts/train_transformer.py
import pickle, torch, math
from torch.utils.data import DataLoader
import torch.nn as nn
from scripts.dataset import NeuralTextDataset, build_vocab_from_pairs, collate_fn
from scripts.models import TransformerSeq2Seq, make_tgt_mask

# CONFIG
PAIRS_PKL = "day_pairs_small.pkl"
BATCH_SIZE = 8
EPOCHS = 6
LR = 1e-4
D_MODEL = 128
NHEAD = 4
ENC_LAYERS = 2
DEC_LAYERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "transformer_checkpoint.pth"

def main():
    pairs = pickle.load(open(PAIRS_PKL, "rb"))

    # Use only mapped strings for consistent char vocab
    tokens, char2idx, idx2char = build_vocab_from_pairs(pairs)
    if char2idx is None:
        raise RuntimeError("No mapped strings found in pairs. Train transformer using character vocab requires mapped strings.")
    print("Using character vocab of size", len(char2idx))
    pairs = [p for p in pairs if p[2] is not None]
    print("Filtered pairs:", len(pairs))

    ds = NeuralTextDataset(pairs, char2idx=char2idx)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    sample = next(iter(loader))
    in_dim = sample[0].shape[2]
    vocab_size = len(char2idx)
    model = TransformerSeq2Seq(in_dim=in_dim, d_model=D_MODEL, nhead=NHEAD, enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, vocab_size=vocab_size).to(DEVICE)

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
            tgt_in = Y[:, :-1].to(DEVICE)
            tgt_out = Y[:, 1:].to(DEVICE)
            tgt_mask = make_tgt_mask(tgt_in.size(1)).to(DEVICE)
            src_padding_mask = (torch.arange(X.size(1))[None, :].to(DEVICE) >= x_lens[:, None].to(DEVICE))
            tgt_padding_mask = (tgt_in == -100)
            logits = model(X, tgt_in, src_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = tgt_out.reshape(-1)
            loss = criterion(logits_flat, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        avg_loss = total_loss / max(1, steps)
        print(f"Epoch {epoch+1}/{EPOCHS} avg loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), SAVE_PATH)
    print("Finished training. Saved:", SAVE_PATH)

if __name__ == "__main__":
    main()
