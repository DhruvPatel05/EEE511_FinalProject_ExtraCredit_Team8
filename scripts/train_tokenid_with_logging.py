# scripts/train_tokenid_with_logging.py
"""
Generic training on token-id pairs with BOS/EOS, logging losses to CSV.

Example:
  python -m scripts.train_tokenid_with_logging \
      --model_type transformer \
      --pairs day_pairs_tokenids_bos.pkl \
      --token_map tokenid_map.json \
      --epochs 20 \
      --log_csv logs/transformer_tokenid_train.csv
"""
import argparse
import csv
import json
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.dataset import NeuralTextDataset, collate_fn
from scripts.models import Seq2SeqGRU, TransformerSeq2Seq, make_tgt_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default="day_pairs_tokenids_bos_small.pkl")
    parser.add_argument("--token_map", type=str, default="tokenid_map.json")
    parser.add_argument("--model_type", type=str, choices=["gru", "transformer"], default="transformer")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128, help="Hidden size for GRU.")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model.")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save checkpoint. Default depends on model_type.")
    parser.add_argument("--log_csv", type=str, default=None,
                        help="CSV file to log epoch, train_loss.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # -- Load pairs --
    print("Loading pairs:", args.pairs)
    pairs = pickle.load(open(args.pairs, "rb"))
    print("Pairs loaded:", len(pairs))

    # Ensure we drop mapped string (we train on token IDs only)
    pairs = [(n, t, None) for (n, t, _) in pairs]

    # -- Token map --
    tm = json.load(open(args.token_map))
    PAD_ID = int(tm["PAD"])
    BOS_ID = int(tm["BOS"])
    EOS_ID = int(tm["EOS"])
    vocab_size = int(tm["vocab_size"])
    print("PAD,BOS,EOS:", PAD_ID, BOS_ID, EOS_ID, "vocab_size:", vocab_size)

    # -- Dataset & loader --
    ds = NeuralTextDataset(pairs, char2idx=None)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    sample = next(iter(loader))
    in_dim = sample[0].shape[2]

    # -- Model --
    if args.model_type == "gru":
        model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=args.hidden, dec_hidden=args.hidden, vocab_size=vocab_size).to(device)
        default_save = "gru_tokenid_with_logging.pth"
    else:
        model = TransformerSeq2Seq(
            in_dim=in_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            enc_layers=args.enc_layers,
            dec_layers=args.dec_layers,
            vocab_size=vocab_size,
        ).to(device)
        default_save = "transformer_tokenid_with_logging.pth"

    save_path = args.save_path or default_save
    print("Checkpoint will be saved to:", save_path)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # -- CSV logging setup --
    writer = None
    csv_file = None
    if args.log_csv is not None:
        log_dir = os.path.dirname(args.log_csv)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss"])
        print("Logging to CSV:", args.log_csv)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for X, x_lens, Y, y_lens in loader:
            X = X.to(device)
            Y = Y.to(torch.long)  # contains token IDs + -100 padding

            B, L = Y.size()

            # Build y_in: BOS followed by previous tokens (PAD where Y == -100)
            y_in = torch.full_like(Y, fill_value=PAD_ID, device=device)
            y_in[:, 0] = BOS_ID
            if L > 1:
                prev = Y[:, :-1].clone()
                prev = torch.where(prev == -100, torch.full_like(prev, fill_value=PAD_ID), prev)
                y_in[:, 1:] = prev.to(device)

            y_tgt = Y.to(device)

            optimizer.zero_grad()

            if args.model_type == "gru":
                logits = model(X, y_in)  # (B, L, V)
            else:
                # Build masks
                T = X.size(1)
                idxs = torch.arange(T, device=device)[None, :]  # (1,T)
                x_lens_dev = x_lens.to(device).unsqueeze(1)
                src_padding_mask = idxs >= x_lens_dev  # (B,T)
                tgt_mask = make_tgt_mask(L).to(device)
                tgt_key_padding = (y_in == -100)

                logits = model(
                    X,
                    y_in,
                    src_key_padding_mask=src_padding_mask,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding,
                )

            loss = criterion(logits.view(-1, vocab_size), y_tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"[{args.model_type.upper()}] Epoch {epoch}/{args.epochs} avg_loss: {avg_loss:.4f}")

        # log to CSV
        if writer is not None:
            writer.writerow([epoch, avg_loss])
            csv_file.flush()

        torch.save(model.state_dict(), save_path)

    if csv_file is not None:
        csv_file.close()

    print("Training complete. Checkpoint saved:", save_path)


if __name__ == "__main__":
    main()
