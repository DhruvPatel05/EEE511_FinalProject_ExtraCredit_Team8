# compare_gru_vs_transformer_figures.py
"""
Compare GRU vs Transformer on the same BOS/EOS token-id dataset
and generate figures showing why the Transformer is better.

Figures saved:
  - gru_vs_transformer_wer_bar.png
  - gru_vs_transformer_wer_scatter.png
  - gru_vs_transformer_delta_hist.png
"""

import argparse
import json
import math
import pickle

from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from jiwer import wer

from scripts.models import Seq2SeqGRU, TransformerSeq2Seq, make_tgt_mask  # adjust import if needed


# ---------- Decoders ----------

def greedy_decode_transformer(model, src_tensor, src_len, bos_id, eos_id, max_len, device):
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device).unsqueeze(0)  # (1, T, C)
        T = src.size(1)
        idxs = torch.arange(T, device=device)[None, :]
        src_padding_mask = idxs >= src_len  # (1, T)

        src_proj = model.input_proj(src) * math.sqrt(model.d_model)
        memory = model.encoder(model.pos(src_proj), src_key_padding_mask=src_padding_mask)

        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # (1, 1)
        out_ids = []

        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys.size(1)).to(device)
            dec_in = model.pos(model.embedding(ys) * math.sqrt(model.d_model))
            out = model.decoder(
                dec_in,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding_mask,
            )
            logits = model.out(out[:, -1, :])  # (1, V)
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == eos_id:
                break
            out_ids.append(next_id)
            ys = torch.cat(
                [ys, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )

        return out_ids


def greedy_decode_gru(model, src_tensor, bos_id, eos_id, max_len, device):
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device).unsqueeze(0)  # (1, T, C)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)  # (1, H)
        hidden = torch.tanh(model.bridge(context)).unsqueeze(0)  # (1, 1, H)

        cur_id = bos_id
        out_ids = []

        for _ in range(max_len):
            cur = torch.tensor([[cur_id]], dtype=torch.long, device=device)  # (1,1)
            logits, hidden = model.decoder(cur, hidden)  # logits: (1,1,V)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if next_id == eos_id:
                break
            out_ids.append(next_id)
            cur_id = next_id

        return out_ids


# ---------- Helper ----------

def idseq_to_str_tokenwise(ids):
    """Represent a list of token IDs as a space-separated string (for jiwer)."""
    return " ".join(str(t) for t in ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str,
                        default="day_pairs_tokenids_bos_small.pkl",
                        help="Token-id pairs with BOS/EOS (same used for tokenid training).")
    parser.add_argument("--token_map", type=str, default="tokenid_map.json")
    parser.add_argument("--gru_checkpoint", type=str,
                        default="gru_tokenid_checkpoint.pth")
    parser.add_argument("--trf_checkpoint", type=str,
                        default="transformer_tokenid_checkpoint.pth")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Max number of examples to evaluate.")
    parser.add_argument("--max_len", type=int, default=200,
                        help="Max decode length.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Using device:", device)

    # --- Load data & mapping ---
    print("Loading pairs from:", args.pairs)
    pairs = pickle.load(open(args.pairs, "rb"))
    print("Total pairs:", len(pairs))

    mapping = json.load(open(args.token_map))
    PAD = int(mapping["PAD"])
    BOS = int(mapping["BOS"])
    EOS = int(mapping["EOS"])
    vocab_size = int(mapping["vocab_size"])
    print("PAD,BOS,EOS:", PAD, BOS, EOS, "vocab_size:", vocab_size)

    # figure out input dimension from one example
    sample_neural = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0)
    in_dim = sample_neural.shape[2]
    print("Input neural feature dim:", in_dim)

    # --- Build models (make sure hyperparameters match your training!) ---
    gru = Seq2SeqGRU(
        in_dim=in_dim,
        enc_hidden=128,
        dec_hidden=128,
        vocab_size=vocab_size,
    ).to(device)

    trf = TransformerSeq2Seq(
        in_dim=in_dim,
        d_model=128,
        nhead=4,
        enc_layers=2,
        dec_layers=2,
        vocab_size=vocab_size,
    ).to(device)

    print("Loading GRU checkpoint:", args.gru_checkpoint)
    gru.load_state_dict(torch.load(args.gru_checkpoint, map_location=device))
    print("Loading Transformer checkpoint:", args.trf_checkpoint)
    trf.load_state_dict(torch.load(args.trf_checkpoint, map_location=device))

    gru.eval()
    trf.eval()

    # --- Evaluate per example ---
    n_eval = min(args.n_eval, len(pairs))
    gru_wers = []
    trf_wers = []

    for i, (neural, token_ids, _) in enumerate(pairs[:n_eval]):
        src = torch.tensor(neural, dtype=torch.float32)
        src_len = torch.tensor([src.size(0)], dtype=torch.long, device=device)

        # true tokens: remove PAD
        true_ids = [int(t) for t in token_ids if int(t) != PAD]
        true_str = idseq_to_str_tokenwise(true_ids)

        # GRU prediction
        gru_pred_ids = greedy_decode_gru(gru, src, BOS, EOS, args.max_len, device)
        gru_str = idseq_to_str_tokenwise(gru_pred_ids)

        # Transformer prediction
        trf_pred_ids = greedy_decode_transformer(trf, src, src_len, BOS, EOS, args.max_len, device)
        trf_str = idseq_to_str_tokenwise(trf_pred_ids)

        # token-level WER for this example
        gru_w = wer(true_str, gru_str)
        trf_w = wer(true_str, trf_str)
        gru_wers.append(gru_w)
        trf_wers.append(trf_w)

        if i < 8:  # print a few qualitative examples
            print("\n==== Example", i, "====")
            print("TRUE ids      :", true_ids[:60])
            print("GRU  pred ids :", gru_pred_ids[:60])
            print("TRF  pred ids :", trf_pred_ids[:60])
            print(f"WER  GRU = {gru_w:.3f} | TRF = {trf_w:.3f}")

    # --- Aggregate stats ---
    import numpy as np
    gru_wers = np.array(gru_wers)
    trf_wers = np.array(trf_wers)

    mean_gru = float(gru_wers.mean())
    mean_trf = float(trf_wers.mean())
    print("\n====================")
    print(f"Mean token-level WER (GRU)        : {mean_gru:.3f}")
    print(f"Mean token-level WER (Transformer): {mean_trf:.3f}")
    print("====================")

    # --- Figure 1: Bar chart of overall WER ---
    plt.figure()
    models = ["GRU", "Transformer"]
    means = [mean_gru, mean_trf]
    plt.bar(models, means)
    plt.ylabel("Token-level WER (lower is better)")
    plt.title("Overall WER: GRU vs Transformer")
    for i, v in enumerate(means):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig("gru_vs_transformer_wer_bar.png", dpi=200)
    print("Saved gru_vs_transformer_wer_bar.png")

    # --- Figure 2: Scatter (per-sentence WER) ---
    plt.figure()
    plt.scatter(gru_wers, trf_wers, alpha=0.6)
    max_val = float(max(gru_wers.max(), trf_wers.max()))
    plt.plot([0, max_val], [0, max_val], linestyle="--")  # y = x line
    plt.xlabel("GRU sentence WER")
    plt.ylabel("Transformer sentence WER")
    plt.title("Sentence-level WER: each point = one example\n(points below diagonal = Transformer better)")
    plt.tight_layout()
    plt.savefig("gru_vs_transformer_wer_scatter.png", dpi=200)
    print("Saved gru_vs_transformer_wer_scatter.png")

    # --- Figure 3: Histogram of WER improvement ---
    delta = gru_wers - trf_wers  # >0 means Transformer better
    plt.figure()
    plt.hist(delta, bins=30)
    plt.xlabel("WER(GRU) - WER(Transformer)")
    plt.ylabel("# of sentences")
    plt.title("Distribution of WER improvement (positive = Transformer better)")
    plt.tight_layout()
    plt.savefig("gru_vs_transformer_delta_hist.png", dpi=200)
    print("Saved gru_vs_transformer_delta_hist.png")


if __name__ == "__main__":
    main()
