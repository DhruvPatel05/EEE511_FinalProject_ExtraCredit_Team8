# scripts/token_confusions.py
"""
Analyze token-level confusions for BOS/EOS token-id models.

Example:
  python -m scripts.token_confusions \
      --model_type transformer \
      --checkpoint transformer_tokenid_checkpoint.pth \
      --pairs day_pairs_tokenids_bos_small.pkl \
      --token_map tokenid_map.json \
      --output_csv token_confusions.csv \
      --top_k 50
"""
import argparse
import csv
import json
import pickle
from collections import Counter

import torch

from scripts.models import Seq2SeqGRU, TransformerSeq2Seq, make_tgt_mask


def greedy_decode_transformer(model, src_tensor, src_len, bos_id, eos_id, max_len, device):
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device).unsqueeze(0)  # (1, T, C)
        T = src.size(1)
        idxs = torch.arange(T, device=device)[None, :]
        src_padding_mask = idxs >= src_len

        src_proj = model.input_proj(src) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float32, device=device))
        memory = model.encoder(model.pos(src_proj), src_key_padding_mask=src_padding_mask)

        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        out_seq = []

        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys.size(1)).to(device)
            dec_in = model.pos(model.embedding(ys) * torch.sqrt(torch.tensor(model.d_model, dtype=torch.float32, device=device)))
            out = model.decoder(dec_in, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            logits = model.out(out[:, -1, :])
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == eos_id:
                break
            out_seq.append(next_id)
            ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)

        return out_seq


def greedy_decode_gru(model, src_tensor, bos_id, eos_id, max_len, device):
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device).unsqueeze(0)  # (1, T, C)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)
        hidden = torch.tanh(model.bridge(context)).unsqueeze(0)

        cur_id = bos_id
        seq = []

        for _ in range(max_len):
            cur = torch.tensor([[cur_id]], dtype=torch.long, device=device)
            logits, hidden = model.decoder(cur, hidden)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            if next_id == eos_id:
                break
            seq.append(next_id)
            cur_id = next_id

        return seq


def id_to_label(token_id, inv_map):
    if inv_map is None:
        return str(token_id)
    return str(inv_map.get(token_id, token_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default="day_pairs_tokenids_bos_small.pkl")
    parser.add_argument("--token_map", type=str, default="tokenid_map.json")
    parser.add_argument("--model_type", type=str, choices=["gru", "transformer"], default="transformer")
    parser.add_argument("--checkpoint", type=str, default="transformer_tokenid_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--ignore_eos", action="store_true",
                        help="Ignore positions where true token is EOS.")
    parser.add_argument("--output_csv", type=str, default="token_confusions.csv")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading pairs:", args.pairs)
    pairs = pickle.load(open(args.pairs, "rb"))
    print("Pairs loaded:", len(pairs))

    mapping = json.load(open(args.token_map))
    PAD = int(mapping["PAD"])
    BOS = int(mapping["BOS"])
    EOS = int(mapping["EOS"])
    vocab_size = int(mapping["vocab_size"])
    old2new = mapping.get("old2new", None)
    inv_map = None
    if old2new is not None:
        inv_map = {v: k for k, v in old2new.items()}

    print("PAD,BOS,EOS:", PAD, BOS, EOS, "vocab_size:", vocab_size)

    sample = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0)
    in_dim = sample.shape[2]

    if args.model_type == "gru":
        model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size).to(device)
    else:
        model = TransformerSeq2Seq(
            in_dim=in_dim,
            d_model=128,
            nhead=4,
            enc_layers=2,
            dec_layers=2,
            vocab_size=vocab_size,
        ).to(device)

    print("Loading checkpoint:", args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    conf = Counter()
    n_eval = min(args.n_eval, len(pairs))

    for i, (neural, token_ids, _) in enumerate(pairs[:n_eval]):
        src = torch.tensor(neural, dtype=torch.float32)
        src_len = torch.tensor([src.size(0)], dtype=torch.long, device=device)

        # true sequence: drop PAD
        true_ids = [int(t) for t in token_ids if int(t) != PAD]

        if args.model_type == "gru":
            pred_ids = greedy_decode_gru(model, src, BOS, EOS, args.max_len, device)
        else:
            pred_ids = greedy_decode_transformer(model, src, src_len, BOS, EOS, args.max_len, device)

        for t, p in zip(true_ids, pred_ids):
            if t == PAD:
                continue
            if args.ignore_eos and t == EOS:
                continue
            conf[(t, p)] += 1

        if i < 5:
            print("Example", i)
            print("TRUE ids:", true_ids[:60])
            print("PRED ids:", pred_ids[:60])
            print("----")

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_id", "pred_id", "count", "true_label", "pred_label"])
        for (t, p), c in conf.most_common():
            writer.writerow([t, p, c, id_to_label(t, inv_map), id_to_label(p, inv_map)])

    print("Saved confusion CSV to", args.output_csv)

    # Print top-K confusions
    print("\nTop", args.top_k, "confusions (true -> pred, count):")
    for (t, p), c in conf.most_common(args.top_k):
        print(
            f"{id_to_label(t, inv_map)} ({t}) -> {id_to_label(p, inv_map)} ({p}): {c}"
        )


if __name__ == "__main__":
    main()
