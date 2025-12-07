# scripts/eval_beam_lm_rescore.py
import argparse, pickle, math, json
from collections import defaultdict

import torch
from jiwer import wer

from scripts.models import Seq2SeqGRU, TransformerSeq2Seq, make_tgt_mask

# ---------- LM STUB ----------

def lm_log_prob(token_ids):
    """
    Stub language model: log P(token_ids) under your LM.

    token_ids: list[int] WITHOUT BOS/EOS.
    Return a scalar float (log probability, e.g. <= 0).

    Right now it returns 0.0 for every sequence.
    Replace this with a call to your n-gram LM.
    """
    return 0.0


# ---------- Beam search helpers (return N-best) ----------

def _strip_bos_eos(seq, bos_id, eos_id):
    """
    Remove BOS and everything after EOS (if present).
    """
    seq = list(seq)
    if seq and seq[0] == bos_id:
        seq = seq[1:]
    if eos_id in seq:
        eos_pos = seq.index(eos_id)
        seq = seq[:eos_pos]
    return seq


def beam_search_transformer_nbest(
    model,
    src_tensor,
    src_len,
    bos_id,
    eos_id,
    beam_size=5,
    max_len=200,
    n_best=5,
    device=None,
):
    if device is None:
        device = src_tensor.device
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device).unsqueeze(0)  # (1, T, C)
        T = src.size(1)
        src_indices = torch.arange(T, device=device)[None, :]
        src_padding_mask = (src_indices >= src_len)

        src_proj = model.input_proj(src) * math.sqrt(model.d_model)
        memory = model.encoder(model.pos(src_proj), src_key_padding_mask=src_padding_mask)

        # beam: (neg_logprob, seq(list[int]), ys(tensor 1xL))
        beams = [(0.0, [bos_id], torch.tensor([[bos_id]], dtype=torch.long, device=device))]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, seq, ys in beams:
                if seq[-1] == eos_id:
                    # completed sequence
                    completed.append((score, seq))
                    continue

                tgt_mask = make_tgt_mask(ys.size(1)).to(device)
                dec_in = model.pos(model.embedding(ys) * math.sqrt(model.d_model))
                out = model.decoder(
                    dec_in,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask,
                )
                logits = model.out(out[:, -1, :])  # (1, V)
                logp = torch.log_softmax(logits, dim=-1).squeeze(0)  # (V,)

                topk = torch.topk(logp, beam_size)
                for idx, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = seq + [idx]
                    new_score = score - float(lp)  # negative logprob
                    new_ys = torch.cat(
                        [ys, torch.tensor([[idx]], dtype=torch.long, device=device)],
                        dim=1,
                    )
                    new_beams.append((new_score, new_seq, new_ys))

            if not new_beams:
                break

            new_beams.sort(key=lambda x: x[0])
            beams = new_beams[:beam_size]

        # add completed beams; also allow unfinished ones as fallbacks
        completed.extend((s, seq) for (s, seq, _) in beams if seq[-1] == eos_id)
        if not completed:  # no EOS ever
            completed.extend((s, seq) for (s, seq, _) in beams)

        # strip BOS/EOS and sort
        cleaned = []
        for s, seq in completed:
            cleaned.append((s, _strip_bos_eos(seq, bos_id, eos_id)))
        cleaned.sort(key=lambda x: x[0])
        return cleaned[:n_best]


def beam_search_gru_nbest(
    model,
    src_tensor,
    bos_id,
    eos_id,
    beam_size=5,
    max_len=200,
    n_best=5,
    device=None,
):
    if device is None:
        device = src_tensor.device
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(device).unsqueeze(0)  # (1, T, C)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)
        dec_init = torch.tanh(model.bridge(context)).unsqueeze(0)  # (1,B,H) with B=1

        beams = [(0.0, [bos_id], dec_init)]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for score, seq, hidden in beams:
                if seq[-1] == eos_id:
                    completed.append((score, seq, hidden))
                    continue

                cur = torch.tensor([seq], dtype=torch.long, device=device)  # (1, L)
                logits, hidden_next = model.decoder(cur, hidden)
                logp = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                topk = torch.topk(logp, beam_size)
                for idx, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                    new_seq = seq + [idx]
                    new_score = score - float(lp)
                    new_beams.append((new_score, new_seq, hidden_next))

            if not new_beams:
                break

            new_beams.sort(key=lambda x: x[0])
            beams = new_beams[:beam_size]

        completed.extend(beams)
        cleaned = [(s, _strip_bos_eos(seq, bos_id, eos_id)) for (s, seq, _) in completed]
        cleaned.sort(key=lambda x: x[0])
        return cleaned[:n_best]


# ---------- Main script ----------

def idseq_to_readable(id_seq, inv_map):
    if inv_map is None:
        return " ".join(str(t) for t in id_seq)
    return " ".join(str(inv_map.get(t, t)) for t in id_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, default="day_pairs_tokenids_bos_small.pkl")
    parser.add_argument("--token_map", type=str, default="tokenid_map.json")
    parser.add_argument("--model_type", type=str, choices=["gru", "transformer"], default="transformer")
    parser.add_argument("--checkpoint", type=str, default="transformer_tokenid_checkpoint.pth")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--beam", type=int, default=5)
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--lm_weight", type=float, default=0.5,
                        help="Weight on LM negative logprob in combined score.")
    parser.add_argument("--max_len", type=int, default=200)
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    print("Loading pairs:", args.pairs)
    pairs = pickle.load(open(args.pairs, "rb"))
    print("Pairs loaded:", len(pairs))

    mapping = json.load(open(args.token_map))
    PAD = int(mapping["PAD"])
    BOS = int(mapping["BOS"])
    EOS = int(mapping["EOS"])
    vocab_size = int(mapping["vocab_size"])
    print("PAD,BOS,EOS:", PAD, BOS, EOS, "vocab_size:", vocab_size)

    old2new = mapping.get("old2new", None)
    inv_map = None
    if old2new is not None:
        inv_map = {v: k for k, v in old2new.items()}

    # build model
    sample = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0)
    in_dim = sample.shape[2]
    if args.model_type == "gru":
        model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size).to(DEVICE)
    else:
        model = TransformerSeq2Seq(
            in_dim=in_dim,
            d_model=128,
            nhead=4,
            enc_layers=2,
            dec_layers=2,
            vocab_size=vocab_size,
        ).to(DEVICE)

    print("Loading checkpoint:", args.checkpoint)
    state = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    trues = []
    preds = []

    n_eval = min(args.n_eval, len(pairs))
    print("Evaluating on", n_eval, "examples with beam =", args.beam, "n_best =", args.n_best)

    for i, (neural, token_ids, _) in enumerate(pairs[:n_eval]):
        src = torch.tensor(neural, dtype=torch.float32)
        src_len = torch.tensor([src.size(0)], dtype=torch.long, device=DEVICE)

        if args.model_type == "gru":
            nbest = beam_search_gru_nbest(
                model, src, BOS, EOS,
                beam_size=args.beam,
                max_len=args.max_len,
                n_best=args.n_best,
                device=DEVICE,
            )
        else:
            nbest = beam_search_transformer_nbest(
                model, src, src_len,
                BOS, EOS,
                beam_size=args.beam,
                max_len=args.max_len,
                n_best=args.n_best,
                device=DEVICE,
            )

        # model-only best (for info)
        best_model_score, best_model_seq = nbest[0]

        # LM rescoring
        best_seq = None
        best_total = None
        for model_score, seq in nbest:
            lm_lp = lm_log_prob(seq)  # log P(seq)
            lm_neg = -float(lm_lp)
            total = model_score + args.lm_weight * lm_neg
            if best_total is None or total < best_total:
                best_total = total
                best_seq = seq

        pred_ids = best_seq

        # true tokens: remove PAD
        true_ids = [int(t) for t in token_ids if int(t) != PAD]

        trues.append(" ".join(str(t) for t in true_ids))
        preds.append(" ".join(str(t) for t in pred_ids))

        if i < 10:
            print("=== Example", i, "===")
            print("TRUE ids:", " ".join(str(t) for t in true_ids[:80]))
            print("PRED ids (model best):", " ".join(str(t) for t in best_model_seq[:80]))
            print("PRED ids (LM rescored):", " ".join(str(t) for t in pred_ids[:80]))
            print("READ TRUE:", idseq_to_readable(true_ids[:50], inv_map))
            print("READ PRED:", idseq_to_readable(pred_ids[:50], inv_map))
            print("Model-only score:", best_model_score, "Rescored best_total:", best_total)
            print("----")

    print("Token-level WER (beam + LM rescoring):", wer(trues, preds))


if __name__ == "__main__":
    main()
