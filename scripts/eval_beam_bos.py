# scripts/eval_beam_bos.py
import argparse, pickle, math, json, torch, heapq
from scripts.models import Seq2SeqGRU, TransformerSeq2Seq, make_tgt_mask
from jiwer import wer

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=str, default="day_pairs_tokenids_bos_small.pkl")
parser.add_argument("--model_type", type=str, choices=["gru","transformer"], default="transformer")
parser.add_argument("--checkpoint", type=str, default="transformer_tokenid_checkpoint.pth")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--n_eval", type=int, default=100)
parser.add_argument("--beam", type=int, default=5)
parser.add_argument("--max_len", type=int, default=200)
args = parser.parse_args()

DEVICE = torch.device(args.device)
pairs = pickle.load(open(args.pairs, "rb"))
pairs = [(n,t,None) for (n,t,s) in pairs]
mapping = json.load(open("tokenid_map.json"))
PAD = int(mapping["PAD"]); BOS = int(mapping["BOS"]); EOS = int(mapping["EOS"])
vocab_size = int(mapping["vocab_size"])

# try to load old2new inverse map to convert back to original token labels (if available)
old2new = mapping.get("old2new", None)
inv_map = None
if old2new is not None:
    inv_map = {v:k for k,v in old2new.items()}

# build model
sample = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0)
in_dim = sample.shape[2]
if args.model_type == "gru":
    model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size).to(DEVICE)
else:
    model = TransformerSeq2Seq(in_dim=in_dim, d_model=128, nhead=4, enc_layers=2, dec_layers=2, vocab_size=vocab_size).to(DEVICE)
sd = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(sd); model.eval()
print("Loaded", args.checkpoint)

def idseq_to_readable(ids):
    # ids: list of ints (new token ids). If inv_map exists, map back to original token (string),
    # otherwise show numeric id string.
    if inv_map is None:
        return " ".join(str(i) for i in ids)
    out = []
    for nid in ids:
        old = inv_map.get(str(nid), None) or inv_map.get(nid, None)
        if old is None:
            out.append(str(nid))
        else:
            out.append(str(old))
    return " ".join(out)

# Beam search for transformer (batch size 1)
def beam_search_transformer(model, src_tensor, src_len, beam_size=5, max_len=200):
    with torch.no_grad():
        src = src_tensor.to(DEVICE).unsqueeze(0)
        src_padding_mask = (torch.arange(src.size(1))[None,:].to(DEVICE) >= src_len)
        src_proj = model.input_proj(src) * math.sqrt(model.d_model)
        memory = model.encoder(model.pos(src_proj), src_key_padding_mask=src_padding_mask)

        # each beam: (score (neg log prob), token_list, ys_tensor)
        beams = [(0.0, [BOS], torch.tensor([[BOS]], dtype=torch.long).to(DEVICE))]
        completed = []
        for _step in range(max_len):
            new_beams = []
            for score, seq, ys in beams:
                tgt_mask = make_tgt_mask(ys.size(1)).to(DEVICE)
                out = model.decoder(model.pos(model.embedding(ys)), memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
                logits = model.out(out[:, -1, :])  # (1, V)
                logp = torch.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
                topk = torch.topk(logp, beam_size)
                vals = topk.values.cpu().numpy()
                idxs = topk.indices.cpu().numpy()
                for v,i in zip(vals, idxs):
                    new_score = score - float(v)  # lower negative logprob = better
                    new_seq = seq + [int(i)]
                    new_ys = torch.cat([ys, torch.tensor([[int(i)]], dtype=torch.long).to(DEVICE)], dim=1)
                    if int(i) == EOS:
                        completed.append((new_score, new_seq))
                    else:
                        new_beams.append((new_score, new_seq, new_ys))
            # keep top beam_size candidates
            new_beams = sorted(new_beams, key=lambda x: x[0])[:beam_size]
            beams = new_beams
            if len(completed) >= beam_size:
                break
        if completed:
            completed = sorted(completed, key=lambda x: x[0])
            return completed[0][1][1:-1]  # remove BOS and EOS
        elif beams:
            # fallback: best partial beam
            best = sorted(beams, key=lambda x: x[0])[0]
            return best[1][1:]  # remove BOS
        else:
            return []

# GRU beam search (simple left-to-right scoring)
def beam_search_gru(model, src_tensor, beam_size=5, max_len=200):
    with torch.no_grad():
        src = src_tensor.to(DEVICE).unsqueeze(0)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)
        dec_init = torch.tanh(model.bridge(context)).unsqueeze(0)
        beams = [(0.0, [BOS], dec_init)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for score, seq, hidden in beams:
                cur = torch.tensor([seq], dtype=torch.long).to(DEVICE)  # shape (1, len)
                logits, hidden_next = model.decoder(cur, hidden)
                logp = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                topk = torch.topk(logp, beam_size)
                vals = topk.values.cpu().numpy()
                idxs = topk.indices.cpu().numpy()
                for v,i in zip(vals, idxs):
                    new_score = score - float(v)
                    new_seq = seq + [int(i)]
                    if int(i) == EOS:
                        completed.append((new_score, new_seq))
                    else:
                        new_beams.append((new_score, new_seq, hidden_next))
            new_beams = sorted(new_beams, key=lambda x: x[0])[:beam_size]
            beams = new_beams
            if len(completed) >= beam_size:
                break
        if completed:
            completed = sorted(completed, key=lambda x: x[0])
            return completed[0][1][1:-1]
        elif beams:
            return beams[0][1][1:]
        else:
            return []

# run eval
n_eval = min(args.n_eval, len(pairs))
trues = []
preds = []
for i in range(n_eval):
    x_np, token_ids, _ = pairs[i]
    x = torch.tensor(x_np, dtype=torch.float32)
    src_len = x_np.shape[0]
    if args.model_type == "gru":
        pred_ids = beam_search_gru(model, x, beam_size=args.beam, max_len=args.max_len)
    else:
        pred_ids = beam_search_transformer(model, x, src_len, beam_size=args.beam, max_len=args.max_len)

    # true tokens - remove PAD if present (but should have EOS appended)
    true_ids = [int(t) for t in token_ids if int(t) != PAD]
    trues.append(" ".join(str(t) for t in true_ids))
    preds.append(" ".join(str(t) for t in pred_ids))

    if i < 10:
        print("TRUE ids:", " ".join(str(t) for t in true_ids[:120]))
        print("PRED ids:", " ".join(str(t) for t in pred_ids[:120]))
        print("READ TRUE:", idseq_to_readable(true_ids[:50]))
        print("READ PRED:", idseq_to_readable(pred_ids[:50]))
        print("----")

print("Token-level WER (beam):", wer(trues, preds))
