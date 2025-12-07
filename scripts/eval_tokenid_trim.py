# scripts/eval_tokenid_trim.py
import pickle, torch, argparse, math
from torch.utils.data import DataLoader
from jiwer import wer
from scripts.dataset import NeuralTextDataset, collate_fn
from scripts.models import Seq2SeqGRU, TransformerSeq2Seq, make_tgt_mask

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", type=str, default="day_pairs_tokenids_small.pkl")
parser.add_argument("--model_type", type=str, choices=["gru","transformer"], default="transformer")
parser.add_argument("--checkpoint", type=str, default="transformer_tokenid_checkpoint.pth")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--n_eval", type=int, default=100)
parser.add_argument("--max_gen", type=int, default=512)
args = parser.parse_args()

DEVICE = torch.device(args.device)

pairs = pickle.load(open(args.pairs, "rb"))
pairs = [(n, t, None) for (n,t,s) in pairs]
print("Loaded pairs:", len(pairs))

# helper to strip trailing padding zeros from a token list
def strip_trailing_zeros(token_list):
    # token_list may be list/iterable of ints
    if len(token_list) == 0:
        return []
    i = len(token_list)-1
    while i >= 0 and int(token_list[i]) == 0:
        i -= 1
    return token_list[:i+1]

# greedy decoders (same as previous eval_tokenid)
def greedy_decode_gru(model, src_tensor, max_len=512):
    with torch.no_grad():
        src = src_tensor.to(DEVICE).unsqueeze(0)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)
        dec_init = torch.tanh(model.bridge(context)).unsqueeze(0)
        cur = torch.tensor([[0]], dtype=torch.long).to(DEVICE)
        preds = []
        hidden = dec_init
        for _ in range(max_len):
            logits, hidden = model.decoder(cur, hidden)
            next_logits = logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1)
            nid = int(next_id.item())
            preds.append(nid)
            cur = torch.cat([cur, next_id.unsqueeze(1)], dim=1)
            # break if we see long repetition or trailing zeros
            if len(preds) > 1 and preds[-1] == 0 and all(p == 0 for p in preds[-10:]):
                break
            if len(preds) > 200:
                break
        # trim trailing zeros
        return [p for p in preds if p != 0] if any(preds) else preds

def greedy_decode_transformer(model, src_tensor, src_len, max_len=512):
    with torch.no_grad():
        src = src_tensor.to(DEVICE).unsqueeze(0)
        src_padding_mask = (torch.arange(src.size(1))[None,:].to(DEVICE) >= src_len)
        src_proj = model.input_proj(src) * math.sqrt(model.d_model)
        memory = model.encoder(model.pos(src_proj), src_key_padding_mask=src_padding_mask)
        ys = torch.tensor([[0]], dtype=torch.long).to(DEVICE)
        preds = []
        for i in range(max_len):
            tgt_mask = make_tgt_mask(ys.size(1)).to(DEVICE)
            tgt_emb = model.pos(model.embedding(ys))
            out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            logits = model.out(out[:, -1, :])
            next_id = logits.argmax(dim=-1)
            nid = int(next_id.item())
            preds.append(nid)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
            if len(preds) > 1 and preds[-1] == 0 and all(p == 0 for p in preds[-10:]):
                break
            if len(preds) > 200:
                break
        return [p for p in preds if p != 0] if any(preds) else preds

# load model
sample_x = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0)
in_dim = sample_x.shape[2]
if args.model_type == "gru":
    model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=128, dec_hidden=128, vocab_size=1).to(DEVICE)  # vocab_size will be overwritten
else:
    model = TransformerSeq2Seq(in_dim=in_dim, d_model=128, nhead=4, enc_layers=2, dec_layers=2, vocab_size=1).to(DEVICE)

# fix vocab size (infer)
maxid = -1
for _, t, _ in pairs:
    if len(t)>0:
        maxid = max(maxid, max(int(x) for x in t))
vocab_size = int(maxid + 1)
print("token-id vocab_size:", vocab_size)
# rebuild model with correct vocab size
if args.model_type == "gru":
    model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size).to(DEVICE)
else:
    model = TransformerSeq2Seq(in_dim=in_dim, d_model=128, nhead=4, enc_layers=2, dec_layers=2, vocab_size=vocab_size).to(DEVICE)

sd = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(sd)
model.eval()
print("Loaded model:", args.checkpoint)

n_eval = min(args.n_eval, len(pairs))
trues = []
preds = []
for i in range(n_eval):
    x_np, token_ids, _ = pairs[i]
    x = torch.tensor(x_np, dtype=torch.float32)
    src_len = x_np.shape[0]
    if args.model_type == "gru":
        pred_ids = greedy_decode_gru(model, x, max_len=args.max_gen)
    else:
        pred_ids = greedy_decode_transformer(model, x, src_len, max_len=args.max_gen)

    true_trim = strip_trailing_zeros(list(token_ids))
    pred_trim = strip_trailing_zeros(list(pred_ids))
    # convert to space-separated strings
    true_str = " ".join(str(int(t)) for t in true_trim)
    pred_str = " ".join(str(int(t)) for t in pred_trim)
    trues.append(true_str)
    preds.append(pred_str)
    if i < 5:
        print("TRUE ids:", true_str)
        print("PRED ids:", pred_str)
        print("-----")

print("WER (token-level) after trimming padding:", wer(trues, preds))
