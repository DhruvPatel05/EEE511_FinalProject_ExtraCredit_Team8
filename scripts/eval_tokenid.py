# scripts/eval_tokenid.py
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

# load pairs
pairs = pickle.load(open(args.pairs, "rb"))
print("Loaded pairs:", len(pairs))
pairs = [(n, t, None) for (n, t, s) in pairs]  # ensure format

# dataset (we use batch_size=1 for greedy decoding)
ds = NeuralTextDataset(pairs, char2idx=None)
dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

# infer vocab_size
maxid = -1
for _, t, _ in pairs:
    if len(t) > 0:
        maxid = max(maxid, max(int(x) for x in t))
vocab_size = int(maxid + 1)
print("token-id vocab_size:", vocab_size)

# build model
sample = next(iter(dl))
in_dim = sample[0].shape[2]
if args.model_type == "gru":
    model = Seq2SeqGRU(in_dim=in_dim, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size).to(DEVICE)
else:
    model = TransformerSeq2Seq(in_dim=in_dim, d_model=128, nhead=4, enc_layers=2, dec_layers=2, vocab_size=vocab_size).to(DEVICE)

# load checkpoint
sd = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(sd)
model.eval()
print("Loaded model:", args.checkpoint)

def greedy_decode_gru(model, src_tensor, max_len=512):
    with torch.no_grad():
        src = src_tensor.to(DEVICE).unsqueeze(0)  # (1,T,C)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)
        dec_init = torch.tanh(model.bridge(context)).unsqueeze(0)
        # start token: use 0 as start if no BOS; we will use previous token id from target when teacher-forcing isn't possible
        # Here we start with zero token (0), but it's dataset dependent
        cur = torch.tensor([[0]], dtype=torch.long).to(DEVICE)
        preds = []
        hidden = dec_init
        for _ in range(max_len):
            logits, hidden = model.decoder(cur, hidden)
            next_logits = logits[:, -1, :]  # (1, V)
            next_id = next_logits.argmax(dim=-1)  # (1,)
            nid = int(next_id.item())
            preds.append(nid)
            cur = torch.cat([cur, next_id.unsqueeze(1)], dim=1)  # feed last token (keeps longer input, acceptable for small generation)
            if len(preds) > 1 and preds[-1] == preds[-2] and len(preds) > 50:
                # optional break heuristic if stuck repeating
                break
        return preds

def greedy_decode_transformer(model, src_tensor, src_len, max_len=512):
    with torch.no_grad():
        src = src_tensor.to(DEVICE).unsqueeze(0)  # (1,T,C)
        src_padding_mask = (torch.arange(src.size(1))[None,:].to(DEVICE) >= src_len)
        src_proj = model.input_proj(src) * math.sqrt(model.d_model)
        memory = model.encoder(model.pos(src_proj), src_key_padding_mask=src_padding_mask)
        ys = torch.tensor([[0]], dtype=torch.long).to(DEVICE)  # start with token id 0
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
            if len(preds) > 1 and preds[-1] == preds[-2] and len(preds) > 50:
                break
        return preds

# evaluate N examples
n_eval = min(args.n_eval, len(pairs))
true_texts = []
pred_texts = []
for i in range(n_eval):
    x_np, token_ids, _ = pairs[i]
    x = torch.tensor(x_np, dtype=torch.float32)
    src_len = x_np.shape[0]
    if args.model_type == "gru":
        pred_ids = greedy_decode_gru(model, x, max_len=args.max_gen)
    else:
        pred_ids = greedy_decode_transformer(model, x, src_len, max_len=args.max_gen)
    # convert both sequences to space-joined strings of token ids for jiwer
    true_str = " ".join(str(int(t)) for t in token_ids if int(t) >= 0)
    pred_str = " ".join(str(int(t)) for t in pred_ids)
    true_texts.append(true_str)
    pred_texts.append(pred_str)
    if i < 5:
        print("TRUE ids:", true_str[:200])
        print("PRED ids:", pred_str[:200])
        print("-----")

print("Computing token-level WER (jiwer on whitespace-tokenized strings)...")
score = wer(true_texts, pred_texts)
print("Token-level WER:", score)
