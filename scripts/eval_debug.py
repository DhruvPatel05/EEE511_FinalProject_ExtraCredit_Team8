# scripts/eval_debug.py
import pickle, torch, numpy as np
from scripts.dataset import build_vocab_from_pairs
from scripts.models import TransformerSeq2Seq, make_tgt_mask
from torch.nn.functional import softmax

PAIRS_PKL = "day_pairs_small.pkl"
CKPT = "transformer_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pairs = pickle.load(open(PAIRS_PKL, "rb"))
tokens, char2idx, idx2char = build_vocab_from_pairs(pairs)
pairs = [p for p in pairs if p[2] is not None]
print("Num pairs (with strings):", len(pairs))
print("vocab size:", len(char2idx))
print("sample tokens (first 30):", tokens[:30])
print("PAD idx:", char2idx.get('<pad>'), "BOS idx:", char2idx.get('<bos>'), "EOS idx:", char2idx.get('<eos>'))

# load small sample
x_np, token_ids, s = pairs[0]
T = x_np.shape[0]
print("Sample T, C:", x_np.shape, "example string:", s)

# build model and load ckpt
in_dim = x_np.shape[1]
vocab_size = len(char2idx)
model = TransformerSeq2Seq(in_dim=in_dim, d_model=128, nhead=4, enc_layers=2, dec_layers=2, vocab_size=vocab_size).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(sd)
model.eval()
print("Loaded model OK")

# prepare src and src mask
src = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1,T,C)
src_len = T
src_padding_mask = (torch.arange(src.size(1))[None,:].to(DEVICE) >= src_len)

# compute memory
with torch.no_grad():
    src_proj = model.input_proj(src) * (model.d_model ** 0.5)
    src_proj = model.pos(src_proj)
    memory = model.encoder(src_proj, src_key_padding_mask=src_padding_mask)

# Greedy one-step logits (start with BOS)
bos = char2idx.get('<bos>')
eos = char2idx.get('<eos>')
print("BOS,EOS indices:", bos, eos)

ys = torch.tensor([[bos]], dtype=torch.long).to(DEVICE)
tgt_mask = make_tgt_mask(ys.size(1)).to(DEVICE)
tgt_emb = model.pos(model.embedding(ys))
out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
logits = model.out(out[:, -1, :])
probs = softmax(logits.detach(), dim=-1).cpu().numpy().flatten()
topk = np.argsort(probs)[-10:][::-1]
print("Top 10 tokens (idx -> char -> prob):")
for idx in topk[:10]:
    ch = idx2char.get(idx, '<unk>')
    print(idx, ch, f"{probs[idx]:.4f}")

# If the top-1 is same token 25, show why
print("Top-1 token:", topk[0])

# Teacher-forcing check: give true first token and see next-step logits
# Build true target sequence tokens from the mapped string
true_chars = ['<bos>'] + list(pairs[0][2]) + ['<eos>']
true_ids = torch.tensor([[char2idx[c] for c in true_chars]], dtype=torch.long).to(DEVICE)
with torch.no_grad():
    tgt_in = true_ids[:, :2]  # BOS + first char
    tgt_mask = make_tgt_mask(tgt_in.size(1)).to(DEVICE)
    tgt_emb = model.pos(model.embedding(tgt_in))
    out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
    logits_next = model.out(out[:, -1, :])
    probs_next = softmax(logits_next, dim=-1).cpu().numpy().flatten()
    topk_next = np.argsort(probs_next)[-10:][::-1]
    print("Teacher-forced next-step top tokens:")
    for idx in topk_next[:10]:
        print(idx, idx2char.get(idx,'<unk>'), f"{probs_next[idx]:.4f}")
