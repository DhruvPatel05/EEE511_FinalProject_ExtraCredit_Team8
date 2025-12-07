# scripts/eval_and_infer.py
import pickle, torch
from torch.utils.data import DataLoader
from jiwer import wer
from scripts.dataset import NeuralTextDataset, build_vocab_from_pairs, collate_fn
from scripts.models import TransformerSeq2Seq, Seq2SeqGRU, make_tgt_mask

# CONFIG - edit as needed
PAIRS_PKL = "day_pairs_small.pkl"
MODEL_TYPE = "transformer"   # "transformer" or "gru"
CHECKPOINT = "transformer_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GEN = 300

def greedy_decode_transformer(model, src, src_len, char2idx, idx2char):
    model.eval()
    with torch.no_grad():
        src = src.to(DEVICE).unsqueeze(0)
        src_padding_mask = (torch.arange(src.size(1))[None, :].to(DEVICE) >= src_len)
        memory = model.encoder(model.pos(model.input_proj(src) * (model.d_model ** 0.5)), src_key_padding_mask=src_padding_mask) if hasattr(model, 'pos') else model.encoder(src, src_key_padding_mask=src_padding_mask)
        ys = torch.tensor([[char2idx.get("<bos>", 1)]], dtype=torch.long).to(DEVICE)
        for _ in range(MAX_GEN):
            tgt_mask = make_tgt_mask(ys.size(1)).to(DEVICE)
            tgt_emb = model.pos(model.embedding(ys)) if hasattr(model, 'pos') else model.embedding(ys)
            out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
            prob = model.out(out[:, -1, :])
            next_tok = prob.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == char2idx.get("<eos>", 2):
                break
        toks = ys.squeeze().tolist()
        s = "".join([idx2char[t] for t in toks if t not in (char2idx.get("<bos>"), char2idx.get("<eos>"))])
        return s

def greedy_decode_gru(model, src, src_len, char2idx, idx2char):
    model.eval()
    with torch.no_grad():
        src = src.to(DEVICE).unsqueeze(0)
        enc_out, _ = model.encoder(src)
        context = enc_out.mean(dim=1)
        dec_init = torch.tanh(model.bridge(context)).unsqueeze(0)
        ys = torch.tensor([[char2idx.get("<bos>", 1)]], dtype=torch.long).to(DEVICE)
        for _ in range(MAX_GEN):
            logits, _ = model.decoder(ys, dec_init)
            prob = logits[:, -1, :]
            next_tok = prob.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == char2idx.get("<eos>", 2):
                break
        toks = ys.squeeze().tolist()
        s = "".join([idx2char[t] for t in toks if t not in (char2idx.get("<bos>"), char2idx.get("<eos>"))])
        return s

def main():
    pairs = pickle.load(open(PAIRS_PKL, "rb"))
    tokens, char2idx, idx2char = build_vocab_from_pairs(pairs)
    if char2idx is None:
        raise RuntimeError("No mapped strings available for decoding. Ensure day_pairs_small.pkl contains mapped strings.")
    # filter to mapped-string pairs for evaluation
    pairs = [p for p in pairs if p[2] is not None]
    ds = NeuralTextDataset(pairs, char2idx=char2idx)
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn)

    # load model
    sample = next(iter(dl))
    in_dim = sample[0].shape[2]
    vocab_size = len(char2idx)
    if MODEL_TYPE == "transformer":
        model = TransformerSeq2Seq(in_dim=in_dim, d_model=128, nhead=4, enc_layers=2, dec_layers=2, vocab_size=vocab_size).to(DEVICE)
    else:
        model = Seq2SeqGRU(in_dim, enc_hidden=128, dec_hidden=128, vocab_size=vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print("Loaded model:", CHECKPOINT)

    trues = []
    preds = []
    N = min(50, len(pairs))
    for i in range(N):
        x_np, token_ids, s = pairs[i]
        src_len = x_np.shape[0]
        src = torch.tensor(x_np, dtype=torch.float32)
        if MODEL_TYPE == "transformer":
            pred = greedy_decode_transformer(model, src, src_len, char2idx, idx2char)
        else:
            pred = greedy_decode_gru(model, src, src_len, char2idx, idx2char)
        trues.append(s)
        preds.append(pred)
        if i < 5:
            print("TRUE:", s)
            print("PRED:", pred)
            print("----")
    print("Evaluated", N, "samples. Computing WER...")
    print("WER:", wer(trues, preds))

if __name__ == "__main__":
    main()
