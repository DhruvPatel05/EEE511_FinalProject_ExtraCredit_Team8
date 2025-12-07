# scripts/models.py
import torch
import torch.nn as nn
import math

# --- GRU encoder/decoder seq2seq (simple) ---
class EncoderGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers=2, bidirectional=True):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True, num_layers=n_layers, bidirectional=bidirectional)
        self.bidir = bidirectional
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out, h = self.gru(x)
        return out, h

class DecoderGRU(nn.Module):
    def __init__(self, dec_hidden, out_vocab, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(out_vocab, embed_dim)
        self.gru = nn.GRU(embed_dim, dec_hidden, batch_first=True)
        self.fc = nn.Linear(dec_hidden, out_vocab)

    def forward(self, y_inp, hidden=None):
        emb = self.embed(y_inp)
        out, h = self.gru(emb, hidden)
        logits = self.fc(out)
        return logits, h

class Seq2SeqGRU(nn.Module):
    def __init__(self, in_dim, enc_hidden, dec_hidden, vocab_size):
        super().__init__()
        self.encoder = EncoderGRU(in_dim, enc_hidden, n_layers=2, bidirectional=True)
        self.bridge = nn.Linear(enc_hidden * 2, dec_hidden)
        self.decoder = DecoderGRU(dec_hidden, vocab_size)

    def forward(self, x, y_inp):
        enc_out, _ = self.encoder(x)              # enc_out: (B, T, enc_hidden*dirs)
        context = enc_out.mean(dim=1)             # (B, enc_hidden*2)
        dec_init = torch.tanh(self.bridge(context)).unsqueeze(0)  # (1,B,dec_hidden)
        logits, _ = self.decoder(y_inp, dec_init)
        return logits

# --- Transformer seq2seq ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, in_dim, d_model, nhead, enc_layers, dec_layers, vocab_size):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos(src)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos(tgt_emb)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = self.out(out)
        return logits

def make_tgt_mask(sz):
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask
