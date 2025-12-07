# scripts/dataset.py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

def build_vocab_from_pairs(pairs):
    """
    Build character-level vocabulary from mapped strings present in pairs.
    pairs: list of (neural_np, token_id_list, mapped_string_or_None)
    Returns (tokens_list, char2idx, idx2char) or (None, None, None) if no mapped strings.
    """
    chars = set()
    for _, _, s in pairs:
        if s:
            chars.update(list(s))
    if len(chars) == 0:
        return None, None, None
    chars = sorted(list(chars))
    tokens = [PAD, BOS, EOS] + chars
    char2idx = {c: i for i, c in enumerate(tokens)}
    idx2char = {i: c for i, c in enumerate(tokens)}
    return tokens, char2idx, idx2char

class NeuralTextDataset(Dataset):
    def __init__(self, pairs, char2idx=None):
        """
        pairs: list of tuples (neural_np (T,C), token_ids_list, mapped_string_or_None)
        char2idx: mapping if using mapped strings (character-level). If None, we will use token ids.
        """
        self.pairs = pairs
        self.char2idx = char2idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, token_ids, s = self.pairs[idx]
        x = torch.tensor(x, dtype=torch.float32)  # (T, C)
        if (s is not None) and (self.char2idx is not None):
            toks = [BOS] + list(s) + [EOS]
            y = torch.tensor([self.char2idx.get(ch, 0) for ch in toks], dtype=torch.long)
        else:
            # use token ids directly
            y = torch.tensor(token_ids, dtype=torch.long)
        return x, y

def collate_fn(batch):
    """
    Pads neural X to max time and pads Y to max length.
    Returns: X_padded (B, T, C), x_lens, Y_padded (B, L), y_lens
    Note: uses -100 as ignore_index for padded target positions.
    """
    xs, ys = zip(*batch)
    T_max = max([x.shape[0] for x in xs])
    C = xs[0].shape[1]
    B = len(xs)
    X_p = torch.zeros((B, T_max, C), dtype=torch.float32)
    x_lens = []
    for i, x in enumerate(xs):
        X_p[i, :x.shape[0], :] = x
        x_lens.append(x.shape[0])

    L_max = max([y.shape[0] for y in ys])
    Y_p = torch.full((B, L_max), fill_value=-100, dtype=torch.long)  # -100 -> ignore_index
    y_lens = []
    for i, y in enumerate(ys):
        Y_p[i, :y.shape[0]] = y
        y_lens.append(y.shape[0])

    return X_p, torch.tensor(x_lens, dtype=torch.long), Y_p, torch.tensor(y_lens, dtype=torch.long)
