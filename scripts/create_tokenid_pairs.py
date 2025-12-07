# scripts/create_tokenid_pairs.py
import pickle
import numpy as np
import os

IN_PK = "day_pairs.pkl"         # full pairs produced earlier (neural, transcription_ids, mapped_string or None)
OUT_PK = "day_pairs_tokenids.pkl"
OUT_SMALL = "day_pairs_tokenids_small.pkl"
N_SMALL = 400

pairs = pickle.load(open(IN_PK, "rb"))
print("Loaded pairs:", len(pairs))

# collect original transcription ids (second element) across pairs
all_ids = set()
for _, token_ids, _ in pairs:
    # token_ids is list/array of ints taken from TFRecord 'transcription'
    for t in token_ids:
        all_ids.add(int(t))
all_ids = sorted(list(all_ids))
print("Unique token ids count:", len(all_ids), "min/max:", all_ids[0], all_ids[-1])

# build remapping old->new contiguous
old2new = {old: i for i, old in enumerate(all_ids)}
V = len(old2new)
print("New vocab size:", V)

# remap token arrays and set mapped_string = None (we will use token ids as targets)
new_pairs = []
for neural, token_ids, _ in pairs:
    token_ids = [old2new[int(t)] for t in token_ids]
    new_pairs.append((neural, token_ids, None))

# save full and small
pickle.dump(new_pairs, open(OUT_PK, "wb"))
pickle.dump(new_pairs[:N_SMALL], open(OUT_SMALL, "wb"))
print("Saved:", OUT_PK, "and", OUT_SMALL)
