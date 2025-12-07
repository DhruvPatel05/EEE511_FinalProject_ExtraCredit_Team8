# scripts/create_tokenid_pairs_with_bos.py
"""
Create token-id pairs that reserve PAD(0), BOS(1), EOS(2).
Input: day_pairs.pkl (created previously by load_step2_prepare_pairs.py).
Output:
  - day_pairs_tokenids_bos.pkl
  - day_pairs_tokenids_bos_small.pkl
  - tokenid_map.json
"""
import pickle, json

IN_PK = "day_pairs.pkl"   # original pairs: (neural_np, transcription_ids, maybe mapped string)
OUT_PK = "day_pairs_tokenids_bos.pkl"
OUT_SMALL = "day_pairs_tokenids_bos_small.pkl"
N_SMALL = 400

print("Loading:", IN_PK)
pairs = pickle.load(open(IN_PK, "rb"))
print("Total pairs:", len(pairs))

# collect unique token ids from TFRecord transcription arrays
all_ids = set()
for _, token_ids, _ in pairs:
    for t in token_ids:
        all_ids.add(int(t))
all_ids = sorted(all_ids)
print("Unique original tokens:", len(all_ids), "range:", all_ids[0], all_ids[-1])

# reserve special tokens
PAD = 0
BOS = 1
EOS = 2
shift = 3

old2new = {old: i + shift for i, old in enumerate(all_ids)}
new_vocab_size = len(old2new) + shift
print("New vocab size (including PAD,BOS,EOS):", new_vocab_size)

new_pairs = []
for neural, token_ids, _ in pairs:
    # remap token ids and append EOS
    remapped = [old2new[int(t)] for t in token_ids]
    remapped_with_eos = remapped + [EOS]
    new_pairs.append((neural, remapped_with_eos, None))

# save outputs
pickle.dump(new_pairs, open(OUT_PK, "wb"))
pickle.dump(new_pairs[:N_SMALL], open(OUT_SMALL, "wb"))
mapping = {"old2new": old2new, "PAD": PAD, "BOS": BOS, "EOS": EOS, "vocab_size": new_vocab_size}
open("tokenid_map.json", "w").write(json.dumps(mapping))
print("Saved:", OUT_PK, OUT_SMALL, "and tokenid_map.json")
