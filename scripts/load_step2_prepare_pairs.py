# load_step2_prepare_pairs.py
"""
Load TFRecords for one day, parse inputFeatures and transcription, reshape neural arrays,
and pair with sentence strings (from .mat if available). Saves pairs to day_pairs.pkl.

Requires:
  pip install tensorflow==2.12.0 scipy numpy
"""

import os
import glob
import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import pickle

# ---------- USER CONFIG ----------
TFRECORD_DIR = "derived/tfRecords/t12.2022.07.14"   # change if needed
SENTENCE_MAT = "sentences/t12.2022.07.14_sentences.mat"  # change if needed
MAX_EXAMPLES = None   # set to integer to load only first N examples for speed (None = all)
OUT_PICKLE = "day_pairs.pkl"
# ---------------------------------

def inspect_tfrecord_one_file(tfrecord_file, max_to_inspect=1):
    print("Inspecting:", tfrecord_file)
    ds = tf.data.TFRecordDataset(tfrecord_file)
    for i, raw in enumerate(ds.take(max_to_inspect)):
        example = tf.train.Example()
        example.ParseFromString(raw.numpy())
        keys = list(example.features.feature.keys())
        print(" keys:", keys)
        for k in keys:
            f = example.features.feature[k]
            if f.bytes_list.value:
                print(f"  {k}: bytes_list length = {len(f.bytes_list.value)}")
            if f.float_list.value:
                print(f"  {k}: float_list length = {len(f.float_list.value)}")
            if f.int64_list.value:
                print(f"  {k}: int64_list length = {len(f.int64_list.value)}")
        break

def parse_single_example(serialized_example):
    """
    Parse a tf.train.Example (serialized bytes) into numpy arrays:
      - neural: shape (T, C) numpy.float32
      - transcript_ids: 1D numpy.int64 (token ids)
      - T: scalar int (nTimeSteps)
    """
    # define parsing features: inputFeatures may be stored as VarLenFeature float32,
    # transcription as VarLenFeature int64, nTimeSteps as fixed length scalar int64
    feature_spec = {
        "inputFeatures": tf.io.VarLenFeature(tf.float32),
        "transcription": tf.io.VarLenFeature(tf.int64),
        "nTimeSteps": tf.io.FixedLenFeature([], tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_spec)

    # dense vectors from sparse
    neural_flat = tf.sparse.to_dense(parsed["inputFeatures"])  # 1D float tensor
    T = tf.cast(parsed["nTimeSteps"], tf.int32)               # scalar int tensor
    transcription_sparse = parsed["transcription"]
    transcription = tf.sparse.to_dense(transcription_sparse)  # 1D int tensor (token ids)

    # convert to numpy
    neural_flat_np = neural_flat.numpy()
    T_np = int(T.numpy())
    transcription_np = transcription.numpy().astype(np.int64)

    # infer channels and reshape
    if T_np <= 0:
        raise ValueError("nTimeSteps is zero or negative for example.")
    total_len = neural_flat_np.size
    if total_len % T_np != 0:
        # sometimes neural stored flattened with extra dims or mismatch;
        # try to locate the 'true' T by reading alternate fields (rare)
        # but we'll raise for now to prompt inspection
        raise RuntimeError(f"Cannot reshape neural_flat (len={total_len}) with T={T_np}. Not divisible.")
    channels = total_len // T_np
    neural = neural_flat_np.reshape((T_np, channels)).astype(np.float32)

    return neural, transcription_np

def load_all_tfrecords(tfrecord_dir, max_examples=None):
    # find TFRecord files in subfolders (train/test/competitionHoldOut)
    tf_files = []
    for root, dirs, files in os.walk(tfrecord_dir):
        for f in sorted(files):
            if ".tfrecord" in f:
                tf_files.append(os.path.join(root, f))
    if not tf_files:
        raise FileNotFoundError(f"No TFRecord files found under {tfrecord_dir}")
    print("Found TFRecord files:", tf_files[:5], " (total {})".format(len(tf_files)))

    neural_list = []
    transcription_list = []

    count = 0
    for tf_file in tf_files:
        print("Reading TFRecord file:", tf_file)
        ds = tf.data.TFRecordDataset(tf_file)
        for raw in ds:
            try:
                neural, transcript_ids = parse_single_example(raw)
            except Exception as e:
                print("Skipping example due to parse/reshape error:", str(e))
                continue
            neural_list.append(neural)
            transcription_list.append(transcript_ids)
            count += 1
            if (max_examples is not None) and (count >= max_examples):
                print("Reached max_examples limit:", max_examples)
                return neural_list, transcription_list
    print("Total parsed examples:", len(neural_list))
    return neural_list, transcription_list

def load_mat_sentences(mat_path):
    if not os.path.exists(mat_path):
        print("Sentence .mat not found at", mat_path)
        return None
    print("Loading sentences MAT:", mat_path)
    mat = loadmat(mat_path)
    print("MAT keys:", list(mat.keys()))
    # heuristics to find string list variable
    candidate_keys = [k for k in mat.keys() if ('sentence' in k.lower() or 'utter' in k.lower() or 'text' in k.lower())]
    if not candidate_keys:
        # sometimes variable named 'sentences' or 'utterances'
        candidate_keys = [k for k in mat.keys() if k not in ('__header__','__version__','__globals__')]
    # try to extract a python list of strings from first candidate that looks right
    for key in candidate_keys:
        try:
            arr = mat[key]
            # If array of strings or object dtype, try to flatten
            if isinstance(arr, np.ndarray):
                # many Willett mats store a nested cell array -> convert
                # Flatten and convert each entry to Python str
                flat = np.ravel(arr)
                strings = []
                for item in flat:
                    if isinstance(item, str):
                        strings.append(item)
                    else:
                        # item may be a 1-element array of bytes or char codes
                        try:
                            s = ""
                            # attempt to decode MATLAB char arrays
                            if isinstance(item, np.ndarray) and item.size > 0:
                                # sometimes it's array of shape (N,1) of single characters
                                s = "".join([chr(c) for c in item.flatten()]) if item.dtype.kind in ('u','U','S','O','i','f') else str(item)
                            else:
                                s = str(item)
                            # cleanup
                            s = s.strip()
                            if s == '': 
                                continue
                            strings.append(s)
                        except Exception:
                            continue
                # sanity check: must have reasonable number of strings
                if len(strings) > 5:
                    print(f"Using MAT key '{key}' as sentence strings ({len(strings)} entries).")
                    return strings
        except Exception as e:
            continue
    print("Couldn't find sentence strings in MAT by heuristics.")
    return None

def main():
    print("Root directory:", TFRECORD_DIR)
    # quick inspection of one file to show keys (optional)
    # find one .tfrecord file to inspect
    first_tfr = None
    for root, dirs, files in os.walk(TFRECORD_DIR):
        for f in files:
            if ".tfrecord" in f:
                first_tfr = os.path.join(root, f)
                break
        if first_tfr:
            break
    if first_tfr:
        inspect_tfrecord_one_file(first_tfr, max_to_inspect=1)

    # parse all tfrecords
    neural_list, transcription_list = load_all_tfrecords(TFRECORD_DIR, max_examples=MAX_EXAMPLES)

    if len(neural_list) == 0:
        raise RuntimeError("No neural examples parsed. Inspect TFRecords or update parsing keys.")

    # load sentences .mat if possible
    sentence_strings = load_mat_sentences(SENTENCE_MAT)
    if sentence_strings:
        # If counts match, link them by index. If mismatch, use min length and warn.
        if len(sentence_strings) == len(transcription_list):
            print("Number of .mat sentence strings equals TFRecord examples. Using direct mapping.")
            mapped_strings = sentence_strings
        else:
            print("Counts differ: TFRecord examples =", len(transcription_list), ", .mat strings =", len(sentence_strings))
            m = min(len(transcription_list), len(sentence_strings))
            print("Using first", m, "pairs.")
            # trim lists to same length
            neural_list = neural_list[:m]
            transcription_list = transcription_list[:m]
            mapped_strings = sentence_strings[:m]
    else:
        # No sentence strings available; we will decode transcript ids to string later if token2char exists.
        mapped_strings = [None] * len(transcription_list)
        print("No human-readable sentences found. Transcript IDs retained; you'll need token->char mapping.")

    # convert transcription id arrays to 1D python lists
    transcription_list = [np.array(t).astype(np.int64).flatten().tolist() for t in transcription_list]

    # create final pairs
    pairs = []
    for i, (narr, tid) in enumerate(zip(neural_list, transcription_list)):
        pairs.append((narr, tid, mapped_strings[i] if mapped_strings is not None else None))

    # save a small subset and full pairs
    # small subset first 400 if exists
    small_n = min(400, len(pairs))
    with open("day_pairs_small.pkl", "wb") as f:
        pickle.dump(pairs[:small_n], f)
    with open(OUT_PICKLE, "wb") as f:
        pickle.dump(pairs, f)

    print(f"Saved {small_n} small pairs -> day_pairs_small.pkl and full pairs -> {OUT_PICKLE}")
    print("Example pair shapes:")
    print(" neural[0].shape:", pairs[0][0].shape)
    print(" transcript_ids length:", len(pairs[0][1]))
    print(" mapped_string (may be None):", pairs[0][2])

if __name__ == "__main__":
    main()
