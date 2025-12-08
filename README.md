# EEE511_FinalProject_ExtraCredit_Team8
Al-Powered Brain-Computer Interfaces for Restoring Speech and Silent Communication
# AI-Powered BCI for Speech and Communication Restoration

This repository contains the code for our project on decoding neural activity into text using sequence-to-sequence models (GRU and Transformer).  
We compare recurrent and Transformer-based architectures on the same brain-to-text task and report token-level WER and training curves.

---

## 1. Repository Contents

This GitHub repo contains:

- All **Python source files**, for example:
  - `models.py`, `dataset.py`
  - `train_gru_tokenid.py`, `train_transformer_tokenid.py`
  - `train_gru_tokenid_with_logging.py`, `train_transformer_tokenid_with_logging.py` 
  - `eval_*.py` (evaluation & inference scripts)
  - `create_tokenid_pairs*.py` (preprocessing scripts)
  - `compare_gru_vs_transformer_figures.py` (model comparison & WER figures)
  - `plot_gru_vs_transformer_loss.py` (training loss plots)
- **CSV training logs** (if logging was enabled), e.g.:
  - `logs/gru_tokenid_train.csv`
  - `logs/transformer_tokenid_train.csv`

> ⚠️ The **full dataset, large `.pkl` files, and model checkpoints are *not* stored in this repo** because they are too large for GitHub.  
> They are provided separately via Google Drive (see below).

https://drive.google.com/drive/folders/1EEfoQutCI6oKR-VHY0cCOTEyk9xzwYc8

## 2. Data and Checkpoints (Google Drive)

All large data files and preprocessed datasets are stored in a Google Drive folder:

👉 **Download link (dataset + `.pkl` files):**  
`(https://drive.google.com/drive/folders/1EEfoQutCI6oKR-VHY0cCOTEyk9xzwYc8)`

The Drive folder contains, for example:

- Raw / intermediate data (zipped), e.g.:
  - `dataset_raw.zip`
  - `neural_features.zip`
- Preprocessed Python pickles:
  - `day_pairs.pkl`
  - `day_pairs_small.pkl`
  - `day_pairs_tokenids.pkl`
  - `day_pairs_tokenids_small.pkl`
  - `day_pairs_tokenids_bos.pkl`
  - `day_pairs_tokenids_bos_small.pkl`
- Token mapping file:
  - `tokenid_map.json`
- (Optional) Trained model checkpoints:
  - `gru_tokenid_checkpoint.pth`
  - `transformer_tokenid_checkpoint.pth`

### 2.1. How to place the data locally

1. **Download** the zip(s) from the Drive link.
2. **Unzip** them into the root of this project so the structure looks like:

   ```text
   project-root/
     models.py
     dataset.py
     train_gru_tokenid.py
     train_transformer_tokenid.py
     train_gru_tokenid_with_logging.py
     train_transformer_tokenid_with_logging.py
     create_tokenid_pairs_with_bos.py
     eval_*.py
     compare_gru_vs_transformer_figures.py
     plot_gru_vs_transformer_loss.py
     day_pairs.pkl
     day_pairs_small.pkl
     day_pairs_tokenids.pkl
     day_pairs_tokenids_small.pkl
     day_pairs_tokenids_bos.pkl
     day_pairs_tokenids_bos_small.pkl
     tokenid_map.json
     logs/
       gru_tokenid_train.csv
       transformer_tokenid_train.csv


## 3. Environment Setup

Create and activate a Python environment (Python ≥ 3.8):

```python -m venv venv```

# macOS / Linux:
```source venv/bin/activate```

# Windows:
```venv\Scripts\activate```

Install dependencies:

```pip install -r requirements.txt```

If there is no requirements.txt, the main packages are:

```pip install torch matplotlib jiwer numpy```

## 4. Basic Workflow
4.1. (Optional) Create token-ID pairs with BOS/EOS

If you want to regenerate token-ID pairs from day_pairs.pkl:

```python create_tokenid_pairs_with_bos.py```

This will create:

day_pairs_tokenids_bos.pkl

day_pairs_tokenids_bos_small.pkl

tokenid_map.json

4.2. Train GRU model (token-ID with BOS/EOS)

Basic training:

```python train_gru_tokenid.py```


With CSV logging (if using the logging script):

```python train_gru_tokenid_with_logging.py```


This produces, for example:

gru_tokenid_checkpoint.pth (or gru_tokenid_checkpoint_logged.pth)

logs/gru_tokenid_train.csv (if using the logging script)

4.3. Train Transformer model (token-ID with BOS/EOS)

Basic training:

```python train_transformer_tokenid.py```


With CSV logging:

```python train_transformer_tokenid_with_logging.py```

This produces:

transformer_tokenid_checkpoint.pth

logs/transformer_tokenid_train.csv

 4.4. Evaluate and compare models  
Token-level WER and comparison figures (GRU vs Transformer)

bash
python compare_gru_vs_transformer_figures.py \
  --pairs day_pairs_tokenids_bos_small.pkl \
  --token_map tokenid_map.json \
  --gru_checkpoint gru_tokenid_checkpoint.pth \
  --trf_checkpoint transformer_tokenid_checkpoint.pth \
  --n_eval 200

This script generates:

gru_vs_transformer_wer_bar.png
(bar chart of overall token-level WER)

gru_vs_transformer_wer_scatter.png
(per-sentence WER scatter: GRU vs Transformer)

gru_vs_transformer_delta_hist.png
(histogram of WER(GRU) − WER(Transformer), positive = Transformer better)

Training loss curves (GRU vs Transformer)

python plot_gru_vs_transformer_loss.py \
--gru_csv logs/gru_tokenid_train.csv \
--trf_csv logs/transformer_tokenid_train.csv \
--out training_loss_gru_vs_trf.png

This script plots GRU and Transformer training loss per epoch on the same figure.

## 5. Large Files Policy

To keep this GitHub repository within size limits:

Not committed to GitHub (too large):

.pkl datasets (intermediate and final)

model checkpoints (*.pth)

These files are instead distributed via Google Drive (link in Section 2).

To reproduce results:

Download the data/checkpoints from Drive.

Place them in the project root as described in Section 2.1.

Run the training / evaluation scripts described in Section 4.

## 6. Reproducing Key Results

To reproduce the main results shown in the report/slides:

Ensure all required data files and tokenid_map.json are present in the project root.

Train or download the following checkpoints:

gru_tokenid_checkpoint.pth

transformer_tokenid_checkpoint.pth

Run:

compare_gru_vs_transformer_figures.py to generate WER comparison plots.

plot_gru_vs_transformer_loss.py to generate the training loss comparison plot.

Insert the generated PNGs into the presentation:

Overall WER (bar chart)

Per-sentence WER (scatter)

WER improvement histogram

Training loss curves (GRU vs Transformer)

## 7. Why Transformers Are Better (Short Explanation)

Transformers can model long-range temporal dependencies in neural signals more effectively than GRUs.

In our experiments, the Transformer:

Achieves lower training loss on the same dataset.

Achieves lower token-level WER across most sentences.

Shows a positive shift in WER(GRU) − WER(Transformer), meaning it usually makes fewer decoding errors.

These results motivate using a Transformer architecture for AI-powered BCI speech and communication restoration.

 
   
