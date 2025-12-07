# plot_gru_vs_transformer_loss.py
"""
Plot GRU vs Transformer training loss on the same figure.

Usage:
  python plot_gru_vs_transformer_loss.py \
      --gru_csv logs/gru_tokenid_train.csv \
      --trf_csv logs/transformer_tokenid_train.csv \
      --out training_loss_gru_vs_trf.png
"""

import argparse
import csv
import matplotlib.pyplot as plt


def load_loss_csv(path):
    epochs = []
    losses = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
    return epochs, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gru_csv", type=str, required=True,
                        help="CSV log for GRU (epoch,train_loss).")
    parser.add_argument("--trf_csv", type=str, required=True,
                        help="CSV log for Transformer (epoch,train_loss).")
    parser.add_argument("--out", type=str, default="training_loss_gru_vs_trf.png",
                        help="Output PNG filename.")
    parser.add_argument("--title", type=str, default="Training Loss over Epochs")
    args = parser.parse_args()

    gru_epochs, gru_loss = load_loss_csv(args.gru_csv)
    trf_epochs, trf_loss = load_loss_csv(args.trf_csv)

    plt.figure()
    plt.plot(gru_epochs, gru_loss, marker="o", label="GRU")
    plt.plot(trf_epochs, trf_loss, marker="o", label="Transformer")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(args.title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved combined training curve to", args.out)


if __name__ == "__main__":
    main()
