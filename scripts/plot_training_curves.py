# scripts/plot_training_curves.py
import argparse
import csv
import os

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True,
                        help="CSV log file with epoch, train_loss, optional val_loss, wer.")
    parser.add_argument("--out_prefix", type=str, default="training",
                        help="Prefix for output PNG files.")
    parser.add_argument("--title", type=str, default="Training curves")
    args = parser.parse_args()

    epochs = []
    train_loss = []
    val_loss = []
    wer = []

    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_val = "val_loss" in fieldnames
        has_wer = "wer" in fieldnames or "val_wer" in fieldnames

        wer_key = "wer" if "wer" in fieldnames else ("val_wer" if "val_wer" in fieldnames else None)

        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            if has_val and row.get("val_loss", "") != "":
                val_loss.append(float(row["val_loss"]))
            elif has_val:
                val_loss.append(float("nan"))

            if wer_key is not None and row.get(wer_key, "") != "":
                wer.append(float(row[wer_key]))

    # --- Loss plot ---
    plt.figure()
    plt.plot(epochs, train_loss, label="train_loss")
    if val_loss:
        try:
            # check if any non-NaN
            _ = [v for v in val_loss if v == v]
            plt.plot(epochs, val_loss, label="val_loss")
        except Exception:
            pass
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.title + " - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_png = args.out_prefix + "_loss.png"
    plt.savefig(loss_png)
    print("Saved loss curve to", loss_png)

    # --- WER plot (if available) ---
    if wer:
        plt.figure()
        plt.plot(epochs[:len(wer)], wer, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("WER")
        plt.title(args.title + " - WER")
        plt.grid(True)
        plt.tight_layout()
        wer_png = args.out_prefix + "_wer.png"
        plt.savefig(wer_png)
        print("Saved WER curve to", wer_png)


if __name__ == "__main__":
    main()
