"""Train the 1D CNN modulation classifier.

Examples
--------
# Quick start with the built-in synthetic dataset (no download needed)
python train.py

# Train on the real RadioML 2016.10A dataset
python train.py --radioml data/RML2016.10a_dict.pkl
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow import keras

from src.model import build_model
from src.signals import MODULATIONS, generate_dataset, load_radioml, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radioml", help="path to RML2016.10a_dict.pkl (optional)")
    ap.add_argument("--per-class", type=int, default=200,
                    help="synthetic examples per class per SNR")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.radioml:
        print(f"Loading RadioML from {args.radioml} ...")
        X, y, snr = load_radioml(args.radioml)
    else:
        print("Generating synthetic dataset ...")
        X, y, snr = generate_dataset(per_class_per_snr=args.per_class)
    print(f"Dataset: {X.shape[0]} examples, {len(MODULATIONS)} classes")

    X = normalize(X).transpose(0, 2, 1)  # (N, 128, 2) for Conv1D
    X_tr, X_te, y_tr, y_te, s_tr, s_te = train_test_split(
        X, y, snr, test_size=0.2, stratify=y, random_state=42)

    model = build_model(len(MODULATIONS))
    model.summary()
    model.fit(
        X_tr, y_tr, validation_split=0.1, epochs=args.epochs, batch_size=args.batch,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        ],
    )

    pred = model.predict(X_te, batch_size=1024).argmax(1)
    acc = float((pred == y_te).mean())
    print(f"\nOverall test accuracy: {acc:.3f}")

    # Accuracy vs SNR
    snrs = np.unique(s_te)
    acc_snr = [float((pred[s_te == s] == y_te[s_te == s]).mean()) for s in snrs]
    for s, a in zip(snrs, acc_snr):
        print(f"  SNR {s:+3d} dB : {a:.3f}")
    plt.figure(figsize=(7, 4))
    plt.plot(snrs, acc_snr, marker="o")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy"); plt.ylim(0, 1); plt.grid(alpha=.3)
    plt.title("Classification accuracy vs SNR")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "accuracy_vs_snr.png"), dpi=150)

    # Confusion matrix at high SNR (>= 10 dB)
    hi = s_te >= 10
    cm = confusion_matrix(y_te[hi], pred[hi], labels=range(len(MODULATIONS)), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=MODULATIONS).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False, xticks_rotation=45)
    ax.set_title("Confusion matrix (SNR >= 10 dB)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=150)

    model.save(os.path.join(args.out, "amc_cnn.keras"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"classes": MODULATIONS, "test_accuracy": acc,
                   "accuracy_by_snr": dict(zip(map(int, snrs), acc_snr))}, f, indent=2)
    print(f"\nSaved model and plots to {args.out}/")


if __name__ == "__main__":
    main()
