#!/usr/bin/env python3
"""
PAI Batch Inference Script — All Three Models
----------------------------------------------
Runs all three trained PAI models (EfficientNet-B3, ResNet50, ConvNeXt-Tiny)
on an external dataset folder containing pre-cropped image clips and a
data.csv label file. No GUI, no config.py dependency.

Edit the constants in the CONFIGURATION section below before running.

Expected dataset layout:
    <DATA_DIR>/
        data.csv          # columns: filename (or image_path), PAI (1-5)
        image1.jpg
        image2.tiff
        ...

Output (written to OUTPUT_DIR/<model_name>/):
    predictions.csv       # per-image predictions and class probabilities
    metrics_report.txt    # accuracy, QWK, confusion matrix, per-class metrics

A combined summary across all models is written to OUTPUT_DIR/summary.csv.

Usage:
    python PAI_batch_inference.py
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# ============================================================================
# CONFIGURATION — edit these before running
# ============================================================================

DATA_DIR   = r"C:\path\to\your\dataset"   # folder with data.csv + image clips
OUTPUT_DIR = r"C:\path\to\output"         # results root; per-model subdirs created here

# One checkpoint path per architecture. Set a path to None to skip that model.
CHECKPOINTS = {
    "efficientnet_b3": r"C:\path\to\efficientnet_b3_best.pth",
    "resnet50":        r"C:\path\to\resnet50_best.pth",
    "convnext_tiny":   r"C:\path\to\convnext_tiny_best.pth",
}

# Default normalization — overridden automatically by values stored in the checkpoint.
DEFAULT_MEAN = [0.3784975, 0.3784975, 0.3784975]
DEFAULT_STD  = [0.16739018, 0.16739018, 0.16739018]

# ============================================================================
# Device setup
# ============================================================================

def setup_device():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            print("CUDA not available — using CPU")
    except Exception:
        device = torch.device("cpu")
        print("Defaulting to CPU")
    return device


# ============================================================================
# Model loading (no config.py dependency)
# ============================================================================

def get_input_size(model_name: str) -> int:
    return 300 if "efficientnet" in model_name.lower() else 224


def infer_config_from_filename(checkpoint_path: str) -> dict:
    """Fall back to filename heuristics for old-format checkpoints."""
    fn = os.path.basename(checkpoint_path).lower()
    if "efficientnet" in fn or "effnet" in fn:
        return {"model_name": "efficientnet_b3", "dropout_rate": 0.5, "drop_path_rate": 0.2, "num_classes": 5}
    if "resnet" in fn:
        return {"model_name": "resnet50",        "dropout_rate": 0.5, "drop_path_rate": 0.1, "num_classes": 5}
    if "convnext" in fn:
        return {"model_name": "convnext_tiny",   "dropout_rate": 0.5, "drop_path_rate": 0.2, "num_classes": 5}
    return None


def load_model(checkpoint_path: str, device):
    """
    Load a model checkpoint. Reads architecture and normalization from the
    checkpoint metadata when available (new format); falls back to filename
    heuristics for old-format checkpoints.

    Returns: (model, mean, std, input_size, model_name)
    """
    print(f"  Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_config = checkpoint.get("model_config", {})
    if model_config:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    else:
        print("  Old-format checkpoint — inferring architecture from filename")
        model_config = infer_config_from_filename(checkpoint_path)
        if not model_config:
            raise ValueError(
                f"Cannot infer architecture from '{os.path.basename(checkpoint_path)}'. "
                "Filename must contain 'resnet', 'efficientnet', or 'convnext'."
            )
        state_dict = checkpoint  # old format: checkpoint IS the state_dict

    model_name     = model_config["model_name"]
    num_classes    = model_config.get("num_classes", 5)
    dropout_rate   = model_config.get("dropout_rate", 0.0)
    drop_path_rate = model_config.get("drop_path_rate", 0.0)

    print(f"  Architecture: {model_name}  |  classes: {num_classes}")

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
    )

    # Strip DataParallel prefix if present
    if state_dict and list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    norm = checkpoint.get("normalization", {})
    mean = norm.get("mean", DEFAULT_MEAN)
    std  = norm.get("std",  DEFAULT_STD)
    input_size = get_input_size(model_name)

    print(f"  Input size: {input_size}×{input_size}")
    if "best_metric_val" in checkpoint:
        print(f"  Best val QWK: {checkpoint['best_metric_val']:.4f}")

    return model, mean, std, input_size, model_name


# ============================================================================
# Dataset / CSV loading
# ============================================================================

def load_dataset(data_dir: str) -> pd.DataFrame:
    csv_path = Path(data_dir) / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"data.csv not found in {data_dir}")

    df = pd.read_csv(csv_path)

    if "filename" in df.columns:
        df["filename"] = df["filename"].astype(str)
    elif "image_path" in df.columns:
        df["filename"] = df["image_path"].astype(str)
    else:
        raise ValueError("data.csv must have a 'filename' or 'image_path' column")

    if "PAI" not in df.columns:
        raise ValueError("data.csv must have a 'PAI' column (values 1-5)")

    before = len(df)
    df = df[df["PAI"].astype(str).str.strip().str.upper() != "NS"].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with PAI='NS'")

    df["PAI"] = df["PAI"].astype(int)
    return df


def resolve_image_path(filename: str, data_dir: str):
    p = Path(filename)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = Path(data_dir) / filename
    if candidate.exists():
        return str(candidate)
    candidate2 = Path(data_dir) / p.name   # bare filename as fallback
    if candidate2.exists():
        return str(candidate2)
    return None


# ============================================================================
# Inference
# ============================================================================

def run_inference(model, df: pd.DataFrame, data_dir: str,
                  mean, std, input_size: int, device) -> pd.DataFrame:
    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    records = []
    skipped = 0

    for _, row in df.iterrows():
        img_path = resolve_image_path(row["filename"], data_dir)
        if img_path is None:
            print(f"    WARNING: not found — {row['filename']}")
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"    WARNING: cannot open {img_path}: {e}")
            skipped += 1
            continue

        tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs  = F.softmax(output[0], dim=0).cpu().numpy()

        pred_class = int(np.argmax(probs)) + 1  # 1-indexed PAI

        records.append({
            "filename": row["filename"],
            "PAI_true": row["PAI"],
            "PAI_pred": pred_class,
            "correct":  int(pred_class == row["PAI"]),
            "prob_PAI1": round(float(probs[0]), 4),
            "prob_PAI2": round(float(probs[1]), 4),
            "prob_PAI3": round(float(probs[2]), 4),
            "prob_PAI4": round(float(probs[3]), 4),
            "prob_PAI5": round(float(probs[4]), 4),
        })

    if skipped:
        print(f"    {skipped} image(s) skipped")

    return pd.DataFrame(records)


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(results: pd.DataFrame, model_name: str) -> dict:
    from sklearn.metrics import (
        confusion_matrix, cohen_kappa_score,
        accuracy_score, classification_report
    )

    y_true = results["PAI_true"].values
    y_pred = results["PAI_pred"].values
    labels = [1, 2, 3, 4, 5]

    acc  = accuracy_score(y_true, y_pred)
    qwk  = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm   = confusion_matrix(y_true, y_pred, labels=labels)
    rpt  = classification_report(
        y_true, y_pred, labels=labels,
        target_names=[f"PAI {i}" for i in labels],
        zero_division=0,
    )

    # Per-class specificity
    specificities = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    lines = [
        "=" * 62,
        f"  PAI Inference Report — {model_name}",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  N images:  {len(results)}",
        "=" * 62,
        f"Accuracy:                 {acc:.4f}",
        f"Quadratic Weighted Kappa: {qwk:.4f}",
        "",
        "Per-class specificity:",
    ]
    for i, sp in enumerate(specificities):
        lines.append(f"  PAI {i+1}: {sp:.4f}")
    lines += [
        "",
        "Classification report (sensitivity = recall):",
        rpt,
        "Confusion matrix (rows=true, cols=pred, PAI 1-5):",
        "       P1    P2    P3    P4    P5",
    ]
    for i, row in enumerate(cm):
        lines.append(f"  T{i+1}  " + "  ".join(f"{v:4d}" for v in row))
    lines.append("=" * 62)

    return {"acc": acc, "qwk": qwk, "report_text": "\n".join(lines)}


# ============================================================================
# Main
# ============================================================================

def main():
    # Validate DATA_DIR
    if not Path(DATA_DIR).exists():
        print(f"ERROR: DATA_DIR does not exist: {DATA_DIR}")
        sys.exit(1)

    # Filter to checkpoints that are actually set and exist
    active = {}
    for arch, ckpt in CHECKPOINTS.items():
        if ckpt is None:
            print(f"  Skipping {arch} (checkpoint set to None)")
            continue
        if not Path(ckpt).exists():
            print(f"  WARNING: checkpoint not found for {arch}: {ckpt}")
            continue
        active[arch] = ckpt

    if not active:
        print("ERROR: No valid checkpoints found. Edit CHECKPOINTS in the script.")
        sys.exit(1)

    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    device = setup_device()

    print(f"\nDataset: {DATA_DIR}")
    df = load_dataset(DATA_DIR)
    print(f"Loaded {len(df)} images | PAI distribution: "
          + ", ".join(f"{k}:{v}" for k, v in df["PAI"].value_counts().sort_index().items()))

    summary_rows = []

    for arch, ckpt_path in active.items():
        print(f"\n{'='*62}")
        print(f"  Model: {arch}")
        print(f"{'='*62}")

        try:
            model, mean, std, input_size, model_name = load_model(ckpt_path, device)

            print(f"  Running inference on {len(df)} images...")
            results = run_inference(model, df, DATA_DIR, mean, std, input_size, device)

            if results.empty:
                print(f"  No predictions produced for {arch} — skipping")
                continue

            # Per-model output directory
            model_out = output_root / arch
            model_out.mkdir(parents=True, exist_ok=True)

            pred_path = model_out / "predictions.csv"
            results.to_csv(pred_path, index=False)
            print(f"  Predictions → {pred_path}")

            try:
                metrics = compute_metrics(results, arch)
                print(f"  Accuracy: {metrics['acc']:.4f}  |  QWK: {metrics['qwk']:.4f}")

                report_path = model_out / "metrics_report.txt"
                report_path.write_text(metrics["report_text"])
                print(f"  Report    → {report_path}")

                summary_rows.append({
                    "model":    arch,
                    "n_images": len(results),
                    "accuracy": round(metrics["acc"], 4),
                    "qwk":      round(metrics["qwk"], 4),
                })
            except ImportError:
                print("  sklearn not available — skipping metrics (accuracy only)")
                acc = results["correct"].mean()
                print(f"  Accuracy: {acc:.4f}")
                summary_rows.append({"model": arch, "n_images": len(results),
                                     "accuracy": round(acc, 4), "qwk": None})

            # Free GPU memory before loading next model
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  ERROR processing {arch}: {e}")
            traceback.print_exc()
            continue

    # Combined summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_root / "summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print(f"\n{'='*62}")
        print("  SUMMARY")
        print(f"{'='*62}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
