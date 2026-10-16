#!/usr/bin/env python3
"""
Evaluate batch inference against the human PAI scores
-----------------------------------------------------
Reads the predictions.csv written by PAI_batch_inference.py for ResNet50,
EfficientNet-B3 and ConvNeXt-Tiny, builds an ensemble, and compares every
rater with the human scores (PAI_true).

Primary measure: Quadratic Weighted Kappa (QWK), with a bootstrap 95% CI.
Also reported: accuracy, MAE, agreement within one PAI step, per-class
sensitivity/specificity, a binary healthy (PAI 1-2) vs pathological (PAI 3-5)
sensitivity/specificity, confusion matrices, and the paired QWK difference
between the ensemble and each single model.

Ensemble:
  * median of the three predicted PAI scores (majority vote with an ordinal
    tie-break) - always computed;
  * mean of the class probabilities - only if predictions.csv contains the
    prob_PAI1..prob_PAI5 columns (run without --predictions-only).

Needs only numpy and pandas (no scikit-learn).

Usage:
    python statistics/evaluate_batch_inference.py inference_results/batch_YYYYMMDD_HHMMSS

Output is written to <results_dir>/evaluation/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {
    "resnet50": "ResNet50",
    "efficientnet_b3": "EfficientNet-B3",
    "convnext_tiny": "ConvNeXt-Tiny",
}
NUM_CLASSES = 5
PATHOLOGICAL_FROM = 3          # PAI 3-5 = pathological (see README)
PROB_COLS = [f"prob_PAI{i}" for i in range(1, NUM_CLASSES + 1)]


# ============================================================================
# Metrics
# ============================================================================

def confusion(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """5x5 confusion matrix, rows = human (true), columns = predicted."""
    return np.bincount((truth - 1) * NUM_CLASSES + (pred - 1),
                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def qwk_from_cm(cm: np.ndarray) -> float:
    n = cm.sum()
    if n == 0:
        return float("nan")
    idx = np.arange(NUM_CLASSES)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (NUM_CLASSES - 1) ** 2
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (weights * expected).sum()
    if denom == 0:
        return float("nan")
    return float(1.0 - (weights * cm).sum() / denom)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def per_class_table(cm: np.ndarray, rater: str) -> pd.DataFrame:
    rows = []
    total = cm.sum()
    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        rows.append({
            "rater": rater, "PAI": i + 1, "n_human": int(cm[i, :].sum()),
            "n_predicted": int(cm[:, i].sum()),
            "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        })
    return pd.DataFrame(rows)


def summarize(rater: str, truth: np.ndarray, pred: np.ndarray) -> dict:
    cm = confusion(truth, pred)
    n = int(cm.sum())
    diff = np.abs(truth - pred)

    pos_true = truth >= PATHOLOGICAL_FROM
    pos_pred = pred >= PATHOLOGICAL_FROM
    tp = int((pos_true & pos_pred).sum()); fn = int((pos_true & ~pos_pred).sum())
    tn = int((~pos_true & ~pos_pred).sum()); fp = int((~pos_true & pos_pred).sum())
    sens_lo, sens_hi = wilson(tp, tp + fn)
    spec_lo, spec_hi = wilson(tn, tn + fp)

    return {
        "rater": rater, "n": n,
        "qwk": qwk_from_cm(cm),
        "accuracy": float((diff == 0).mean()),
        "mae": float(diff.mean()),
        "within_1": float((diff <= 1).mean()),
        "binary_sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "binary_sens_ci_low": sens_lo, "binary_sens_ci_high": sens_hi,
        "binary_specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "binary_spec_ci_low": spec_lo, "binary_spec_ci_high": spec_hi,
    }


def bootstrap_qwk(truth: np.ndarray, raters: dict, n_boot: int, seed: int) -> dict:
    """Image-level bootstrap; the same resamples are used for all raters (paired)."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    out = {name: np.empty(n_boot) for name in raters}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        t = truth[idx]
        for name, pred in raters.items():
            out[name][b] = qwk_from_cm(confusion(t, pred[idx]))
    return out


# ============================================================================
# Loading
# ============================================================================

def load_predictions(results_dir: Path) -> dict:
    frames = {}
    for key in MODELS:
        path = results_dir / key / "predictions.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found - {MODELS[key]} skipped")
            continue
        df = pd.read_csv(path)
        missing = {"filename", "PAI_true", "PAI_pred"} - set(df.columns)
        if missing:
            sys.exit(f"ERROR: {path} lacks columns {sorted(missing)}")
        if df["filename"].duplicated().any():
            sys.exit(f"ERROR: duplicate filenames in {path}")
        frames[key] = df.set_index("filename")
    return frames


def align(frames: dict):
    """Keep only images present for every model and check the human score agrees."""
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    for key, df in frames.items():
        dropped = len(df) - len(common)
        if dropped:
            print(f"  NOTE: {MODELS[key]}: {dropped} image(s) not present for all models were dropped")
    first = next(iter(frames.values()))
    truth = first.loc[common, "PAI_true"].to_numpy()
    for key, df in frames.items():
        if not np.array_equal(df.loc[common, "PAI_true"].to_numpy(), truth):
            sys.exit(f"ERROR: PAI_true differs between models ({key}) - files are not from the same data.csv")
    return common, truth


# ============================================================================
# Main
# ============================================================================

def fmt_ci(est, lo, hi, digits=3):
    return f"{est:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def main():
    parser = argparse.ArgumentParser(description="Compare model and ensemble PAI predictions with human PAI scores.")
    parser.add_argument("results_dir", help="Folder containing <model>/predictions.csv (e.g. inference_results/batch_...)")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples (default 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the bootstrap (default 42)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f"ERROR: not a folder: {results_dir}")

    print(f"Reading {results_dir}")
    frames = load_predictions(results_dir)
    if not frames:
        sys.exit("ERROR: no predictions.csv found")
    names, truth = align(frames)
    truth = truth.astype(int)
    if truth.min() < 1 or truth.max() > NUM_CLASSES:
        sys.exit("ERROR: PAI_true outside 1-5")

    raters = {}
    for key, df in frames.items():
        pred = df.loc[names, "PAI_pred"].to_numpy().astype(int)
        if pred.min() < 1 or pred.max() > NUM_CLASSES:
            sys.exit(f"ERROR: PAI_pred outside 1-5 for {key}")
        raters[MODELS[key]] = pred

    ensemble_names = []
    if len(frames) >= 3:
        stack = np.vstack(list(raters.values()))
        raters["Ensemble (median of models)"] = np.median(stack, axis=0).astype(int)
        ensemble_names.append("Ensemble (median of models)")
        if all(set(PROB_COLS) <= set(df.columns) for df in frames.values()):
            probs = np.mean([df.loc[names, PROB_COLS].to_numpy() for df in frames.values()], axis=0)
            raters["Ensemble (mean probability)"] = probs.argmax(axis=1) + 1
            ensemble_names.append("Ensemble (mean probability)")
    else:
        print("  NOTE: fewer than 3 models - no ensemble")

    # ---- statistics ----
    summary = pd.DataFrame([summarize(name, truth, pred) for name, pred in raters.items()])
    boots = bootstrap_qwk(truth, raters, args.n_boot, args.seed)
    summary["qwk_ci_low"] = [np.nanpercentile(boots[r], 2.5) for r in summary["rater"]]
    summary["qwk_ci_high"] = [np.nanpercentile(boots[r], 97.5) for r in summary["rater"]]

    per_class = pd.concat([per_class_table(confusion(truth, p), name) for name, p in raters.items()],
                          ignore_index=True)

    diffs = []
    for ens in ensemble_names:
        for key in frames:
            model = MODELS[key]
            d = boots[ens] - boots[model]
            diffs.append({
                "comparison": f"{ens} - {model}",
                "qwk_difference": float(summary.set_index("rater").loc[ens, "qwk"] - summary.set_index("rater").loc[model, "qwk"]),
                "ci_low": float(np.nanpercentile(d, 2.5)), "ci_high": float(np.nanpercentile(d, 97.5)),
            })
    diffs = pd.DataFrame(diffs)

    # ---- outputs ----
    out_dir = results_dir / "evaluation"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.4f")
    per_class.to_csv(out_dir / "per_class.csv", index=False, float_format="%.4f")
    if len(diffs):
        diffs.to_csv(out_dir / "qwk_differences.csv", index=False, float_format="%.4f")

    line = "=" * 78
    out = [line, "  PAI batch inference vs human scores", line,
           f"Results folder : {results_dir}",
           f"Images         : {len(truth)}",
           "Human PAI      : " + ", ".join(f"PAI {k}: {int((truth == k).sum())}" for k in range(1, NUM_CLASSES + 1)),
           f"Bootstrap      : {args.n_boot} image-level resamples, seed {args.seed}",
           "", "Quadratic weighted kappa (QWK) vs human, 95% bootstrap CI",
           "-" * 78]
    for _, r in summary.sort_values("qwk", ascending=False).iterrows():
        out.append(f"  {r['rater']:<30} {fmt_ci(r['qwk'], r['qwk_ci_low'], r['qwk_ci_high'])}")
    if len(diffs):
        out += ["", "Paired QWK difference (ensemble minus single model), 95% bootstrap CI", "-" * 78]
        for _, r in diffs.iterrows():
            out.append(f"  {r['comparison']:<52} {fmt_ci(r['qwk_difference'], r['ci_low'], r['ci_high'], 4)}")

    out += ["", "Other agreement measures", "-" * 78,
            f"  {'Rater':<30} {'Accuracy':>9} {'MAE':>7} {'Within 1':>9}"]
    for _, r in summary.iterrows():
        out.append(f"  {r['rater']:<30} {r['accuracy']:>9.4f} {r['mae']:>7.4f} {r['within_1']:>9.4f}")

    out += ["", f"Binary: healthy (PAI 1-2) vs pathological (PAI {PATHOLOGICAL_FROM}-5), Wilson 95% CI", "-" * 78]
    for _, r in summary.iterrows():
        out.append(f"  {r['rater']:<30} sens {fmt_ci(r['binary_sensitivity'], r['binary_sens_ci_low'], r['binary_sens_ci_high'])}"
                   f"   spec {fmt_ci(r['binary_specificity'], r['binary_spec_ci_low'], r['binary_spec_ci_high'])}")

    out += ["", "Per-class sensitivity / specificity (one-vs-rest)", "-" * 78,
            f"  {'Rater':<30} " + " ".join(f"{'PAI ' + str(k):>13}" for k in range(1, NUM_CLASSES + 1))]
    for name in raters:
        sub = per_class[per_class["rater"] == name]
        out.append(f"  {name:<30} " + " ".join(f"{s:>6.3f}/{p:<6.3f}" for s, p in zip(sub["sensitivity"], sub["specificity"])))
    out.append("  (each cell: sensitivity/specificity)")

    out += ["", "Confusion matrices (rows = human, columns = predicted)", "-" * 78]
    for name, pred in raters.items():
        cm = confusion(truth, pred)
        out += ["", f"  {name}", "          " + "".join(f"{'P' + str(k):>8}" for k in range(1, NUM_CLASSES + 1))]
        for i in range(NUM_CLASSES):
            out.append(f"    Human {i + 1}" + "".join(f"{v:>8d}" for v in cm[i]))
    out.append(line)

    text = "\n".join(out)
    (out_dir / "evaluation_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
