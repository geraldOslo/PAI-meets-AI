#!/usr/bin/env python3
"""
Phase 3: PAI 2/3 Diagnostic Boundary Optimization (Top 10 Experiments)
Focus on the clinically critical healthy/diseased threshold

**Clinical Priority:** PAI-2 vs PAI-3 is the diagnostic boundary between healthy
and diseased. This is more important than overall QWK.

**Current Performance (Phase 1 & 2):**
- Best balanced: PAI-2=0.57, PAI-3=0.57 (ResNet50 exp1_baseline)
- Best PAI-2: 0.65 (EfficientNet exp3)
- Best PAI-3: 0.68 (ConvNeXt exp2)
- Problem: ~20-30% of PAI-2/3 cases confused with each other

**Phase 3 Goals:**
- PAI-2 Sensitivity: ≥0.70 (current best: 0.65)
- PAI-3 Sensitivity: ≥0.70 (current best: 0.68)
- Overall QWK: Maintain ≥0.78

**Strategy (10 experiments):**
1-3. Boundary-aware training (higher focal gamma, extended training)
4-5. Label noise robustness (smoothing, mixup)
6. Combined approach (smoothing + mixup)
7. EfficientNet optimized
8-10. Statistical validation (3 replicates)

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import os
import argparse
import subprocess
from pathlib import Path

# ==============================================================================
# PHASE 3: TOP 10 EXPERIMENTS FOR PAI 2/3 BOUNDARY
# ==============================================================================

SEARCH_SPACE_PHASE3 = {
    # =========================================================================
    # BOUNDARY-AWARE TRAINING (3 experiments)
    # Strategy: Focus model attention on hard PAI 2/3 cases
    # =========================================================================

    "p3_boundary_high_gamma": {
        "description": "High focal gamma (3.5) to focus on hard 2/3 boundary cases",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 3.5,  # Higher than Phase 1 best (2.0)
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "p3_boundary_extended": {
        "description": "Extended training (120 epochs) for fine-grained boundary learning",
        "models": ["resnet50"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.5,  # Between Phase 1 best (2.0) and Phase 2 experiments (3.0)
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "p3_boundary_aggressive": {
        "description": "Very high gamma (4.0) for maximum boundary focus",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 4.0,  # Aggressive focus on hard examples
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    # =========================================================================
    # LABEL NOISE ROBUSTNESS (3 experiments)
    # Strategy: Handle annotation ambiguity at PAI 2/3 boundary
    # =========================================================================

    "p3_noise_label_smoothing": {
        "description": "Label smoothing (0.10) for annotation noise robustness",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 2.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.10,  # Handle label noise
            "--use_oversampling": ""
        }
    },

    "p3_noise_mixup": {
        "description": "Mixup (0.2) for noise + PAI-2 boost (worked in Phase 1)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 2.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": "",
            "--mixup": "",
            "--mixup_alpha": 0.2
        }
    },

    "p3_noise_combined": {
        "description": "Combined smoothing (0.05) + mixup (0.15) for maximum robustness",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 2.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.05,
            "--use_oversampling": "",
            "--mixup": "",
            "--mixup_alpha": 0.15
        }
    },

    # =========================================================================
    # OPTIMIZED VARIANTS (1 experiment)
    # Strategy: Best from Phase 2 EfficientNet + improvements
    # =========================================================================

    "p3_effnet_optimized": {
        "description": "EfficientNet: Phase 2 best + gamma 3.5 + label smoothing",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 3.5,
            "--max_lr": 0.001,  # Phase 2 learning: lower LR worked better
            "--dropout": 0.5,
            "--label_smoothing": 0.05,  # Combat EfficientNet overfitting
            "--use_oversampling": ""
        }
    },

    # =========================================================================
    # STATISTICAL VALIDATION (3 experiments)
    # Strategy: Replicate most promising config for mean ± std reporting
    # =========================================================================

    "p3_validation_run1": {
        "description": "VALIDATION RUN 1: Optimized baseline for publication",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 2.5,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "p3_validation_run2": {
        "description": "VALIDATION RUN 2: Replicate for statistical confidence",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 2.5,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "p3_validation_run3": {
        "description": "VALIDATION RUN 3: Replicate for statistical confidence",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 2.5,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },
}

# --- Slurm Script Template ---
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=ec192
#SBATCH --partition=accel
#SBATCH --gres=gpu:rtx30:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time={time_str}
#SBATCH --output={log_path}
#SBATCH --error={err_path}

# ============================================================================
# PAI Phase 3: PAI 2/3 Boundary Optimization
# ============================================================================

echo "=================================================="
echo "Starting Phase 3 Training Job: {job_name}"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# --- Load modules ---
echo "Loading modules..."
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

# --- Environment variables ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# --- Activate virtual environment ---
VENV_PATH="/fp/projects01/ec192/python_envs/cnn_env"
echo "Activating virtual environment at $VENV_PATH..."
source $VENV_PATH/bin/activate

# --- Navigate to project root ---
PROJECT_ROOT="/fp/projects01/ec192/Github/PAI-meets-AI-2"
echo "Navigating to project root: $PROJECT_ROOT"
cd $PROJECT_ROOT

# --- Job Execution ---
echo ""
echo "Starting Phase 3 training..."
echo "Model: {model}"
echo "Experiment: {experiment}"
echo "Parameters: {params}"
echo ""

python code/training/train_simple.py \\
    --models {model} \\
    --experiment_name {experiment} \\
    {params}

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job Completed: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=================================================="

exit $EXIT_CODE
"""

# Time allocation based on model and epochs
TIME_ALLOCATIONS = {
    ("resnet50", 100): "02:00:00",
    ("resnet50", 120): "02:30:00",
    ("efficientnet_b3", 100): "02:30:00",
}

def get_time_allocation(model: str, epochs: int) -> str:
    """Estimate time allocation based on model and epochs."""
    key = (model, epochs)
    if key in TIME_ALLOCATIONS:
        return TIME_ALLOCATIONS[key]

    # Fallback calculation
    base_times = {
        "resnet50": 72,  # minutes per 100 epochs
        "efficientnet_b3": 90,
    }

    base_min = base_times.get(model, 80)
    total_min = int(base_min * (epochs / 100) * 1.2)
    hours = total_min // 60
    minutes = total_min % 60
    return f"{hours:02d}:{minutes:02d}:00"


def format_params(params: dict) -> str:
    """Format parameter dictionary into command line arguments."""
    parts = []
    for key, value in params.items():
        if value == "":  # Boolean flag
            parts.append(key)
        else:
            parts.append(f"{key} {value}")
    return " \\\n    ".join(parts)


def generate_slurm_script(model: str, experiment: str, exp_config: dict, output_dir: Path):
    """Generate a Slurm script for a single experiment."""
    params = exp_config["params"]
    epochs = params.get("--epochs", 100)

    job_name = f"p3_{model}_{experiment}"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_dir / f"autogen_phase3_{model}_{experiment}.sh"
    log_path = log_dir / f"{job_name}.out"
    err_path = log_dir / f"{job_name}.err"

    time_str = get_time_allocation(model, epochs)
    params_str = format_params(params)

    script_content = SLURM_TEMPLATE.format(
        job_name=job_name,
        time_str=time_str,
        log_path=log_path,
        err_path=err_path,
        model=model,
        experiment=experiment,
        params=params_str
    )

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)

    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: PAI 2/3 Boundary Optimization (Top 10 Experiments)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "efficientnet_b3"],
        default=["resnet50", "efficientnet_b3"],
        help="Models to optimize (default: both)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Automatically submit generated Slurm scripts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts but don't submit"
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    slurm_dir = script_dir / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)

    print("="*80)
    print("PHASE 3: PAI 2/3 DIAGNOSTIC BOUNDARY OPTIMIZATION")
    print("="*80)
    print()
    print("CLINICAL PRIORITY: Distinguish PAI-2 (healthy) from PAI-3 (diseased)")
    print()
    print("CURRENT PERFORMANCE:")
    print("  - Best balanced: PAI-2=0.57, PAI-3=0.57 (ResNet50 Phase 1)")
    print("  - Best PAI-2: 0.65, Best PAI-3: 0.68")
    print("  - Problem: ~20-30% confusion between PAI-2 and PAI-3")
    print()
    print("PHASE 3 TARGETS:")
    print("  - PAI-2 Sensitivity: ≥0.70")
    print("  - PAI-3 Sensitivity: ≥0.70")
    print("  - Overall QWK: Maintain ≥0.78")
    print()
    print(f"Generating scripts in: {slurm_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Total experiments: {len(SEARCH_SPACE_PHASE3)}")
    print()
    print("Experiment Groups:")
    print("  1. Boundary-Aware (3 exp): High gamma, extended training")
    print("  2. Noise Robustness (3 exp): Label smoothing, mixup")
    print("  3. EfficientNet Optimized (1 exp)")
    print("  4. Statistical Validation (3 exp): For publication mean±std")
    print()

    generated_scripts = []

    for exp_name, exp_config in SEARCH_SPACE_PHASE3.items():
        print(f"\n{exp_name}:")
        print(f"  {exp_config['description']}")

        exp_models = exp_config.get("models", args.models)
        exp_models = [m for m in exp_models if m in args.models]

        for model in exp_models:
            script_path = generate_slurm_script(model, exp_name, exp_config, slurm_dir)
            generated_scripts.append(script_path)
            print(f"  ✓ Generated: {script_path.name}")

    print()
    print("="*80)
    print(f"Generated {len(generated_scripts)} Slurm scripts")
    print("="*80)

    if args.submit and not args.dry_run:
        print("\nSubmitting jobs...")
        for script in generated_scripts:
            try:
                result = subprocess.run(
                    ["sbatch", str(script)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"  ✓ Submitted: {script.name} - {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to submit {script.name}: {e.stderr}")

        print("\nAll jobs submitted!")
        print("Monitor with: squeue -u $USER")
    else:
        print("\nTo submit these jobs:")
        print("  python code/hyperparam_search_phase3.py --submit")

    print()
    print("="*80)
    print("AFTER COMPLETION - ANALYSIS STEPS")
    print("="*80)
    print()
    print("1. Run summary to collect all results:")
    print("   python code/summarize_and_infer.py \\")
    print("       --exp_dir /fp/projects01/ec192/Github/PAI-meets-AI-2/experiments \\")
    print("       --print-table")
    print()
    print("2. Calculate statistics from validation runs (p3_validation_run[1-3]):")
    print("   - Extract PAI-2 and PAI-3 sensitivities")
    print("   - Calculate mean ± std for publication")
    print()
    print("3. Compare with Phase 1 & 2 baselines:")
    print("   - ResNet50 exp1_baseline: PAI-2=0.57, PAI-3=0.57, QWK=0.7951")
    print("   - Target improvement: Both PAI-2 and PAI-3 ≥0.70")
    print()
    print("4. Key metrics to report:")
    print("   - PAI-2 Sensitivity (target: ≥0.70)")
    print("   - PAI-3 Sensitivity (target: ≥0.70)")
    print("   - PAI 2↔3 Confusion Rate (target: <20%)")
    print("   - Overall QWK (maintain: ≥0.78)")
    print()
    print("EXPERIMENT NAMING:")
    print("  - Phase 3 experiments will be saved as:")
    print("    experiments/p3_*_YYYYMMDD_HHMMSS/")
    print("  - Compatible with summarize_and_infer.py (same as Phase 1 & 2)")
    print()


if __name__ == "__main__":
    main()
