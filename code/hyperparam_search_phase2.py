#!/usr/bin/env python3
"""
Phase 2: Focused Optimization for Clean Dataset
Deep dive into best-performing architecture with refined hyperparameters

Based on Phase 1 results, this focuses primarily on the winning architecture
(likely EfficientNet-B3) with fine-tuned hyperparameters.

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import os
import argparse
import subprocess
from pathlib import Path

# ==============================================================================
# PHASE 2: FOCUSED OPTIMIZATION - ALL ARCHITECTURES
# Based on Phase 1 Results:
# - EfficientNet-B3: 0.8142 QWK (winner, but overfits: val 0.935 → test 0.814)
# - ResNet50: 0.7959 QWK (solid, needs optimization)
# - ConvNeXt-Tiny: 0.6595 QWK (training instability issues, needs stability fixes)
#
# Strategy: Model-specific optimizations based on Phase 1 learnings
# Total: 18 experiments (6 per architecture)
# ==============================================================================
SEARCH_SPACE_PHASE2 = {
    # =========================================================================
    # EFFICIENTNET-B3: Optimize the winner (reduce overfitting, improve PAI-5)
    # =========================================================================
    "effnet_exp1_baseline_confirm": {
        "description": "EfficientNet: Replicate best Phase 1 (QWK=0.8142)",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "effnet_exp2_focal_gamma_3_5": {
        "description": "EfficientNet: Higher gamma for PAI-4/5 improvement",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.5,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "effnet_exp3_label_smoothing": {
        "description": "EfficientNet: Reduce overfitting with label smoothing",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "effnet_exp4_stronger_reg": {
        "description": "EfficientNet: Combat overfitting with stronger regularization",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--weight_decay": 0.02,
            "--use_oversampling": ""
        }
    },

    "effnet_exp5_lower_lr": {
        "description": "EfficientNet: Lower LR for better generalization",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 90,
            "--patience": 25,
            "--focal_gamma": 3.0,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "effnet_exp6_optimized_combo": {
        "description": "EfficientNet: Best combo (gamma 3.5 + label smoothing)",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 85,
            "--patience": 25,
            "--focal_gamma": 3.5,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--weight_decay": 0.015,
            "--label_smoothing": 0.05,
            "--use_oversampling": ""
        }
    },

    # =========================================================================
    # RESNET50: Boost from 0.7959 → 0.82+ (seemed to prefer gamma=3.0)
    # =========================================================================
    "resnet_exp1_baseline_improved": {
        "description": "ResNet: Best Phase 1 settings with improvements",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "resnet_exp2_focal_gamma_3_5": {
        "description": "ResNet: Higher gamma for hard examples",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 3.5,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "resnet_exp3_label_smoothing": {
        "description": "ResNet: Add label smoothing",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "resnet_exp4_lower_lr": {
        "description": "ResNet: Lower LR for stability",
        "models": ["resnet50"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 3.0,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.5,
            "--use_oversampling": ""
        }
    },

    "resnet_exp5_higher_dropout": {
        "description": "ResNet: Test if more regularization helps",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
            "--dropout": 0.6,
            "--weight_decay": 0.015,
            "--use_oversampling": ""
        }
    },

    "resnet_exp6_optimized_combo": {
        "description": "ResNet: Best combo (lower LR + gamma 3.5 + label smoothing)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 3.5,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.5,
            "--label_smoothing": 0.05,
            "--use_oversampling": ""
        }
    },

    # =========================================================================
    # CONVNEXT-TINY: Fix stability issues (many Phase 1 runs collapsed)
    # Strategy: MUCH lower LR, conservative settings, longer training
    # =========================================================================
    "convnext_exp1_stability_focus": {
        "description": "ConvNeXt: Very low LR for training stability",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.0,  # Lower gamma for stability
            "--base_lr": 0.00005,  # Very low
            "--max_lr": 0.0005,    # Very low
            "--dropout": 0.3,      # Lower dropout
            "--use_oversampling": ""
        }
    },

    "convnext_exp2_moderate_settings": {
        "description": "ConvNeXt: Moderate settings with conservative LR",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.5,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.4,
            "--use_oversampling": ""
        }
    },

    "convnext_exp3_label_smoothing": {
        "description": "ConvNeXt: Label smoothing for stability",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.5,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.4,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "convnext_exp4_higher_gamma": {
        "description": "ConvNeXt: Test gamma=3.0 with stable LR",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 3.0,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.4,
            "--use_oversampling": ""
        }
    },

    "convnext_exp5_stronger_reg": {
        "description": "ConvNeXt: Higher regularization for generalization",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.5,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.5,
            "--weight_decay": 0.02,
            "--use_oversampling": ""
        }
    },

    "convnext_exp6_optimized_combo": {
        "description": "ConvNeXt: Best stable combo if exp1-5 succeed",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 140,
            "--patience": 35,
            "--focal_gamma": 2.5,
            "--base_lr": 0.0001,
            "--max_lr": 0.001,
            "--dropout": 0.4,
            "--label_smoothing": 0.05,
            "--weight_decay": 0.015,
            "--use_oversampling": ""
        }
    }
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
# PAI Phase 2 Focused Optimization - Clean Dataset
# ============================================================================

echo "=================================================="
echo "Starting Phase 2 Training Job: {job_name}"
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
echo "Starting training..."
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
    ("resnet50", 150): "03:00:00",
    ("efficientnet_b3", 75): "02:00:00",
    ("efficientnet_b3", 85): "02:15:00",
    ("efficientnet_b3", 90): "02:30:00",
    ("efficientnet_b3", 100): "02:30:00",
    ("efficientnet_b3", 150): "03:30:00",
    ("convnext_tiny", 120): "03:00:00",
    ("convnext_tiny", 140): "03:30:00",
}

def get_time_allocation(model: str, epochs: int) -> str:
    """Estimate time allocation based on model and epochs."""
    key = (model, epochs)
    if key in TIME_ALLOCATIONS:
        return TIME_ALLOCATIONS[key]

    # Fallback
    base_times = {
        "resnet50": 80,  # minutes per 100 epochs
        "efficientnet_b3": 100,
    }

    base_min = base_times.get(model, 90)
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

    job_name = f"p2_{model}_{experiment}"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_dir / f"autogen_phase2_{model}_{experiment}.sh"
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
        description="Phase 2 Focused Optimization - Clean Dataset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "efficientnet_b3", "convnext_tiny"],
        default=["efficientnet_b3", "resnet50", "convnext_tiny"],
        help="Models to run experiments on (all three architectures)"
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
    print("PHASE 2: FOCUSED OPTIMIZATION - CLEAN DATASET")
    print("="*80)
    print("\nNOTE: Run this AFTER Phase 1 completes and you've analyzed results!")
    print(f"\nGenerating scripts in: {slurm_dir}")
    print(f"Total experiments: {len(SEARCH_SPACE_PHASE2)}")
    print(f"Models: {', '.join(args.models)}")
    print()

    generated_scripts = []

    for exp_name, exp_config in SEARCH_SPACE_PHASE2.items():
        print(f"\n{exp_name}: {exp_config['description']}")

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
        print("  python hyperparam_search_phase2.py --submit")

    print()
    print("After completion, run:")
    print("  python code/summarize_and_infer.py")
    print()


if __name__ == "__main__":
    main()
