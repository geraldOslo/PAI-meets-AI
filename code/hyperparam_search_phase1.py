#!/usr/bin/env python3
"""
Phase 1: Baseline Hyperparameter Search for Clean Dataset
Systematic exploration of key hyperparameters across all architectures

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import os
import argparse
import subprocess
from pathlib import Path

# ==============================================================================
# PHASE 1: BASELINE EXPLORATION
# ==============================================================================
SEARCH_SPACE_PHASE1 = {
    "exp1_baseline": {
        "description": "Baseline with standard focal loss",
        "models": ["resnet50", "efficientnet_b3", "convnext_tiny"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.0,
            "--max_lr": 0.003,
            "--use_oversampling": ""
        }
    },

    "exp2_stronger_regularization": {
        "description": "Test higher dropout for better generalization",
        "models": ["resnet50", "efficientnet_b3", "convnext_tiny"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--dropout": 0.6,
            "--focal_gamma": 2.0,
            "--max_lr": 0.003,
            "--use_oversampling": ""
        }
    },

    "exp3_lower_lr_longer": {
        "description": "Lower LR with extended training for stability",
        "models": ["resnet50", "efficientnet_b3", "convnext_tiny"],
        "params": {
            "--epochs": 100,
            "--patience": 25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.6,
            "--focal_gamma": 2.0,
            "--use_oversampling": ""
        }
    },

    "exp4_with_mixup": {
        "description": "Test mixup augmentation for robustness",
        "models": ["resnet50", "efficientnet_b3", "convnext_tiny"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.0,
            "--max_lr": 0.003,
            "--use_oversampling": "",
            "--mixup": "",
            "--mixup_alpha": 0.2
        }
    },

    "exp5_focal_loss_tuned": {
        "description": "Higher focal gamma to address hard examples",
        "models": ["resnet50", "efficientnet_b3", "convnext_tiny"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.0,
            "--max_lr": 0.003,
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
# PAI Phase 1 Hyperparameter Search - Clean Dataset
# ============================================================================

echo "=================================================="
echo "Starting Phase 1 Training Job: {job_name}"
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
    ("resnet50", 75): "01:30:00",
    ("resnet50", 100): "02:00:00",
    ("resnet50", 150): "03:00:00",
    ("efficientnet_b3", 75): "02:00:00",
    ("efficientnet_b3", 100): "02:30:00",
    ("efficientnet_b3", 150): "03:30:00",
    ("convnext_tiny", 75): "01:45:00",
    ("convnext_tiny", 100): "02:15:00",
    ("convnext_tiny", 150): "03:15:00",
}

def get_time_allocation(model: str, epochs: int) -> str:
    """Estimate time allocation based on model and epochs."""
    key = (model, epochs)
    if key in TIME_ALLOCATIONS:
        return TIME_ALLOCATIONS[key]

    # Fallback: rough estimation
    base_times = {
        "resnet50": 60,  # minutes per 75 epochs
        "efficientnet_b3": 90,
        "convnext_tiny": 70
    }

    base_min = base_times.get(model, 75)
    total_min = int(base_min * (epochs / 75) * 1.2)  # 20% buffer
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
    epochs = params.get("--epochs", 75)

    job_name = f"p1_{model}_{experiment}"
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_dir / f"autogen_phase1_{model}_{experiment}.sh"

    # Use Linux paths for Slurm (will be executed on HPC cluster)
    log_path = f"/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/{job_name}.out"
    err_path = f"/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/{job_name}.err"

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

    # Make executable
    os.chmod(script_path, 0o755)

    return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Hyperparameter Search - Clean Dataset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "efficientnet_b3", "convnext_tiny"],
        default=["resnet50", "efficientnet_b3", "convnext_tiny"],
        help="Models to run experiments on"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Automatically submit generated Slurm scripts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts but don't submit (overrides --submit)"
    )

    args = parser.parse_args()

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    slurm_dir = script_dir / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)

    print("="*80)
    print("PHASE 1: BASELINE HYPERPARAMETER SEARCH - CLEAN DATASET")
    print("="*80)
    print(f"\nGenerating scripts in: {slurm_dir}")
    print(f"Total experiments: {len(SEARCH_SPACE_PHASE1)}")
    print(f"Models: {', '.join(args.models)}")
    print()

    generated_scripts = []

    # Generate scripts for each experiment × model combination
    for exp_name, exp_config in SEARCH_SPACE_PHASE1.items():
        print(f"\n{exp_name}: {exp_config['description']}")

        # Check if experiment restricts models
        exp_models = exp_config.get("models", args.models)
        exp_models = [m for m in exp_models if m in args.models]

        for model in exp_models:
            script_path = generate_slurm_script(model, exp_name, exp_config, slurm_dir)
            generated_scripts.append(script_path)
            print(f"  [OK] Generated: {script_path.name}")

    print()
    print("="*80)
    print(f"Generated {len(generated_scripts)} Slurm scripts")
    print("="*80)

    # Submit if requested
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
                print(f"  [OK] Submitted: {script.name} - {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] Failed to submit {script.name}: {e.stderr}")

        print("\nAll jobs submitted!")
        print("Monitor with: squeue -u $USER")
        print("View logs in: code/slurm_scripts/logs/")
    else:
        print("\nTo submit these jobs:")
        print("  python hyperparam_search_phase1.py --submit")
        print("\nOr manually:")
        print("  cd code/slurm_scripts")
        print("  sbatch autogen_phase1_*.sh")

    print()
    print("After completion, run:")
    print("  python code/summarize_and_infer.py")
    print()


if __name__ == "__main__":
    main()
