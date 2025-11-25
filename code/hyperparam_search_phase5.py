#!/usr/bin/env python3
"""
Phase 5: Focal Loss Deep Dive (15 Experiments)
==============================================

Goal: Systematically explore focal loss parameter space (gamma and alpha)
      based on best configurations from Phases 1-4 to optimize for PAI 2/3 boundary.

Background:
    - Phase 1-4 tested gamma: 2.0, 2.5, 3.0, 3.5
    - Best overall: gamma=2.0 (ResNet50, QWK=0.7951)
    - Best boundary: gamma=3.5 (ResNet50, PAI-2/3=0.60/0.57)
    - Focal alpha fixed at 0.25 (never varied)

Strategy: 5 experiments per model × 3 models = 15 total
    For each model's best configuration:
        1. Baseline reproduction (gamma=2.0, alpha=0.25)
        2. Gamma sweep: 2.25, 2.75, 3.25
        3. Alpha variation: gamma=3.0, alpha=0.5
        4. Combined: gamma=2.5, alpha=0.35

Models and Base Configurations:
    - ResNet50: exp1_baseline (QWK=0.7951, champion)
    - EfficientNet-B3: exp5_lower_lr (QWK=0.7811)
    - ConvNeXt-Tiny: exp2_moderate_settings (QWK=0.7447)

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
Copyright (c) 2025 Gerald Torgersen
"""

import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# PHASE 5 EXPERIMENT DEFINITIONS
# =============================================================================

SEARCH_SPACE_PHASE5 = {
    # ===========================================================================
    # RESNET50 EXPERIMENTS (Champion baseline: QWK=0.7951)
    # ===========================================================================
    "p5_resnet_baseline_reproduce": {
        "description": "ResNet50 champion baseline reproduction (gamma=2.0, alpha=0.25)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.0,
            "--focal_alpha": 0.25,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_resnet_gamma_2_25": {
        "description": "ResNet50 with gamma=2.25 (between 2.0 and 2.5)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.25,  # Fine-grained gamma exploration
            "--focal_alpha": 0.25,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_resnet_gamma_2_75": {
        "description": "ResNet50 with gamma=2.75 (between 2.5 and 3.0)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.75,
            "--focal_alpha": 0.25,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_resnet_gamma_3_25": {
        "description": "ResNet50 with gamma=3.25 (between 3.0 and 3.5)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.25,
            "--focal_alpha": 0.25,
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_resnet_alpha_0_5": {
        "description": "ResNet50 with gamma=3.0, alpha=0.5 (higher alpha for class imbalance)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 3.0,
            "--focal_alpha": 0.5,  # Doubled alpha
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    # ===========================================================================
    # EFFICIENTNET-B3 EXPERIMENTS (Best: exp5_lower_lr, QWK=0.7811)
    # ===========================================================================
    "p5_effnet_baseline_reproduce": {
        "description": "EfficientNet-B3 best config reproduction (gamma=2.0, alpha=0.25)",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 90,
            "--patience": 25,
            "--focal_gamma": 2.0,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_effnet_gamma_2_25": {
        "description": "EfficientNet-B3 with gamma=2.25",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 90,
            "--patience": 25,
            "--focal_gamma": 2.25,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_effnet_gamma_2_75": {
        "description": "EfficientNet-B3 with gamma=2.75",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 90,
            "--patience": 25,
            "--focal_gamma": 2.75,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_effnet_gamma_3_25": {
        "description": "EfficientNet-B3 with gamma=3.25",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 90,
            "--patience": 25,
            "--focal_gamma": 3.25,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_effnet_alpha_0_5": {
        "description": "EfficientNet-B3 with gamma=3.0, alpha=0.5",
        "models": ["efficientnet_b3"],
        "params": {
            "--epochs": 90,
            "--patience": 25,
            "--focal_gamma": 3.0,
            "--focal_alpha": 0.5,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    # ===========================================================================
    # CONVNEXT-TINY EXPERIMENTS (Best: exp2_moderate_settings, QWK=0.7447)
    # ===========================================================================
    "p5_convnext_baseline_reproduce": {
        "description": "ConvNeXt-Tiny best config reproduction (gamma=2.5, alpha=0.25)",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.5,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.4,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_convnext_gamma_2_25": {
        "description": "ConvNeXt-Tiny with gamma=2.25 (lower than baseline 2.5)",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.25,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.4,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_convnext_gamma_2_75": {
        "description": "ConvNeXt-Tiny with gamma=2.75 (higher than baseline 2.5)",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 2.75,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.4,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_convnext_gamma_3_25": {
        "description": "ConvNeXt-Tiny with gamma=3.25",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 3.25,
            "--focal_alpha": 0.25,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.4,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    "p5_convnext_alpha_0_5": {
        "description": "ConvNeXt-Tiny with gamma=3.0, alpha=0.5",
        "models": ["convnext_tiny"],
        "params": {
            "--epochs": 120,
            "--patience": 30,
            "--focal_gamma": 3.0,
            "--focal_alpha": 0.5,
            "--base_lr": 1e-4,
            "--max_lr": 1e-3,
            "--dropout": 0.4,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },
}

# =============================================================================
# SLURM SCRIPT TEMPLATE
# =============================================================================

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=ec192
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time={time_str}
#SBATCH --output={log_path}
# Request any available GPU (RTX30, A100, V100, etc.)

# ============================================================================
# Phase 5: Focal Loss Deep Dive - {experiment}
# ============================================================================
# {description}
# ============================================================================

set -e
set -u

PROJECT_ROOT="/fp/projects01/ec192/Github/PAI-meets-AI-2"
VENV_PATH="/fp/projects01/ec192/python_envs/cnn_env"

echo "=================================================="
echo "Phase 5 Experiment: {experiment}"
echo "Model: {model}"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Node: ${{SLURM_NODELIST}}"
echo "Started: $(date)"
echo "=================================================="

# Load modules
echo "Loading modules..."
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0
echo "✓ Modules loaded"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✓ Python: $(which python)"
echo "✓ Version: $(python --version)"

# Navigate to project root
echo "Navigating to: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
echo "✓ Current directory: $(pwd)"

# Set PYTHONPATH
export PYTHONPATH="${{PROJECT_ROOT}}:${{PYTHONPATH:-}}"
echo "✓ PYTHONPATH: $PYTHONPATH"

# Verify config
if [ ! -f "code/config.py" ]; then
    echo "ERROR: code/config.py not found!"
    exit 1
fi
echo "✓ Config file found"

# Run training
echo ""
echo "=================================================="
echo "Starting Training"
echo "=================================================="
echo ""

python code/training/train_simple.py \\
    --models {model} \\
    --experiment_name {experiment} \\
    {param_string}

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training Completed"
echo "Exit Code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_time(model: str, epochs: int) -> str:
    """Estimate runtime based on model and epochs."""
    # Base times per epoch (minutes) from previous phases
    base_times = {
        "resnet50": 2.0,
        "efficientnet_b3": 2.5,
        "convnext_tiny": 2.5,
    }

    base_time = base_times.get(model, 2.5)
    estimated_minutes = int(base_time * epochs * 1.3)  # 30% buffer

    hours = estimated_minutes // 60
    minutes = estimated_minutes % 60

    return f"{hours:02d}:{minutes:02d}:00"


def generate_slurm_script(experiment: str, config: dict, output_dir: Path) -> Path:
    """Generate a Slurm submission script."""
    models = config.get("models", ["resnet50"])
    model = models[0]  # Phase 5: one model per experiment

    # Extract epochs for time estimation
    epochs = config["params"].get("--epochs", 75)
    time_str = estimate_time(model, epochs)

    # Build parameter string
    param_parts = []
    for key, value in config["params"].items():
        if value == "":
            param_parts.append(f"    {key}")
        else:
            param_parts.append(f"    {key} {value}")
    param_string = " \\\n".join(param_parts)

    # Generate job name
    job_name = f"p5_{model}_{experiment}"[:50]  # Slurm limit

    # Log paths
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_name}.out"

    # Fill template
    script_content = SLURM_TEMPLATE.format(
        job_name=job_name,
        time_str=time_str,
        log_path=log_path,
        experiment=experiment,
        description=config["description"],
        model=model,
        param_string=param_string
    )

    # Write script
    script_path = output_dir / f"autogen_phase5_{model}_{experiment}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)

    return script_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Focal Loss Deep Dive - Generate and optionally submit Slurm jobs"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Automatically submit generated scripts to Slurm queue"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "efficientnet_b3", "convnext_tiny"],
        default=["resnet50", "efficientnet_b3", "convnext_tiny"],
        help="Models to generate scripts for (default: all)"
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).resolve().parent
    slurm_dir = project_root / "slurm_scripts"
    slurm_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Phase 5: Focal Loss Deep Dive - Slurm Script Generator")
    print("=" * 80)
    print(f"\nTotal experiments defined: {len(SEARCH_SPACE_PHASE5)}")
    print(f"Models to process: {', '.join(args.models)}")
    print(f"Output directory: {slurm_dir}")
    print(f"Auto-submit: {'Yes' if args.submit else 'No'}")
    print()

    # Generate scripts
    generated_scripts = []

    for exp_name, exp_config in SEARCH_SPACE_PHASE5.items():
        # Check if this experiment's model is in the requested models
        exp_models = exp_config.get("models", [])
        if not any(m in args.models for m in exp_models):
            continue

        print(f"Generating: {exp_name}")
        print(f"  Description: {exp_config['description']}")
        print(f"  Model: {exp_models[0]}")

        script_path = generate_slurm_script(exp_name, exp_config, slurm_dir)
        generated_scripts.append((exp_name, script_path))
        print(f"  ✓ Generated: {script_path.name}")
        print()

    # Summary
    print("=" * 80)
    print(f"Generated {len(generated_scripts)} Slurm scripts")
    print("=" * 80)
    print()

    # Submit if requested
    if args.submit:
        print("Submitting jobs to Slurm queue...")
        print()

        submitted = 0
        failed = 0

        for exp_name, script_path in generated_scripts:
            try:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=slurm_dir
                )
                job_id = result.stdout.strip().split()[-1]
                print(f"✓ {exp_name}: Job {job_id}")
                submitted += 1
            except subprocess.CalledProcessError as e:
                print(f"✗ {exp_name}: FAILED - {e.stderr.strip()}")
                failed += 1

        print()
        print("=" * 80)
        print(f"Submission complete: {submitted} submitted, {failed} failed")
        print("=" * 80)
        print()
        print("Monitor jobs with: squeue -u $USER")
        print("View logs in: code/slurm_scripts/logs/")
    else:
        print("Scripts generated but NOT submitted.")
        print()
        print("To submit all jobs, run:")
        print("  python code/hyperparam_search_phase5.py --submit")
        print()
        print("Or submit individually:")
        for exp_name, script_path in generated_scripts[:3]:
            print(f"  sbatch {script_path.name}")
        if len(generated_scripts) > 3:
            print(f"  ... and {len(generated_scripts) - 3} more")


if __name__ == "__main__":
    main()
