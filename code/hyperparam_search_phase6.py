#!/usr/bin/env python3
"""
================================================================================
Phase 6: ResNet50 Focal Loss Optimization (Targeted Refinement)
================================================================================
Purpose:
  Based on Phase 5 results showing ResNet50 with focal_alpha=0.5 and
  focal_gamma=3.0 as the winner, this phase implements 5 targeted experiments
  to further optimize ResNet50 performance.

Strategy:
  🎯 Experiment Set 1 (p6_resnet_alpha_*): Fine-grained alpha search
     - Test α = [0.35, 0.40, 0.45, 0.55, 0.60] with γ=3.0 fixed
     - Goal: Find optimal alpha between 0.25 and 0.5
     - Expected: QWK 0.810-0.815

  🎯 Experiment Set 2 (p6_resnet_gamma_*): Gamma fine-tuning
     - Test γ = [2.85, 2.90, 2.95, 3.05, 3.10] with best α from Set 1
     - Goal: Find true optimum around γ=3.0
     - Expected: QWK 0.810-0.820

  Note: Experiments 3-5 (class-specific alpha, adaptive gamma, hybrid label
  smoothing) require code modifications and will be implemented separately
  if needed based on results from Sets 1-2.

Timeline:
  - 10 experiments × ~25 minutes = ~4.2 GPU hours total
  - Expected completion: Same day if started on Fox HPC

Hardware:
  - GPU: Any available (RTX30, A100, V100) with 24GB+ memory
  - Batch size: 128 (ResNet50 optimized)
  - Input size: 224×224

Based on Phase 5 Best Configuration:
  Model: ResNet50
  Base Config: Phase 4 champion settings
  - base_lr: 0.0003, max_lr: 0.003
  - weight_decay: 0.005
  - dropout: 0.5
  - focal_gamma: 3.0 (Phase 5 winner)
  - focal_alpha: 0.5 (Phase 5 winner)
  - use_oversampling: True
  - epochs: 75

Author: Gerald Torgersen
Date: January 2025
SPDX-License-Identifier: MIT
================================================================================
"""

import os
from pathlib import Path
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLURM_SCRIPTS_DIR = PROJECT_ROOT / "code" / "slurm_scripts"

# Base configuration from Phase 5 winner (ResNet50, alpha=0.5, gamma=3.0)
BASE_CONFIG_PHASE5_WINNER = {
    "--epochs": 75,
    "--base_lr": 0.0003,
    "--max_lr": 0.003,
    "--weight_decay": 0.005,
    "--dropout": 0.5,
    "--focal_gamma": 3.0,
    "--focal_alpha": 0.5,
    "--use_oversampling": "",  # Boolean flag
}

# ============================================================================
# PHASE 6 SEARCH SPACE: ResNet50 Focal Loss Optimization
# ============================================================================

SEARCH_SPACE_PHASE6 = {
    # ========================================================================
    # SET 1: Fine-Grained Alpha Search (5 experiments)
    # Priority: HIGHEST (85% success probability)
    # ========================================================================
    "p6_resnet_alpha_0_35": {
        "description": "ResNet50: Fine alpha search α=0.35, γ=3.0 fixed",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.35,
            "--focal_gamma": 3.0,
        }
    },
    "p6_resnet_alpha_0_40": {
        "description": "ResNet50: Fine alpha search α=0.40, γ=3.0 fixed",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.40,
            "--focal_gamma": 3.0,
        }
    },
    "p6_resnet_alpha_0_45": {
        "description": "ResNet50: Fine alpha search α=0.45, γ=3.0 fixed",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.45,
            "--focal_gamma": 3.0,
        }
    },
    "p6_resnet_alpha_0_55": {
        "description": "ResNet50: Fine alpha search α=0.55, γ=3.0 fixed",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.55,
            "--focal_gamma": 3.0,
        }
    },
    "p6_resnet_alpha_0_60": {
        "description": "ResNet50: Fine alpha search α=0.60, γ=3.0 fixed",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.60,
            "--focal_gamma": 3.0,
        }
    },

    # ========================================================================
    # SET 2: Gamma Fine-Tuning (5 experiments)
    # Priority: HIGH (70% success probability)
    # Note: Using α=0.5 as baseline; update to best α from Set 1 if better
    # ========================================================================
    "p6_resnet_gamma_2_85": {
        "description": "ResNet50: Fine gamma search γ=2.85, α=0.5",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.5,
            "--focal_gamma": 2.85,
        }
    },
    "p6_resnet_gamma_2_90": {
        "description": "ResNet50: Fine gamma search γ=2.90, α=0.5",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.5,
            "--focal_gamma": 2.90,
        }
    },
    "p6_resnet_gamma_2_95": {
        "description": "ResNet50: Fine gamma search γ=2.95, α=0.5",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.5,
            "--focal_gamma": 2.95,
        }
    },
    "p6_resnet_gamma_3_05": {
        "description": "ResNet50: Fine gamma search γ=3.05, α=0.5",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.5,
            "--focal_gamma": 3.05,
        }
    },
    "p6_resnet_gamma_3_10": {
        "description": "ResNet50: Fine gamma search γ=3.10, α=0.5",
        "models": ["resnet50"],
        "params": {
            **BASE_CONFIG_PHASE5_WINNER,
            "--focal_alpha": 0.5,
            "--focal_gamma": 3.10,
        }
    },
}

# ============================================================================
# SLURM TEMPLATE FOR PHASE 6
# ============================================================================

SLURM_TEMPLATE_PHASE6 = """#!/bin/bash
#SBATCH --job-name=pai_{experiment}
#SBATCH --account=ec192
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time={time_allocation}
#SBATCH --output=code/slurm_scripts/logs/pai_{experiment}_%j.out
#SBATCH --error=code/slurm_scripts/logs/pai_{experiment}_%j.err
# Request any available GPU (RTX30, A100, V100, etc.)

# ============================================================================
# Phase 6: Focal Loss Optimization - {experiment}
# ============================================================================
# {description}
# ============================================================================

set -e
set -u

PROJECT_ROOT="/fp/projects01/ec192/Github/PAI-meets-AI-2"
VENV_PATH="/fp/projects01/ec192/python_envs/cnn_env"

echo "=================================================="
echo "Phase 6 Experiment: {experiment}"
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
echo "Experiment: {experiment}"
echo "Description: {description}"
echo ""

python -u code/training/train_simple.py \\
    --models {model} \\
    --output_dir experiments/{experiment} \\
    {params}

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Experiment: {experiment}"
echo "Status: $([ $EXIT_CODE -eq 0 ] && echo 'COMPLETED' || echo 'FAILED')"
echo "Exit Code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def estimate_time_allocation(epochs: int, model: str) -> str:
    """
    Estimate time allocation for Slurm job based on epochs and model.

    Based on Phase 5 empirical timing:
    - ResNet50: ~20 seconds/epoch on Fox HPC GPUs
    """
    # Base time per epoch (minutes)
    time_per_epoch = {
        'resnet50': 0.33,        # ~20 seconds/epoch
    }

    base_minutes = epochs * time_per_epoch.get(model, 0.33)

    # Add 30% buffer for startup, data loading, inference
    total_minutes = int(base_minutes * 1.3)

    # Minimum 20 minutes, maximum 8 hours
    total_minutes = max(20, min(total_minutes, 480))

    hours = total_minutes // 60
    minutes = total_minutes % 60

    return f"{hours:02d}:{minutes:02d}:00"


def format_params_for_slurm(params: dict) -> str:
    """Format parameter dictionary for Slurm script."""
    lines = []
    for key, value in params.items():
        if value == "":  # Boolean flags
            lines.append(f"    {key} \\")
        else:
            lines.append(f"    {key} {value} \\")

    # Remove trailing backslash from last line
    if lines:
        lines[-1] = lines[-1].rstrip(" \\")

    return "\n".join(lines)


def generate_slurm_script(experiment_name: str, config: dict, model: str) -> str:
    """Generate Slurm script content for an experiment."""
    epochs = config["params"].get("--epochs", 75)
    time_allocation = estimate_time_allocation(epochs, model)
    params_str = format_params_for_slurm(config["params"])

    return SLURM_TEMPLATE_PHASE6.format(
        experiment=experiment_name,
        description=config["description"],
        time_allocation=time_allocation,
        model=model,
        params=params_str
    )


def generate_all_scripts(submit: bool = False):
    """Generate all Phase 6 Slurm scripts."""

    # Ensure logs directory exists
    logs_dir = SLURM_SCRIPTS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 6: ResNet50 Focal Loss Optimization - Script Generation")
    print("=" * 80)
    print()
    print("Strategy:")
    print("  Set 1: Fine-grained alpha search (5 experiments)")
    print("         α = [0.35, 0.40, 0.45, 0.55, 0.60] with γ=3.0")
    print("  Set 2: Gamma fine-tuning (5 experiments)")
    print("         γ = [2.85, 2.90, 2.95, 3.05, 3.10] with α=0.5")
    print()
    print(f"Total experiments: {len(SEARCH_SPACE_PHASE6)}")
    print(f"Model: ResNet50 only")
    print(f"Estimated total GPU time: ~4.2 hours")
    print()

    generated_scripts = []

    for experiment_name, config in SEARCH_SPACE_PHASE6.items():
        # Phase 6 uses only ResNet50
        model = "resnet50"

        script_content = generate_slurm_script(experiment_name, config, model)
        script_filename = f"autogen_p6_{experiment_name}.sh"
        script_path = SLURM_SCRIPTS_DIR / script_filename

        # Write script
        with open(script_path, 'w', newline='\n') as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)

        generated_scripts.append(script_path)
        print(f"✓ Generated: {script_filename}")

    print()
    print("=" * 80)
    print(f"✓ Generated {len(generated_scripts)} Slurm scripts")
    print(f"Location: {SLURM_SCRIPTS_DIR}/autogen_p6_*.sh")
    print()

    if submit:
        print("Submitting jobs to Slurm...")
        print()
        import subprocess

        job_ids = []
        for script_path in generated_scripts:
            try:
                result = subprocess.run(
                    ['sbatch', str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                output = result.stdout.strip()
                print(f"✓ Submitted: {script_path.name}")
                print(f"  {output}")

                # Extract job ID
                if "Submitted batch job" in output:
                    job_id = output.split()[-1]
                    job_ids.append(job_id)

            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to submit {script_path.name}: {e}")

        print()
        print("=" * 80)
        print(f"✓ Submitted {len(job_ids)} jobs successfully")
        print()
        print("Monitor jobs with:")
        print("  squeue -u $USER")
        print()
        print("View logs:")
        print("  tail -f code/slurm_scripts/logs/pai_p6_*_*.out")
        print()
        print("After completion, summarize results:")
        print("  sbatch code/slurm_scripts/run_summary.sh")
        print("=" * 80)
    else:
        print("Review the generated scripts, then submit with:")
        print("  python code/hyperparam_search_phase6.py --submit")
        print()
        print("Or submit individual experiments:")
        print(f"  cd {SLURM_SCRIPTS_DIR}")
        print("  sbatch autogen_p6_resnet_alpha_0_35.sh")
        print()
        print("Monitor with:")
        print("  squeue -u $USER")
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: ResNet50 Focal Loss Optimization - Generate and submit Slurm scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate scripts only (review before submitting)
  python code/hyperparam_search_phase6.py

  # Generate and auto-submit all experiments
  python code/hyperparam_search_phase6.py --submit

Phase 6 Strategy:
  Set 1: Fine-grained alpha search (α = 0.35, 0.40, 0.45, 0.55, 0.60)
  Set 2: Gamma fine-tuning (γ = 2.85, 2.90, 2.95, 3.05, 3.10)

  Total: 10 experiments, ~4.2 GPU hours
  Expected improvement: QWK 0.810-0.820

After completion:
  sbatch code/slurm_scripts/run_summary.sh
        """
    )

    parser.add_argument(
        '--submit',
        action='store_true',
        help='Automatically submit generated scripts to Slurm'
    )

    args = parser.parse_args()

    generate_all_scripts(submit=args.submit)


if __name__ == '__main__':
    main()
