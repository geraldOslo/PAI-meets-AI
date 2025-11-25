#!/usr/bin/env python3
"""
Phase 4: Champion Refinement (3 Experiments)
==============================================

Goal: Fine-tune the best overall configuration (ResNet50 Phase 1 baseline, QWK=0.7951)
      with small variations to see if we can squeeze out additional performance.

Strategy: Test 3 targeted modifications:
    1. Slightly higher focal gamma (2.0 → 2.5) for better boundary focus
    2. Add mixup (helped PAI-2: 0.47→0.62 in Phase 1)
    3. Extended training (75 → 100 epochs) + gamma 2.5

Baseline Configuration (Champion):
    Model: ResNet50
    Focal Gamma: 2.0
    Max LR: 3e-3
    Dropout: 0.5
    Epochs: 75
    Mixup: No
    Performance: QWK=0.7951, PAI-2/3=0.57/0.57
"""

import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# PHASE 4 EXPERIMENT DEFINITIONS
# =============================================================================

SEARCH_SPACE_PHASE4 = {
    # Experiment 1: Slightly higher gamma for boundary focus
    "p4_champion_gamma_2_5": {
        "description": "Champion baseline + focal gamma 2.5 (modest boundary focus)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.5,  # 2.0 → 2.5 (modest increase)
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": ""
        }
    },

    # Experiment 2: Add mixup (proved effective for PAI-2 in Phase 1)
    "p4_champion_with_mixup": {
        "description": "Champion baseline + mixup 0.2 (PAI-2 boost)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 75,
            "--patience": 20,
            "--focal_gamma": 2.0,  # Keep original
            "--max_lr": 0.003,
            "--dropout": 0.5,
            "--label_smoothing": 0.1,
            "--use_oversampling": "",
            "--mixup": "",
            "--mixup_alpha": 0.2  # Proved effective in Phase 1
        }
    },

    # Experiment 3: Extended training + moderate gamma
    "p4_champion_extended_gamma": {
        "description": "Champion baseline + 100 epochs + gamma 2.5 (combined refinement)",
        "models": ["resnet50"],
        "params": {
            "--epochs": 100,  # 75 → 100 (extended)
            "--patience": 25,  # Adjusted for longer training
            "--focal_gamma": 2.5,  # Modest increase
            "--max_lr": 0.003,
            "--dropout": 0.5,
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
#SBATCH --gres=gpu:rtx30:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time={time_str}
#SBATCH --output={log_path}

# ============================================================================
# Phase 4: Champion Refinement - {experiment}
# ============================================================================
# {description}
# ============================================================================

set -e
set -u

PROJECT_ROOT="/fp/projects01/ec192/Github/PAI-meets-AI-2"
VENV_PATH="/fp/projects01/ec192/python_envs/cnn_env"

echo "=================================================="
echo "Phase 4 Experiment: {experiment}"
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

python -u code/training/train_simple.py \\
    --models {model} \\
    --experiment_name {experiment} \\
    {params}

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Training finished with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def estimate_time(model, epochs):
    """Estimate training time based on model and epochs."""
    # Base time per epoch (minutes)
    time_per_epoch = {
        "resnet50": 0.4,
        "efficientnet_b3": 0.8,
        "convnext_tiny": 0.6
    }

    base_time = time_per_epoch.get(model, 0.6) * epochs
    # Add 50% buffer for safety
    total_time = int(base_time * 1.5)

    hours = total_time // 60
    minutes = total_time % 60

    return f"{hours:02d}:{minutes:02d}:00"

def format_params(params_dict):
    """Format parameters for command line."""
    formatted = []
    for key, value in params_dict.items():
        if value == "":  # Boolean flags
            formatted.append(key)
        else:
            formatted.append(f"{key} {value}")
    return " \\\n    ".join(formatted)

def generate_slurm_script(experiment, config, models):
    """Generate Slurm script for an experiment."""
    scripts_generated = []

    for model in models:
        # Estimate time
        epochs = config["params"].get("--epochs", 75)
        time_str = estimate_time(model, epochs)

        # Format job name
        job_name = f"p4_{model}_{experiment[:20]}"

        # Format log path
        log_path = f"/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/pai_train_{experiment}_{model}_%j.out"

        # Format parameters
        params_str = format_params(config["params"])

        # Generate script content
        script_content = SLURM_TEMPLATE.format(
            job_name=job_name,
            experiment=experiment,
            model=model,
            description=config["description"],
            time_str=time_str,
            log_path=log_path,
            params=params_str
        )

        # Write script
        script_path = Path("code/slurm_scripts") / f"autogen_phase4_{model}_{experiment}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        scripts_generated.append(script_path)
        print(f"✓ Generated: {script_path}")

    return scripts_generated

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Champion Refinement (3 Focused Experiments)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "efficientnet_b3"],
        default=["resnet50"],
        help="Models to optimize (default: resnet50 only)"
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

    print("=" * 70)
    print("PHASE 4: CHAMPION REFINEMENT")
    print("=" * 70)
    print(f"\nTarget: Improve on best overall QWK=0.7951")
    print(f"Strategy: 3 small variations of champion configuration")
    print(f"Models: {', '.join(args.models)}")
    print(f"\nExperiments: {len(SEARCH_SPACE_PHASE4)}")
    print()

    # Display experiments
    for i, (exp_name, config) in enumerate(SEARCH_SPACE_PHASE4.items(), 1):
        print(f"{i}. {exp_name}")
        print(f"   {config['description']}")
        if "models" in config:
            print(f"   Models: {', '.join(config['models'])}")
        print()

    # Generate scripts
    print("\nGenerating Slurm scripts...")
    print("-" * 70)

    all_scripts = []
    for exp_name, config in SEARCH_SPACE_PHASE4.items():
        # Determine which models to use
        exp_models = config.get("models", args.models)
        # Filter to only requested models
        exp_models = [m for m in exp_models if m in args.models]

        if exp_models:
            scripts = generate_slurm_script(exp_name, config, exp_models)
            all_scripts.extend(scripts)

    print(f"\n✓ Generated {len(all_scripts)} Slurm scripts")

    # Submit if requested
    if args.submit and not args.dry_run:
        print("\n" + "=" * 70)
        print("SUBMITTING JOBS")
        print("=" * 70)

        for script in all_scripts:
            try:
                result = subprocess.run(
                    ["sbatch", str(script)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"✓ Submitted: {script.name}")
                print(f"  {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to submit {script.name}")
                print(f"  Error: {e.stderr}")

        print(f"\n✓ Submitted {len(all_scripts)} jobs")
        print("\nMonitor with: squeue -u $USER")
        print("View logs: tail -f code/slurm_scripts/logs/pai_train_p4_*.out")

    elif args.dry_run:
        print("\n✓ Dry run complete. Scripts generated but not submitted.")
        print("  Review scripts in: code/slurm_scripts/autogen_phase4_*.sh")
        print("  To submit: python hyperparam_search_phase4.py --submit")

    else:
        print("\n✓ Scripts generated successfully!")
        print("\nNext steps:")
        print("  1. Review scripts: ls -lh code/slurm_scripts/autogen_phase4_*.sh")
        print("  2. Submit jobs: python code/hyperparam_search_phase4.py --submit")
        print("  3. Monitor: squeue -u $USER")
        print("  4. After completion: sbatch code/slurm_scripts/run_summary.sh")

if __name__ == "__main__":
    main()
