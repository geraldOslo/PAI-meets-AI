#!/bin/bash
#SBATCH --job-name=PAI-Summary
#SBATCH --account=ec192
#SBATCH --partition=accel
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx30:1
#SBATCH --output=/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/summary_%j.out
#SBATCH --error=/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/summary_%j.err

# ============================================================================
# PAI Hyperparameter Summary & Inference
# ============================================================================
# Output: CSV file in experiments/hyperparam_search/
# ============================================================================

set -e
set -u

PROJECT_ROOT="/fp/projects01/ec192/Github/PAI-meets-AI-2"
VENV_PATH="/fp/projects01/ec192/python_envs/cnn_env"

echo "=================================================="
echo "PAI Summary and Inference Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "=================================================="

# Load modules (use PyTorch module which has all dependencies)
echo "Loading modules..."
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
echo "✓ Modules loaded"
echo "✓ Python: $(which python)"
echo "✓ Version: $(python --version)"

# Navigate to project root
echo "Navigating to: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
echo "✓ Current directory: $(pwd)"

# CRITICAL: Set PYTHONPATH so Python can find the code package
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
echo "✓ PYTHONPATH: $PYTHONPATH"

# Verify files exist
if [ ! -f "code/config.py" ]; then
    echo "ERROR: code/config.py not found!"
    exit 1
fi

if [ ! -f "code/summarize_and_infer.py" ]; then
    echo "ERROR: code/summarize_and_infer.py not found!"
    exit 1
fi

echo "✓ All required files found"

# Test the imports before running the full script
echo ""
echo "Testing Python imports..."
python -c "from code.config import DataConfig; print('✓ config imports work')" || {
    echo "ERROR: Cannot import from code.config"
    echo "Checking Python path:"
    python -c "import sys; print('\n'.join(sys.path))"
    exit 1
}

# Run the script
echo ""
echo "=================================================="
echo "Running Summary & Inference"
echo "=================================================="
echo ""

python -u code/summarize_and_infer.py --exp_dir /fp/projects01/ec192/Github/PAI-meets-AI-2/experiments --print-table

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Script finished with exit code: $EXIT_CODE"
echo "=================================================="

# Show results if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Checking for output files..."
    SUMMARY_DIR="$PROJECT_ROOT/experiments/hyperparam_search"
    
    if [ -d "$SUMMARY_DIR" ]; then
        LATEST_CSV=$(ls -t "$SUMMARY_DIR"/hyperparameter_summary_*.csv 2>/dev/null | head -1)
        
        if [ -n "$LATEST_CSV" ]; then
            echo "✓ Summary created: $(basename "$LATEST_CSV")"
            echo "  Location: $LATEST_CSV"
            echo "  Size: $(du -h "$LATEST_CSV" | cut -f1)"
            echo "  Lines: $(wc -l < "$LATEST_CSV")"
            
            # Show first few rows
            echo ""
            echo "First 5 rows of summary:"
            head -5 "$LATEST_CSV"
        else
            echo "⚠ No summary CSV found"
        fi
    fi
fi

echo ""
echo "Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE