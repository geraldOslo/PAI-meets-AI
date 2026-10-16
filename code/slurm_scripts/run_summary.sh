#!/bin/bash
#SBATCH --job-name=PAI-Summary
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/summary_%j.out
#SBATCH --error=logs/summary_%j.err

# ============================================================================
# PAI Hyperparameter Summary & Inference
# ============================================================================
# Output: CSV file in experiments/hyperparam_search/
# Submit:  code/slurm_scripts/submit.sh code/slurm_scripts/run_summary.sh
# Account, partition and environment come from cluster.env; this job uses
# SUMMARY_MODULES / SUMMARY_VENV_PATH from there if they are set.
# ============================================================================

set -e
set -u

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
JOB_PROFILE=SUMMARY

echo "=================================================="
echo "PAI Summary and Inference Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo "=================================================="

# Modules, virtual environment and PYTHONPATH (from cluster.env)
source "$PROJECT_ROOT/code/slurm_scripts/job_env.sh" || exit 1

# Navigate to project root
cd "$PROJECT_ROOT"
echo "✓ Current directory: $(pwd)"

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

python -u code/summarize_and_infer.py --exp_dir "$PROJECT_ROOT/experiments" --print-table

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