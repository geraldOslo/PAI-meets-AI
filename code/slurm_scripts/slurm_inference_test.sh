#!/bin/bash
#SBATCH --job-name=PAI_Inference
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# ============================================================================
# PAI Classification Inference & GradCAM Generation (Final Version)
# ============================================================================
# Submit:  code/slurm_scripts/submit.sh code/slurm_scripts/slurm_inference_test.sh
# ============================================================================
echo "=================================================="
echo "PAI Inference Job"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Modules, virtual environment and PYTHONPATH (from cluster.env); sets CODE_DIR
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$PROJECT_ROOT/code/slurm_scripts/job_env.sh" || exit 1

CMD="python $CODE_DIR/test_inference/inference_gradcam.py"

echo ""
echo "Running command:"
echo "$CMD"
echo "=================================================="

# Execute the script
eval $CMD

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Inference script finished"
echo "Exit code: $EXIT_CODE"
echo "Job Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE