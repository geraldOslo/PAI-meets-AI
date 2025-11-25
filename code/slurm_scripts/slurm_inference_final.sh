#!/bin/bash
#SBATCH --job-name=PAI_Inference
#SBATCH --account=ec192
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=accel
#SBATCH --gres=gpu:rtx30:1
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# ============================================================================
# PAI Classification Inference & GradCAM Generation (Final Version)
# ============================================================================
echo "=================================================="
echo "PAI Inference Job"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Load modules
echo "Loading modules..."
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.6.0

# Activate virtual environment
VENV_PATH="/fp/projects01/ec192/python_envs/cnn_env"
echo "Activating virtual environment: $VENV_PATH"
source $VENV_PATH/bin/activate

# --- THE DEFINITIVE FIX: EXPLICITLY SET THE PYTHONPATH ---
# This tells the Python interpreter exactly where to find your 'code' package.
# This resolves all relative import issues in the SLURM environment.
CODE_DIR="/fp/projects01/ec192/Github/PAI-meets-AI-2/code"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
echo "Setting PYTHONPATH to: $PYTHONPATH"
# ---------------------------------------------------------

# The rest of the script remains simple. No 'cd' is needed.
# We will call the script using its full path.
CMD="python $CODE_DIR/test_inference/inference_gradcam.py  --checkpoint-dir /fp/projects01/ec192/Github/PAI-meets-AI-2/model_checkpoints"

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