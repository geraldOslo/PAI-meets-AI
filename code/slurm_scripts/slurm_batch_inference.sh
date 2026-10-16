#!/bin/bash
#SBATCH --job-name=PAI_Batch_Inference
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/batch_inference_%j.out
#SBATCH --error=logs/batch_inference_%j.err

# ============================================================================
# PAI Batch Inference on an external set of clips
# ============================================================================
# Runs ResNet50, EfficientNet-B3 and ConvNeXt-Tiny on a folder of clips with a
# data.csv (columns: filename, PAI, ...) and writes, per model, predictions.csv
# with the human PAI (PAI_true) and the predicted PAI (PAI_pred), plus
# metrics_report.txt and a combined summary.csv.
#
# Submit (account, partition, environment and DATA_DIR come from cluster.env):
#   code/slurm_scripts/submit.sh code/slurm_scripts/slurm_batch_inference.sh
#
# Settings, from cluster.env or the environment (the environment wins):
#   DATA_DIR                folder with data.csv and the clips (required)
#   MODEL_CHECKPOINTS_BASE  folder with resnet50_best.pth, efficientnet-b3_best.pth,
#                           convnext-tiny_best.pth   [default: <project root>/model_checkpoints]
#   OUTPUT_DIR              results folder           [default: <project root>/inference_results/batch_<timestamp>]
#   PREDICTIONS_ONLY        1 = only filename, PAI_true, PAI_pred in predictions.csv (default),
#                           0 = also class probabilities and a 'correct' flag
# Example with another set of clips:
#   DATA_DIR=/path/to/other_clips code/slurm_scripts/submit.sh code/slurm_scripts/slurm_batch_inference.sh
# ============================================================================

echo "=================================================="
echo "PAI Batch Inference"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Modules, virtual environment and PYTHONPATH (from cluster.env); sets CODE_DIR
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$PROJECT_ROOT/code/slurm_scripts/job_env.sh" || exit 1

# Paths
if [ -z "${DATA_DIR:-}" ]; then
    echo "ERROR: DATA_DIR is not set - add it to cluster.env or pass DATA_DIR=... when submitting"
    exit 1
fi
MODEL_CHECKPOINTS_BASE="${MODEL_CHECKPOINTS_BASE:-$PROJECT_ROOT/model_checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/inference_results/batch_$(date +%Y%m%d_%H%M%S)}"

RESNET50_CKPT="$MODEL_CHECKPOINTS_BASE/resnet50_best.pth"
EFFICIENTNET_CKPT="$MODEL_CHECKPOINTS_BASE/efficientnet-b3_best.pth"
CONVNEXT_CKPT="$MODEL_CHECKPOINTS_BASE/convnext-tiny_best.pth"

export PYTHONIOENCODING=utf-8

echo "Data directory:  $DATA_DIR"
echo "Checkpoints:     $MODEL_CHECKPOINTS_BASE"
echo "Output:          $OUTPUT_DIR"

# Pre-flight checks
if [ ! -f "$DATA_DIR/data.csv" ]; then
    echo "ERROR: $DATA_DIR/data.csv not found"
    exit 1
fi
for ckpt in "$RESNET50_CKPT" "$EFFICIENTNET_CKPT" "$CONVNEXT_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "WARNING: checkpoint not found (that model will be skipped): $ckpt"
    fi
done
echo "Images in data directory: $(ls "$DATA_DIR" | wc -l) files (incl. data.csv)"
echo "Rows in data.csv: $(( $(wc -l < "$DATA_DIR/data.csv") - 1 ))"

EXTRA_ARGS=()
if [ "${PREDICTIONS_ONLY:-1}" = "1" ]; then
    EXTRA_ARGS+=(--predictions-only)
fi

echo ""
echo "Running inference..."
echo "=================================================="

python -u "$CODE_DIR/test_inference/PAI_batch_inference.py" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --resnet50 "$RESNET50_CKPT" \
    --efficientnet-b3 "$EFFICIENTNET_CKPT" \
    --convnext-tiny "$CONVNEXT_CKPT" \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Inference script finished"
echo "Exit code: $EXIT_CODE"
echo "Results: $OUTPUT_DIR"
echo "Job Finished: $(date)"
echo "=================================================="

exit $EXIT_CODE
