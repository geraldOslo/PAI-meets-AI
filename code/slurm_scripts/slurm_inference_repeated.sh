#!/bin/bash
#SBATCH --job-name=PAI_Repeated_Inference
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/inference_repeated_%j.out
#SBATCH --error=logs/inference_repeated_%j.err

# ============================================================================
# PAI Classification Repeated Inference (3 Runs)
# ============================================================================
# This script runs inference 3 times for the top performing model from each 
# architecture to check for stability/deviations in predictions.
#
# Models:
#   1. ResNet50
#   2. EfficientNet-B3
#   3. ConvNeXt-Tiny
#
# Flags: --no-heatmaps (Output: CSV predictions and statistics report only)
# Submit:  code/slurm_scripts/submit.sh code/slurm_scripts/slurm_inference_repeated.sh
# ============================================================================

echo "=================================================="
echo "PAI Repeated Inference (3 Runs)"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Modules, virtual environment and PYTHONPATH (from cluster.env); sets CODE_DIR
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$PROJECT_ROOT/code/slurm_scripts/job_env.sh" || exit 1

# Base directory for model checkpoints
MODEL_CHECKPOINTS_BASE="${MODEL_CHECKPOINTS_BASE:-$PROJECT_ROOT/model_checkpoints}"
# Main Output directory
DATE_STR=$(date +%Y%m%d_%H%M%S)
MAIN_OUTPUT_DIR="$PROJECT_ROOT/inference_results/repeated_run_${DATE_STR}"
mkdir -p "$MAIN_OUTPUT_DIR"
echo "Main Output directory: $MAIN_OUTPUT_DIR"

# Number of repetitions
NUM_RUNS=3

echo ""
echo "=================================================="
echo "STARTING REPEATED INFERENCE LOOP ($NUM_RUNS runs)"
echo "=================================================="

for ((i=1; i<=NUM_RUNS; i++))
do
    echo ""
    echo "--------------------------------------------------"
    echo "RUN $i / $NUM_RUNS"
    echo "--------------------------------------------------"
    
    RUN_OUTPUT_DIR="$MAIN_OUTPUT_DIR/run_$i"
    mkdir -p "$RUN_OUTPUT_DIR"
    
    # ------------------------------------------------------------------------
    # MODEL 1: ResNet50
    # ------------------------------------------------------------------------
    echo "  > [1/3] ResNet50..."
    
    CMD1="python $CODE_DIR/test_inference/inference_gradcam.py \
      --resnet50 $MODEL_CHECKPOINTS_BASE/resnet50_best.pth \
      --models resnet50 \
      --output-dir $RUN_OUTPUT_DIR/resnet50 \
      --no-heatmaps"
    
    echo "    Command: $CMD1"
    eval $CMD1
    EXIT_1=$?
    
    if [ $EXIT_1 -ne 0 ]; then
        echo "    ERROR: ResNet50 failed in Run $i (Exit code: $EXIT_1)"
    else
        echo "    ✓ ResNet50 completed"
    fi

    # ------------------------------------------------------------------------
    # MODEL 2: EfficientNet-B3
    # ------------------------------------------------------------------------
    echo "  > [2/3] EfficientNet-B3..."
    
    CMD2="python $CODE_DIR/test_inference/inference_gradcam.py \
      --efficientnet-b3 $MODEL_CHECKPOINTS_BASE/efficientnet-b3_best.pth \
      --models efficientnet_b3 \
      --output-dir $RUN_OUTPUT_DIR/efficientnet_b3 \
      --no-heatmaps"
    
    echo "    Command: $CMD2"
    eval $CMD2
    EXIT_2=$?
    
    if [ $EXIT_2 -ne 0 ]; then
        echo "    ERROR: EfficientNet-B3 failed in Run $i (Exit code: $EXIT_2)"
    else
        echo "    ✓ EfficientNet-B3 completed"
    fi

    # ------------------------------------------------------------------------
    # MODEL 3: ConvNeXt-Tiny
    # ------------------------------------------------------------------------
    echo "  > [3/3] ConvNeXt-Tiny..."
    
    CMD3="python $CODE_DIR/test_inference/inference_gradcam.py \
      --convnext-tiny $MODEL_CHECKPOINTS_BASE/convnext-tiny_best.pth \
      --models convnext_tiny \
      --output-dir $RUN_OUTPUT_DIR/convnext_tiny \
      --no-heatmaps"
    
    echo "    Command: $CMD3"
    eval $CMD3
    EXIT_3=$?
    
    if [ $EXIT_3 -ne 0 ]; then
        echo "    ERROR: ConvNeXt-Tiny failed in Run $i (Exit code: $EXIT_3)"
    else
        echo "    ✓ ConvNeXt-Tiny completed"
    fi
    
done

echo ""
echo "=================================================="
echo "ALL REPEATED INFERENCE RUNS COMPLETED"
echo "=================================================="
echo "Results saved to: $MAIN_OUTPUT_DIR"
echo "Job Finished: $(date)"
echo "=================================================="
