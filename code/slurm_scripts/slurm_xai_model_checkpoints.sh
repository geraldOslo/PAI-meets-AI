#!/bin/bash
#SBATCH --job-name=xai_checkpoints
#SBATCH --account=ec192
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/xai_model_checkpoints_%j.out
#SBATCH --error=/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/xai_model_checkpoints_%j.err

# ============================================================================
# Multi-Layer XAI Analysis for Model Checkpoints
# ============================================================================
# This script performs comprehensive explainability analysis using GradCAM
# and GradCAM++ on the three models in model_checkpoints/:
#
#   - ConvNeXt-Tiny:   convnext_exp1_stability_focus
#   - ResNet50:        exp3_lower_lr_longer
#   - EfficientNet-B3: p5_effnet_gamma_2_75
#
# Features:
#   - Multi-layer heatmap generation with weighted fusion
#   - Per-class average heatmaps with quadrant-aware orientation
#   - Individual image visualizations (4-row comprehensive layout)
#   - Comparison between late-layer and combined heatmaps
#
# Usage:
#   sbatch code/slurm_scripts/slurm_xai_model_checkpoints.sh
#
# Output: experiments/{model}/multilayer_xai_{timestamp}/
# ============================================================================

echo "=================================================="
echo "Multi-Layer XAI Analysis - Model Checkpoints"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_JOB_GPUS"
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

# ============================================================================
# CONFIGURATION
# ============================================================================

# Extract test data paths from config.py
echo "Extracting paths from config.py..."
TEST_CSV_1=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_csv_paths[0])")
TEST_CSV_2=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_csv_paths[1])")
TEST_ROOT_1=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_root_dirs[0])")
TEST_ROOT_2=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_root_dirs[1])")
OUTPUT_DIR=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().base_experiments_dir)")

# Use BOTH test datasets (concatenated automatically by the Python script)
TEST_CSVS="$TEST_CSV_1 $TEST_CSV_2"
TEST_ROOTS="$TEST_ROOT_1 $TEST_ROOT_2"

# Checkpoint paths - Model Checkpoints Directory
CONVNEXT_CHECKPOINT="$PROJECT_ROOT/model_checkpoints/convnext_exp1_stability_focus_20251026_051829/convnext-tiny_best.pth"
RESNET_CHECKPOINT="$PROJECT_ROOT/model_checkpoints/exp3_lower_lr_longer_20251024_234835/resnet50_best.pth"
EFFICIENTNET_CHECKPOINT="$PROJECT_ROOT/model_checkpoints/p5_effnet_gamma_2_75_20251028_114607/efficientnet-b3_best.pth"

echo "Using paths from config.py:"
echo "  Test CSV 1: $TEST_CSV_1"
echo "  Test Root 1: $TEST_ROOT_1"
echo "  Test CSV 2: $TEST_CSV_2"
echo "  Test Root 2: $TEST_ROOT_2"
echo "  Output Dir: $OUTPUT_DIR"
echo "  --> Both test sets will be combined for inference"
echo ""
echo "Using model checkpoints:"
echo "  ConvNeXt-Tiny: $CONVNEXT_CHECKPOINT"
echo "  ResNet50: $RESNET_CHECKPOINT"
echo "  EfficientNet-B3: $EFFICIENTNET_CHECKPOINT"
echo ""

# Models to analyze (comment out models you don't want to analyze)
MODELS="convnext_tiny resnet50 efficientnet_b3"

# CAM methods (options: gradcam gradcamplusplus)
CAM_METHODS="gradcam gradcamplusplus"

# Processing options
BATCH_SIZE=""  # Leave empty to use model default
NUM_WORKERS=8
HEATMAP_TRANSPARENCY=0.5

# ============================================================================
# VERIFY CHECKPOINTS EXIST
# ============================================================================

echo "Verifying checkpoint files exist..."
MISSING_FILES=0

if [ ! -f "$CONVNEXT_CHECKPOINT" ]; then
    echo "✗ ERROR: ConvNeXt checkpoint not found: $CONVNEXT_CHECKPOINT"
    MISSING_FILES=1
fi

if [ ! -f "$RESNET_CHECKPOINT" ]; then
    echo "✗ ERROR: ResNet checkpoint not found: $RESNET_CHECKPOINT"
    MISSING_FILES=1
fi

if [ ! -f "$EFFICIENTNET_CHECKPOINT" ]; then
    echo "✗ ERROR: EfficientNet checkpoint not found: $EFFICIENTNET_CHECKPOINT"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo "✗ FATAL ERROR: One or more checkpoint files are missing!"
    echo "Please verify the paths in model_checkpoints/ directory"
    exit 1
fi

echo "✓ All checkpoint files found"
echo ""

# ============================================================================
# RUN ANALYSIS FOR EACH MODEL
# ============================================================================

echo "=================================================="
echo "Starting Multi-Layer XAI Analysis"
echo "=================================================="
echo "Models to process: $MODELS"
echo "CAM Methods: $CAM_METHODS"
echo "Test Data: 2 sets combined (~200 images total)"
echo "  - Set 1: $TEST_CSV_1"
echo "  - Set 2: $TEST_CSV_2"
echo "Output Directory: $OUTPUT_DIR"
echo "=================================================="
echo ""

# Track overall exit code
OVERALL_EXIT_CODE=0

# Process ConvNeXt-Tiny
if [[ "$MODELS" == *"convnext_tiny"* ]]; then
    echo ""
    echo "=== Processing ConvNeXt-Tiny ==="
    echo "Checkpoint: convnext_exp1_stability_focus"
    echo "File: $CONVNEXT_CHECKPOINT"

    python code/test_inference/multilayer_xai_analysis.py \
        --model convnext_tiny \
        --checkpoint "$CONVNEXT_CHECKPOINT" \
        --test-csv $TEST_CSVS \
        --test-root $TEST_ROOTS \
        --output-dir "$OUTPUT_DIR" \
        --cam-methods $CAM_METHODS \
        --num-workers $NUM_WORKERS \
        --heatmap-transparency $HEATMAP_TRANSPARENCY \
        ${BATCH_SIZE:+--batch-size $BATCH_SIZE}

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ ConvNeXt-Tiny analysis failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
    else
        echo "✓ ConvNeXt-Tiny analysis completed successfully"
    fi
fi

# Process ResNet50
if [[ "$MODELS" == *"resnet50"* ]]; then
    echo ""
    echo "=== Processing ResNet50 ==="
    echo "Checkpoint: exp3_lower_lr_longer"
    echo "File: $RESNET_CHECKPOINT"

    python code/test_inference/multilayer_xai_analysis.py \
        --model resnet50 \
        --checkpoint "$RESNET_CHECKPOINT" \
        --test-csv $TEST_CSVS \
        --test-root $TEST_ROOTS \
        --output-dir "$OUTPUT_DIR" \
        --cam-methods $CAM_METHODS \
        --num-workers $NUM_WORKERS \
        --heatmap-transparency $HEATMAP_TRANSPARENCY \
        ${BATCH_SIZE:+--batch-size $BATCH_SIZE}

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ ResNet50 analysis failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
    else
        echo "✓ ResNet50 analysis completed successfully"
    fi
fi

# Process EfficientNet-B3
if [[ "$MODELS" == *"efficientnet_b3"* ]]; then
    echo ""
    echo "=== Processing EfficientNet-B3 ==="
    echo "Checkpoint: p5_effnet_gamma_2_75"
    echo "File: $EFFICIENTNET_CHECKPOINT"

    python code/test_inference/multilayer_xai_analysis.py \
        --model efficientnet_b3 \
        --checkpoint "$EFFICIENTNET_CHECKPOINT" \
        --test-csv $TEST_CSVS \
        --test-root $TEST_ROOTS \
        --output-dir "$OUTPUT_DIR" \
        --cam-methods $CAM_METHODS \
        --num-workers $NUM_WORKERS \
        --heatmap-transparency $HEATMAP_TRANSPARENCY \
        ${BATCH_SIZE:+--batch-size $BATCH_SIZE}

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "✗ EfficientNet-B3 analysis failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
    else
        echo "✓ EfficientNet-B3 analysis completed successfully"
    fi
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "=================================================="
echo "Job Completed: $(date)"
echo "Overall Exit Code: $OVERALL_EXIT_CODE"
echo "=================================================="

# Print output locations
if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS! All models processed successfully"
    echo ""
    echo "Results saved to:"
    echo "  $OUTPUT_DIR/convnext_tiny/multilayer_xai_{timestamp}/"
    echo "  $OUTPUT_DIR/resnet50/multilayer_xai_{timestamp}/"
    echo "  $OUTPUT_DIR/efficientnet_b3/multilayer_xai_{timestamp}/"
    echo ""
    echo "Each directory contains:"
    echo "  - xai_individual_gradcam/         : Individual image visualizations (GradCAM)"
    echo "  - xai_individual_gradcamplusplus/ : Individual image visualizations (GradCAM++)"
    echo "  - average_heatmaps/               : Per-class average heatmaps (KEY OUTPUT)"
    echo "  - prediction_results.csv          : Predictions with probabilities"
    echo "  - confusion_matrix.png            : Confusion matrix visualization"
    echo "  - final_metrics.txt               : Accuracy, QWK, MAE, confusion matrix"
    echo ""
    echo "Model Checkpoints Analyzed:"
    echo "  - ConvNeXt-Tiny:   convnext_exp1_stability_focus_20251026_051829"
    echo "  - ResNet50:        exp3_lower_lr_longer_20251024_234835"
    echo "  - EfficientNet-B3: p5_effnet_gamma_2_75_20251028_114607"
    echo ""
else
    echo ""
    echo "✗ WARNING: Some models failed (see errors above)"
    echo "Exit code: $OVERALL_EXIT_CODE"
    echo ""
fi

exit $OVERALL_EXIT_CODE
