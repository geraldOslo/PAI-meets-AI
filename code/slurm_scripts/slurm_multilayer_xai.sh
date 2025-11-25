#!/bin/bash
#SBATCH --job-name=xai_top3_final
#SBATCH --account=ec192
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/multilayer_xai_%j.out
#SBATCH --error=/fp/projects01/ec192/Github/PAI-meets-AI-2/code/slurm_scripts/logs/multilayer_xai_%j.err

# ============================================================================
# Multi-Layer XAI Analysis for TOP 3 WINNING PAI Classification Models
# ============================================================================
# This script performs comprehensive explainability analysis using GradCAM
# and GradCAM++ on the three best-performing models from all phases:
#
#   🥇 ResNet50:        QWK=0.8068 (p5_resnet_alpha_0_5)
#   🥈 EfficientNet-B3: QWK=0.7811 (effnet_exp5_lower_lr)
#   🥉 ConvNeXt-Tiny:   QWK=0.7447 (convnext_exp2_moderate_settings)
#
# Features:
#   - Multi-layer heatmap generation with weighted fusion
#   - Per-class average heatmaps with quadrant-aware orientation (KEY OUTPUT)
#   - Individual image visualizations (4-row comprehensive layout)
#   - Comparison between late-layer and combined heatmaps
#
# Usage:
#   sbatch slurm_multilayer_xai.sh
#
# Output: experiments/{model}/multilayer_xai_{timestamp}/average_heatmaps/
# ============================================================================

echo "=================================================="
echo "Multi-Layer XAI Analysis Job"
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
# CONFIGURATION - Paths loaded from config.py
# ============================================================================

# Python command to extract config values
# This reads from code/config.py to ensure consistency
echo "Extracting paths from config.py..."
cd $PROJECT_ROOT

# Extract test data paths from InferenceConfig
TEST_CSV_1=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_csv_paths[0])")
TEST_CSV_2=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_csv_paths[1])")
TEST_ROOT_1=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_root_dirs[0])")
TEST_ROOT_2=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().test_root_dirs[1])")
OUTPUT_DIR=$(python -c "from code.config import InferenceConfig; print(InferenceConfig().base_experiments_dir)")

# Use BOTH test datasets (concatenated automatically by the Python script)
TEST_CSVS="$TEST_CSV_1 $TEST_CSV_2"
TEST_ROOTS="$TEST_ROOT_1 $TEST_ROOT_2"

# Checkpoint paths - TOP 3 WINNING MODELS from final results
# These are the best performing models from each architecture:
#   - ResNet50:        p5_resnet_alpha_0_5        (QWK=0.8068, focal_alpha=0.5)
#   - EfficientNet-B3: effnet_exp5_lower_lr       (QWK=0.7811, lower_lr)
#   - ConvNeXt-Tiny:   convnext_exp2_moderate_settings (QWK=0.7447, moderate_settings)

RESNET_CHECKPOINT="$PROJECT_ROOT/model_checkpoints/p5_resnet_alpha_0_5_20251028_113346/resnet50_best.pth"
EFFICIENTNET_CHECKPOINT="$PROJECT_ROOT/model_checkpoints/effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth"
CONVNEXT_CHECKPOINT="$PROJECT_ROOT/model_checkpoints/convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth"

echo "Using paths from config.py and model_checkpoints:"
echo "  Test CSV 1: $TEST_CSV_1"
echo "  Test Root 1: $TEST_ROOT_1"
echo "  Test CSV 2: $TEST_CSV_2"
echo "  Test Root 2: $TEST_ROOT_2"
echo "  Output Dir: $OUTPUT_DIR"
echo "  --> Both test sets will be combined for inference"
echo ""
echo "Using best model checkpoints:"
echo "  ResNet50: $RESNET_CHECKPOINT"
echo "  EfficientNet-B3: $EFFICIENTNET_CHECKPOINT"
echo "  ConvNeXt-Tiny: $CONVNEXT_CHECKPOINT"
echo ""

# Models to analyze (comment out models you don't want to analyze)
MODELS="efficientnet_b3 resnet50 convnext_tiny"

# CAM methods (options: gradcam gradcamplusplus)
CAM_METHODS="gradcam gradcamplusplus"

# Processing options
BATCH_SIZE=""  # Leave empty to use model default, or specify (e.g., 32)
NUM_WORKERS=8
HEATMAP_TRANSPARENCY=0.5

# ============================================================================
# RUN ANALYSIS FOR EACH MODEL (Sequential Execution)
# ============================================================================
# Since we have individual checkpoint paths, we'll run each model separately
# This is more robust than trying to auto-detect from different directories

echo ""
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

# Process ResNet50
if [[ "$MODELS" == *"resnet50"* ]]; then
    echo ""
    echo "=== Processing ResNet50 (CHAMPION MODEL, QWK=0.8068) ==="
    echo "Config: Phase 5, focal_alpha=0.5, weight_decay=0.01"
    echo "Checkpoint: $RESNET_CHECKPOINT"

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
    echo "=== Processing EfficientNet-B3 (RUNNER-UP, QWK=0.7811) ==="
    echo "Config: Phase 2, lower_lr (0.0001/0.001), focal_gamma=3.0, 90 epochs"
    echo "Checkpoint: $EFFICIENTNET_CHECKPOINT"

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

# Process ConvNeXt-Tiny
if [[ "$MODELS" == *"convnext_tiny"* ]]; then
    echo ""
    echo "=== Processing ConvNeXt-Tiny (THIRD PLACE, QWK=0.7447) ==="
    echo "Config: Phase 2, moderate settings, focal_gamma=2.5, 120 epochs"
    echo "Checkpoint: $CONVNEXT_CHECKPOINT"

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
    echo "  $OUTPUT_DIR/resnet50/multilayer_xai_{timestamp}/"
    echo "  $OUTPUT_DIR/efficientnet_b3/multilayer_xai_{timestamp}/"
    echo "  $OUTPUT_DIR/convnext_tiny/multilayer_xai_{timestamp}/"
    echo ""
    echo "Each directory contains:"
    echo "  - xai_individual_gradcam/         : Individual image visualizations (GradCAM)"
    echo "  - xai_individual_gradcamplusplus/ : Individual image visualizations (GradCAM++)"
    echo "  - average_heatmaps/               : Per-class average heatmaps (KEY OUTPUT)"
    echo "  - prediction_results.csv          : Predictions with probabilities"
    echo "  - confusion_matrix.png            : Confusion matrix visualization"
    echo "  - final_metrics.txt               : Accuracy, QWK, MAE, confusion matrix"
    echo ""
    echo "TOP 3 WINNING MODELS ANALYZED:"
    echo "  🥇 ResNet50:        QWK=0.8068 (p5_resnet_alpha_0_5)"
    echo "  🥈 EfficientNet-B3: QWK=0.7811 (effnet_exp5_lower_lr)"
    echo "  🥉 ConvNeXt-Tiny:   QWK=0.7447 (convnext_exp2_moderate_settings)"
    echo ""
else
    echo ""
    echo "✗ WARNING: Some models failed (see errors above)"
    echo "Exit code: $OVERALL_EXIT_CODE"
    echo ""
fi

exit $OVERALL_EXIT_CODE
