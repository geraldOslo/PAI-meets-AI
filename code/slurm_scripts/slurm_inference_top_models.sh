#!/bin/bash
#SBATCH --job-name=PAI_Top3_Inference
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
# (previously run on an A100: SBATCH_GRES=gpu:a100:1 code/slurm_scripts/submit.sh ...)
#SBATCH --output=logs/inference_top3_%j.out
#SBATCH --error=logs/inference_top3_%j.err

# ============================================================================
# PAI Classification Inference & GradCAM Generation - Top 3 Models
# ============================================================================
# This script runs inference with full GradCAM heatmap generation for the
# top performing model from each architecture:
#   1. ResNet50:        p5_resnet_alpha_0_5 (Test QWK: 0.8068)
#   2. EfficientNet-B3: effnet_exp5_lower_lr (Test QWK: 0.7811)
#   3. ConvNeXt-Tiny:   convnext_exp2_moderate_settings (Test QWK: 0.7447)
# Submit:  code/slurm_scripts/submit.sh code/slurm_scripts/slurm_inference_top_models.sh
# ============================================================================

echo "=================================================="
echo "PAI Top 3 Models Inference with GradCAM"
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="

# Modules, virtual environment and PYTHONPATH (from cluster.env); sets CODE_DIR
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
source "$PROJECT_ROOT/code/slurm_scripts/job_env.sh" || exit 1

# Base directory for model checkpoints
MODEL_CHECKPOINTS_BASE="${MODEL_CHECKPOINTS_BASE:-$PROJECT_ROOT/model_checkpoints}"

# Output directory for inference results
OUTPUT_DIR="$PROJECT_ROOT/inference_results/top_models_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

echo ""
echo "=================================================="
echo "RUNNING INFERENCE FOR TOP 3 MODELS"
echo "=================================================="

# ============================================================================
# MODEL 1: ResNet50 - p5_resnet_alpha_0_5
# ============================================================================
echo ""
echo "--- [1/3] ResNet50: p5_resnet_alpha_0_5 ---"
echo "Test QWK: 0.8068 | Focal Alpha: 0.5"

CMD1="python $CODE_DIR/test_inference/inference_gradcam.py \
  --resnet50 $MODEL_CHECKPOINTS_BASE/p5_resnet_alpha_0_5_20251028_113346/resnet50_best.pth \
  --models resnet50 \
  --output-dir $OUTPUT_DIR/resnet50_p5_alpha_0_5"

echo "Command: $CMD1"
eval $CMD1
EXIT_CODE_1=$?
echo "ResNet50 inference finished with exit code: $EXIT_CODE_1"

# ============================================================================
# MODEL 2: EfficientNet-B3 - effnet_exp5_lower_lr
# ============================================================================
echo ""
echo "--- [2/3] EfficientNet-B3: effnet_exp5_lower_lr ---"
echo "Test QWK: 0.7811 | Lower LR (0.0001/0.001)"

CMD2="python $CODE_DIR/test_inference/inference_gradcam.py \
  --efficientnet-b3 $MODEL_CHECKPOINTS_BASE/effnet_exp5_lower_lr_20251025_235232/efficientnet-b3_best.pth \
  --models efficientnet_b3 \
  --output-dir $OUTPUT_DIR/efficientnet_b3_exp5_lower_lr"

echo "Command: $CMD2"
eval $CMD2
EXIT_CODE_2=$?
echo "EfficientNet-B3 inference finished with exit code: $EXIT_CODE_2"

# ============================================================================
# MODEL 3: ConvNeXt-Tiny - convnext_exp2_moderate_settings
# ============================================================================
echo ""
echo "--- [3/3] ConvNeXt-Tiny: convnext_exp2_moderate_settings ---"
echo "Test QWK: 0.7447 | Moderate settings"

CMD3="python $CODE_DIR/test_inference/inference_gradcam.py \
  --convnext-tiny $MODEL_CHECKPOINTS_BASE/convnext_exp2_moderate_settings_20251026_061225/convnext-tiny_best.pth \
  --models convnext_tiny \
  --output-dir $OUTPUT_DIR/convnext_tiny_exp2_moderate"

echo "Command: $CMD3"
eval $CMD3
EXIT_CODE_3=$?
echo "ConvNeXt-Tiny inference finished with exit code: $EXIT_CODE_3"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "=================================================="
echo "ALL INFERENCE JOBS COMPLETED"
echo "=================================================="
echo "ResNet50 exit code:        $EXIT_CODE_1"
echo "EfficientNet-B3 exit code: $EXIT_CODE_2"
echo "ConvNeXt-Tiny exit code:   $EXIT_CODE_3"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Job Finished: $(date)"
echo "=================================================="

# Exit with non-zero if any model failed
if [ $EXIT_CODE_1 -ne 0 ] || [ $EXIT_CODE_2 -ne 0 ] || [ $EXIT_CODE_3 -ne 0 ]; then
    echo "WARNING: At least one inference job failed"
    exit 1
else
    echo "SUCCESS: All inference jobs completed successfully"
    exit 0
fi
