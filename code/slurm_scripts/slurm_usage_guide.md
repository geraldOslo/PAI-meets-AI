# Slurm Scripts Usage Guide

## Overview

Three separate Slurm scripts for parallel training:
- `submit_resnet50.sh` - Fastest model (2.5-3 hours)
- `submit_convnext_tiny.sh` - Medium speed (3.5-4.5 hours)
- `submit_efficientnet_b3.sh` - Slowest model (7-9 hours)

**Advantages:**
- Can run in parallel if multiple GPUs available
- Smaller jobs = faster queue times
- One model failing doesn't affect others
- Easy to test individual models

---

## Quick Start: Test Run (1 Epoch)

### Step 1: Update Email and Paths

Edit all three scripts and change:

```bash
#SBATCH --mail-user=your.email@uio.no  # Line 13

# And verify these paths (around line 60-62):
DATA_CSV="/fp/projects01/ec192/data/endo-radiographs/full_dataset/data.csv"
DATA_ROOT="/fp/projects01/ec192/data/endo-radiographs/full_dataset"
```

### Step 2: Ensure EPOCHS=1 (Already Set for Testing)

Each script has this at the top:
```bash
EPOCHS=1              # Set to 1 for testing, 50 for full training
```

This is already set to 1, so you're ready for testing.

### Step 3: Submit Test Jobs

```bash
# Create logs directory first
mkdir -p logs

# Submit all three test jobs
sbatch submit_resnet50.sh
sbatch submit_convnext_tiny.sh
sbatch submit_efficientnet_b3.sh
```

### Step 4: Monitor Jobs

```bash
# Check queue status
squeue -u $USER

# Watch continuously (updates every 10 seconds)
watch -n 10 'squeue -u $USER'

# Check specific job details
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
```

### Step 5: View Output in Real-Time

```bash
# Find your job IDs
squeue -u $USER

# Tail the output files (replace JOBID)
tail -f logs/resnet50_<JOBID>.out
tail -f logs/convnext_tiny_<JOBID>.out
tail -f logs/efficientnet_b3_<JOBID>.out
```

---

## Expected Test Run Times (1 Epoch)

| Model | Time | Output File |
|-------|------|-------------|
| ResNet50 | 5-10 min | logs/resnet50_JOBID.out |
| ConvNeXt-Tiny | 5-10 min | logs/convnext_tiny_JOBID.out |
| EfficientNet-B3 | 10-15 min | logs/efficientnet_b3_JOBID.out |

**Total parallel time: 10-15 minutes** (if all get GPUs simultaneously)
**Total sequential time: 20-35 minutes** (if running one at a time)

---

## What to Check After Test Runs

### 1. Check Job Completed Successfully

```bash
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode

# State should be: COMPLETED
# ExitCode should be: 0:0
```

### 2. Check Output Files

```bash
# Look for errors
grep -i error logs/resnet50_*.err
grep -i "traceback" logs/resnet50_*.out

# Check if training completed
grep -i "training completed" logs/resnet50_*.out
```

### 3. Check Results Were Created

```bash
# ResNet50
ls -lh /fp/projects01/ec192/pai_experiments/resnet50/
cat /fp/projects01/ec192/pai_experiments/resnet50/*/experiment_summary.json

# ConvNeXt-Tiny
ls -lh /fp/projects01/ec192/pai_experiments/convnext_tiny/

# EfficientNet-B3
ls -lh /fp/projects01/ec192/pai_experiments/efficientnet_b3/
```

### 4. Verify Oversampling Worked

Look in the output logs for:
```
Class distribution in subset:
  PAI 1: XXXX samples (weight: X.XXXX)
  PAI 2: XXXX samples (weight: X.XXXX)
  ...
  PAI 5: XXX samples (weight: X.XXXX)
```

Weights should be higher for minority classes (PAI 5 should have highest weight).

---

## Full Training Run (50 Epochs)

**After test runs succeed:**

### Step 1: Edit EPOCHS Variable in Each Script

Change line ~20 in each script from:
```bash
EPOCHS=1              # Set to 1 for testing, 50 for full training
```

To:
```bash
EPOCHS=50             # Set to 1 for testing, 50 for full training
```

### Step 2: Submit Full Training Jobs

```bash
# Submit all three
sbatch submit_resnet50.sh
sbatch submit_convnext_tiny.sh
sbatch submit_efficientnet_b3.sh
```

### Step 3: Expected Runtimes

| Model | Time Allocation | Expected Completion |
|-------|----------------|---------------------|
| ResNet50 | 4 hours | 2.5-3 hours (with early stopping) |
| ConvNeXt-Tiny | 6 hours | 3.5-4.5 hours (with early stopping) |
| EfficientNet-B3 | 10 hours | 7-9 hours (with early stopping) |

**If all run in parallel: ~9 hours total**
**If sequential: ~15-16 hours total**

---

## Monitoring Long-Running Jobs

### Check Progress

```bash
# See current state
squeue -u $USER

# Check how long it's been running
sacct -j <JOBID> --format=JobID,Elapsed,State

# See training progress (updated live)
tail -f logs/resnet50_<JOBID>.out

# Look for epoch completion messages
grep "Epoch" logs/resnet50_<JOBID>.out | tail -20
```

### Check for Early Stopping

```bash
# Look for this message in output
grep "Early stopping" logs/resnet50_*.out
```

If early stopping triggers at epoch 35, job will finish early (saving time and compute credits).

### GPU Usage (While Job Running)

```bash
# SSH to the node (get node name from squeue)
ssh <nodename>

# Check GPU utilization
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Job Stuck in Queue (State: PENDING)

```bash
# Check why it's pending
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
# Look at the NODELIST(REASON) column
```

**Common reasons:**
- `Resources`: Waiting for GPU
- `Priority`: Other jobs have higher priority
- `QOSMaxWallDuration`: Time limit exceeded

**Solutions:**
- Wait (especially for A100 GPUs, may take hours)
- Try without `--gres=gpu:a100:1` (any GPU)
- Request less time for test runs

### Job Failed (State: FAILED)

```bash
# Check exit code
sacct -j <JOBID> --format=JobID,State,ExitCode

# Check error log
cat logs/resnet50_<JOBID>.err

# Check what happened in output
tail -100 logs/resnet50_<JOBID>.out
```

**Common issues:**
- Import errors: Check PYTHONPATH
- File not found: Check DATA_CSV and DATA_ROOT paths
- CUDA OOM: GPU ran out of memory
- Module not found: Virtual environment issue

### Job Cancelled (State: CANCELLED)

Check if you cancelled it or if it hit time limit:
```bash
sacct -j <JOBID> --format=JobID,State,Elapsed,TimeLimit
```

If `Elapsed ≈ TimeLimit`, it ran out of time. Increase time allocation.

### Import Error: data_utils or train_utils

Add to the script before `python train_pai_simple_split.py`:
```bash
export PYTHONPATH=/fp/projects01/ec192/code:$PYTHONPATH
```

---

## Cancelling Jobs

### Cancel All Your Jobs
```bash
scancel -u $USER
```

### Cancel Specific Job
```bash
scancel <JOBID>
```

### Cancel by Job Name
```bash
scancel -n pai_resnet50
```

---

## Results Location

After completion, check:

```bash
/fp/projects01/ec192/pai_experiments/
├── resnet50/
│   └── pai_simple_split_experiment_TIMESTAMP/
│       ├── training.log
│       ├── resnet50_best.pth
│       ├── resnet50_history.json
│       └── experiment_summary.json
├── convnext_tiny/
│   └── pai_simple_split_experiment_TIMESTAMP/
└── efficientnet_b3/
    └── pai_simple_split_experiment_TIMESTAMP/
```

**Key files:**
- `experiment_summary.json`: Final QWK, accuracy, F1 scores
- `*_history.json`: Training/validation metrics per epoch
- `*_best.pth`: Model checkpoint with best validation QWK
- `training.log`: Complete training log

---

## Comparing Results

After all three complete:

```bash
# Quick comparison
for model in resnet50 convnext_tiny efficientnet_b3; do
    echo "=== $model ==="
    grep -A 5 "best_metrics" /fp/projects01/ec192/pai_experiments/$model/*/experiment_summary.json
done
```

Look for:
- **QWK** (Quadratic Weighted Kappa): Primary metric, higher is better
- **Accuracy**: Overall correctness
- **F1**: Weighted F1 score
- **MAE**: Mean absolute error (lower is better)

---

## Email Notifications

You'll receive emails for:
- **BEGIN**: Job started
- **END**: Job completed successfully
- **FAIL**: Job failed

**Note:** Update the email address in all three scripts:
```bash
#SBATCH --mail-user=your.email@uio.no
```

Or comment out if you don't want emails:
```bash
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=your.email@uio.no
```

---

## Running Only One or Two Models

Don't need all three? Just submit the ones you want:

```bash
# Only ResNet50 and ConvNeXt
sbatch submit_resnet50.sh
sbatch submit_convnext_tiny.sh

# Just ResNet50 for quick test
sbatch submit_resnet50.sh
```

---

## Reusing Scripts for Different Experiments

Want to try different hyperparameters? Easy to modify:

```bash
# Copy and modify
cp submit_resnet50.sh submit_resnet50_lr1e4.sh

# Edit the copy to add custom parameters
# Change the python command to:
python train_pai_simple_split.py \
    ... existing params ... \
    --learning_rate 1e-4

# Then submit
sbatch submit_resnet50_lr1e4.sh
```

---

## Summary: Test → Full Training Workflow

### Phase 1: Test (30 minutes total)
```bash
# 1. Verify EPOCHS=1 in all scripts
# 2. Update email and paths
# 3. Submit test jobs
sbatch submit_resnet50.sh
sbatch submit_convnext_tiny.sh
sbatch submit_efficientnet_b3.sh

# 4. Wait 10-15 minutes
# 5. Check results exist and no errors
```

### Phase 2: Full Training (if tests pass)
```bash
# 1. Change EPOCHS=1 to EPOCHS=50 in all scripts
# 2. Submit full training jobs
sbatch submit_resnet50.sh
sbatch submit_convnext_tiny.sh
sbatch submit_efficientnet_b3.sh

# 3. Wait 7-9 hours (or check tomorrow)
# 4. Compare results and pick best model
```

---

## Quick Reference Commands

```bash
# Submit jobs
sbatch submit_resnet50.sh

# Check status
squeue -u $USER

# View progress
tail -f logs/resnet50_*.out

# Cancel job
scancel <JOBID>

# Check completed jobs
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed

# View results
cat /fp/projects01/ec192/pai_experiments/resnet50/*/experiment_summary.json
```

---

**You're ready to test!** Start with the 1-epoch test runs to verify everything works before committing to the full 50-epoch training.
