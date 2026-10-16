#!/bin/bash
# Submit a Slurm script with the local settings in cluster.env (account, partition, GPU, ...).
#
# Usage (from anywhere):
#   code/slurm_scripts/submit.sh code/slurm_scripts/slurm_batch_inference.sh [sbatch options]
#   DATA_DIR=/other/clips code/slurm_scripts/submit.sh code/slurm_scripts/slurm_batch_inference.sh
#
# The job is submitted from the repository root, so logs go to <repository>/logs/.
set -e
[ $# -ge 1 ] || { echo "Usage: $0 <slurm script> [sbatch options]" >&2; exit 1; }

SCRIPT="$(realpath "$1")"; shift
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/code/slurm_scripts/cluster_env.sh"

cd "$PROJECT_ROOT"
mkdir -p logs
sbatch "$@" "$SCRIPT"
