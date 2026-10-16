# Common environment for the Slurm jobs: loads cluster.env, modules, the virtual
# environment and PYTHONPATH. Sourced at the start of each job script:
#   PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
#   source "$PROJECT_ROOT/code/slurm_scripts/job_env.sh" || exit 1
# Set JOB_PROFILE=SUMMARY before sourcing to use SUMMARY_MODULES / SUMMARY_VENV_PATH.

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$PROJECT_ROOT/code" ]; then
    echo "ERROR: $PROJECT_ROOT is not the repository root - submit with code/slurm_scripts/submit.sh" >&2
    exit 1
fi
CODE_DIR="$PROJECT_ROOT/code"
source "$CODE_DIR/slurm_scripts/cluster_env.sh"

if [ -n "${JOB_PROFILE:-}" ]; then
    _modules="${JOB_PROFILE}_MODULES"; _venv="${JOB_PROFILE}_VENV_PATH"
    [ -n "${!_modules+x}" ] && MODULES="${!_modules}"
    [ -n "${!_venv+x}" ] && VENV_PATH="${!_venv}"
fi

if [ -n "${MODULES:-}" ]; then
    module purge
    module load $MODULES
fi
if [ -n "${VENV_PATH:-}" ]; then
    source "$VENV_PATH/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT:$CODE_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

echo "Project root: $PROJECT_ROOT"
echo "Modules:      ${MODULES:-none}"
echo "Venv:         ${VENV_PATH:-none}"
echo "Python:       $(command -v python) ($(python --version 2>&1))"
