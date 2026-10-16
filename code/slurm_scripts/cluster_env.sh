# Load cluster.env (local settings, gitignored) into the environment.
# Sourced by submit.sh and job_env.sh. Variables already set in the environment win,
# so  DATA_DIR=/other/clips code/slurm_scripts/submit.sh ...  overrides the file.

pai_load_env() {   # export KEY=value lines from file $1 unless KEY is already set
    local line key value
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line#"${line%%[![:space:]]*}"}"
        case "$line" in ''|\#*) continue ;; esac
        line="${line#export }"
        key="${line%%=*}"
        value="${line#*=}"
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [ -n "${!key+x}" ] && continue
        value="${value%%[[:space:]]#*}"
        eval "value=$value"
        export "$key=$value"
    done < "$1"
}

if [ -f "$PROJECT_ROOT/cluster.env" ]; then
    pai_load_env "$PROJECT_ROOT/cluster.env"
else
    echo "WARNING: $PROJECT_ROOT/cluster.env not found - copy cluster.env.example to cluster.env and edit it" >&2
fi
