"""
Local cluster settings for the Slurm script generators (hyperparam_search_phase*.py).

Values come from cluster.env in the repository root (gitignored; copy cluster.env.example).
The shell scripts in code/slurm_scripts/ read the same file, so there is one place for
account, partition, GPU type, modules and paths. Environment variables take precedence.

Author: Gerald Torgersen
SPDX-License-Identifier: MIT
"""

import os
import shlex
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_ENV = PROJECT_ROOT / "cluster.env"


@lru_cache(maxsize=None)
def load_cluster_env() -> dict:
    """Parse cluster.env (KEY=value lines, shell quoting and comments allowed)."""
    values = {}
    if CLUSTER_ENV.exists():
        for line in CLUSTER_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = " ".join(shlex.split(value, comments=True))
    else:
        print(f"WARNING: {CLUSTER_ENV} not found - copy cluster.env.example to cluster.env and edit it")
    for key in set(values) | {"SBATCH_ACCOUNT", "SBATCH_PARTITION", "SBATCH_GRES"}:
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def sbatch_lines(default_gres: str = "gpu:1") -> str:
    """#SBATCH lines for account, partition and GPU request, for a generated job script."""
    env = load_cluster_env()
    lines = []
    if env.get("SBATCH_ACCOUNT"):
        lines.append(f"#SBATCH --account={env['SBATCH_ACCOUNT']}")
    if env.get("SBATCH_PARTITION"):
        lines.append(f"#SBATCH --partition={env['SBATCH_PARTITION']}")
    lines.append(f"#SBATCH --gres={env.get('SBATCH_GRES') or default_gres}")
    return "\n".join(lines)


def project_root() -> str:
    """Repository root as a POSIX path (the generated scripts run on Linux)."""
    return PROJECT_ROOT.as_posix()
