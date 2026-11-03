#!/bin/bash
# Train a countfield run on a Perlmutter CPU node, from this repository's root:
#   sbatch scripts/slurm/train.sbatch PATH/TO/CONFIG.yaml
# Interpreter: this checkout's .venv/bin/python (override with COUNTFIELD_PYTHON).
# Defaults: account m2676, debug QOS, 15 min. Real runs, e.g.
#   sbatch -q regular -t 01:00:00 scripts/slurm/train.sbatch CONFIG.yaml
# Logs go to SLURM_LOGS (default outputs/slurm_logs, relative to the submit directory).
#SBATCH --account=m2676
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --job-name=countfield
#SBATCH --output=outputs/slurm_logs/%x-%j.out
set -euo pipefail
config=${1:?usage: sbatch scripts/slurm/train.sbatch CONFIG.yaml}
cd "$SLURM_SUBMIT_DIR"
threads=${TORCH_THREADS:-8}
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=$threads
python=${COUNTFIELD_PYTHON:-.venv/bin/python}
echo "host $(hostname), threads $threads, config $config, commit $(git rev-parse --short HEAD)"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$python" -m countfield train "$config"
