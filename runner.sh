#!/bin/bash
#SBATCH --job-name=mednext_mitral
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

echo "=== Job started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated cores: $SLURM_NTASKS"

python mednext_pipeline.py

echo "=== Job finished at $(date) ==="