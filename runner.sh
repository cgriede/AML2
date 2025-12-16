#!/bin/bash
#SBATCH --job-name=mednext_mitral
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=29:00:00
#SBATCH --mem-per-cpu=2G

TOTAL_CPUS=$(($SLURM_NTASKS * $SLURM_CPUS_PER_TASK))

export        OMP_NUM_THREADS=${TOTAL_CPUS}
export        MKL_NUM_THREADS=${TOTAL_CPUS}
export   OPENBLAS_NUM_THREADS=${TOTAL_CPUS}  # just in case
export    NUMEXPR_NUM_THREADS=${TOTAL_CPUS}
export VECLIB_MAXIMUM_THREADS=${TOTAL_CPUS}


echo "=== Job started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated cores: $SLURM_NTASKS"

python mednext_pipeline.py
