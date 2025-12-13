#!/bin/bash
#SBATCH --job-name=mednext_mitral          # Change if you want
#SBATCH --ntasks=8                         # 8 CPU cores (total)
#SBATCH --cpus-per-task=1                  # 1 thread per task (standard for PyTorch CPU training)
#SBATCH --time=24:00:00                    # 24 hours wall-clock time
#SBATCH --output=slurm-%j.out              # Output log file (%j = job ID)
#SBATCH --error=slurm-%j.err               # Error log file

echo "=== Job started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocated cores: $SLURM_NTASKS"

# Activate your conda/virtualenv (adjust path/name)
conda activate AML2

# Optional: copy data to local scratch for faster I/O (if your dataset is large)
# rsync -a /cluster/work/yourgroup/yourdata/ $TMPDIR/data/

# === YOUR TRAINING COMMAND HERE ===
# Example for nnU-Net style training with MedNeXt (adapt to your script)
python mednext_pipeline.py

# If you use the official MedNeXt/nnU-Net pipeline:
# nnUNet_train 2d MedNeXt TrainerName DatasetID fold --npz

echo "=== Job finished at $(date) ==="