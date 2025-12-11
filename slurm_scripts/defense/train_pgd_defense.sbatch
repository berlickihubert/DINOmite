#!/bin/bash
#SBATCH --job-name=dinomite_pgd_defense
#SBATCH --output=../../experiments/logs/pgd_defense_%j.out
#SBATCH --error=../../experiments/logs/pgd_defense_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=student-nvidia

# Load modules if needed
# module load cuda/11.8
# module load python/3.10

# Activate environment
cd "$SLURM_SUBMIT_DIR/../.."
source .venv/bin/activate 2>/dev/null || true

# Dataset (can be overridden via command line argument)
DATASET=${1:-"cifar10"}

# Run PGD adversarial training
python scripts/train_defense.py \
    --defense pgd \
    --dataset "$DATASET" \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --epochs 20 \
    --eps 0.03137 \
    --alpha 0.00784 \
    --steps 7 \
    --output-dir models \
    --model-name "pgd_${DATASET}" \
    --device cuda

echo "PGD defense training completed"

