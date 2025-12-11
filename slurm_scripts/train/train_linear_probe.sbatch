#!/bin/bash
#SBATCH --job-name=dinomite_train_linear
#SBATCH --output=../../experiments/logs/train_linear_%j.out
#SBATCH --error=../../experiments/logs/train_linear_%j.err
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=student-nvidia

# Load modules if needed (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate environment
cd "$SLURM_SUBMIT_DIR/../.."
source .venv/bin/activate 2>/dev/null || true

# Dataset (can be overridden via command line argument)
DATASET=${1:-"cifar10"}

# Run training
python scripts/train_linear_probe.py \
    --dataset "$DATASET" \
    --batch-size 32 \
    --learning-rate 1e-4 \
    --epochs 20 \
    --output-dir models \
    --model-name "${DATASET}_linear_classifier" \
    --device cuda

echo "Training completed"

