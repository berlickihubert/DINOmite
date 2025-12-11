#!/bin/bash
#SBATCH --job-name=dinomite_eval
#SBATCH --output=../../experiments/logs/eval_%j.out
#SBATCH --error=../../experiments/logs/eval_%j.err
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

# Model path and dataset (can be overridden via command line arguments)
MODEL_PATH=${1:-"models/cifar10_linear_classifier_best.pth"}
DATASET=${2:-"cifar10"}

# Run comprehensive evaluation against all attacks
python scripts/evaluate_all_attacks.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --epsilons 0 0.00392 0.00784 0.01569 0.03137 0.06275 \
    --batch-size 32 \
    --output-dir results \
    --device cuda

echo "Model evaluation completed"

