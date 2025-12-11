#!/bin/bash
#SBATCH --job-name=dinomite_poster_results
#SBATCH --output=../../experiments/logs/poster_results_%j.out
#SBATCH --error=../../experiments/logs/poster_results_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --partition=student-short

# Load modules if needed
# module load python/3.10

# Activate environment
cd "$SLURM_SUBMIT_DIR/../.."
source .venv/bin/activate 2>/dev/null || true

# Run poster results generation
python scripts/generate_poster_results.py \
    --results-dir results \
    --output-dir results/poster

echo "Poster results generation completed"

