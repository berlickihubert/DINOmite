# SLURM Scripts for DINOmite

This directory contains SLURM batch scripts for running training, evaluation, and attacks on a cluster.

## Directory Structure

```
slurm_scripts/
├── train/          # Training scripts
│   └── train_linear_probe.sbatch
├── defense/        # Adversarial training scripts
│   ├── train_pgd_defense.sbatch
│   ├── train_trades_defense.sbatch
│   └── train_mart_defense.sbatch
├── attack/         # Attack evaluation scripts
│   ├── evaluate_fgsm.sbatch
│   ├── evaluate_pgd.sbatch
│   ├── evaluate_cw.sbatch
│   ├── evaluate_bim.sbatch
│   ├── evaluate_deepfoold.sbatch
│   ├── evaluate_autoattack.sbatch
│   └── evaluate_all_attacks.sbatch
└── eval/           # General evaluation scripts
    ├── evaluate_model.sbatch
    └── generate_poster_results.sbatch
```

## Usage

### Training Linear Probe

Train a linear classifier on top of DINOv3 features:

```bash
# CIFAR-10 (default)
sbatch slurm_scripts/train/train_linear_probe.sbatch

# Other datasets
sbatch slurm_scripts/train/train_linear_probe.sbatch gtsrb
sbatch slurm_scripts/train/train_linear_probe.sbatch tiny_imagenet
```

### Training with Defenses

Train models with adversarial defenses:

```bash
# PGD Adversarial Training
sbatch slurm_scripts/defense/train_pgd_defense.sbatch [dataset]

# TRADES
sbatch slurm_scripts/defense/train_trades_defense.sbatch [dataset]

# MART
sbatch slurm_scripts/defense/train_mart_defense.sbatch [dataset]

# Examples:
sbatch slurm_scripts/defense/train_pgd_defense.sbatch cifar10
sbatch slurm_scripts/defense/train_pgd_defense.sbatch gtsrb
```

### Evaluating Attacks

Evaluate a model against specific attacks:

```bash
# FGSM
sbatch slurm_scripts/attack/evaluate_fgsm.sbatch [model_path] [dataset]

# PGD
sbatch slurm_scripts/attack/evaluate_pgd.sbatch [model_path] [dataset]

# BIM
sbatch slurm_scripts/attack/evaluate_bim.sbatch [model_path] [dataset]

# DeepFool
sbatch slurm_scripts/attack/evaluate_deepfool.sbatch [model_path] [dataset]

# C&W
sbatch slurm_scripts/attack/evaluate_cw.sbatch [model_path] [dataset]

# AutoAttack
sbatch slurm_scripts/attack/evaluate_autoattack.sbatch [model_path] [dataset]

# All attacks
sbatch slurm_scripts/attack/evaluate_all_attacks.sbatch [model_path] [dataset]
```

### Comprehensive Evaluation

Evaluate a model against all attacks and generate results:

```bash
sbatch slurm_scripts/eval/evaluate_model.sbatch [model_path] [dataset]
```

### Generate Poster Results

Generate all plots and tables for the poster:

```bash
sbatch slurm_scripts/eval/generate_poster_results.sbatch
```

## Customizing Scripts

Before using these scripts, you may need to:

1. **Adjust module loading**: Uncomment and modify the `module load` lines for your cluster
2. **Set environment activation**: Modify the `source .venv/bin/activate` line if using a different environment
3. **Adjust resource requirements**: Modify `--time`, `--mem`, `--gres` based on your needs
4. **Set default model paths**: Modify the `MODEL_PATH` variable in attack/eval scripts

## Example Workflow

1. **Train baseline model**:
   ```bash
   sbatch slurm_scripts/train/train_linear_probe.sbatch
   ```

2. **Train robust models**:
   ```bash
   sbatch slurm_scripts/defense/train_pgd_defense.sbatch
   sbatch slurm_scripts/defense/train_trades_defense.sbatch
   sbatch slurm_scripts/defense/train_mart_defense.sbatch
   ```

3. **Evaluate all models**:
   ```bash
   sbatch slurm_scripts/eval/evaluate_model.sbatch models/cifar10_linear_classifier_best.pth cifar10
   sbatch slurm_scripts/eval/evaluate_model.sbatch models/pgd_cifar10_best.pth cifar10
   sbatch slurm_scripts/eval/evaluate_model.sbatch models/trades_cifar10_best.pth cifar10
   sbatch slurm_scripts/eval/evaluate_model.sbatch models/mart_cifar10_best.pth cifar10
   ```

4. **Generate poster results**:
   ```bash
   sbatch slurm_scripts/eval/generate_poster_results.sbatch
   ```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View job output:
```bash
tail -f experiments/logs/train_linear_<job_id>.out
```

Cancel a job:
```bash
scancel <job_id>
```

## Notes

- All scripts assume the project root is accessible from the working directory
- Logs are saved to `experiments/logs/`
- Results are saved to `results/`
- Models are saved to `models/`
- Scripts use `scripts/` directory for Python scripts (not `src/`)
