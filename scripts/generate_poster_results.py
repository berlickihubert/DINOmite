#!/usr/bin/env python3
"""
Generate all results, plots, and tables for the poster.

This script collects results from all experiments and generates
publication-ready figures and tables for LaTeX poster.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


def load_results(results_dir):
    """Load all result files."""
    results = {}

    # Find all result JSON files
    result_files = glob.glob(os.path.join(results_dir, "**", "*_results.json"), recursive=True)

    for file_path in result_files:
        with open(file_path, "r") as f:
            data = json.load(f)
            model_name = os.path.basename(os.path.dirname(file_path))
            if model_name not in results:
                results[model_name] = []
            results[model_name].append(data)

    return results


def plot_robustness_comparison(results_dict, save_path):
    """Plot robustness comparison across models."""
    fig, ax = plt.subplots(figsize=(12, 8))

    epsilons = [0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

    for model_name, model_results in results_dict.items():
        accuracies = []
        for eps in epsilons:
            # Find accuracy for this epsilon
            acc = None
            for result in model_results:
                if "attack_results" in result:
                    if eps in result["attack_results"]:
                        acc = result["attack_results"][eps]["accuracy"]
                        break
                elif "results" in result:
                    # Handle different result formats
                    for attack_name, attack_results in result["results"].items():
                        if eps in attack_results:
                            acc = attack_results[eps]["accuracy"]
                            break

            accuracies.append(acc if acc is not None else 0)

        ax.plot(epsilons, accuracies, marker="o", label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel("Epsilon (ε)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title("Adversarial Robustness Comparison", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved robustness comparison plot to {save_path}")


def plot_attack_comparison(results_dict, save_path):
    """Plot attack comparison for a single model."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get first model's results
    model_name = list(results_dict.keys())[0]
    model_results = results_dict[model_name]

    # Collect attack results
    attack_results = defaultdict(dict)
    for result in model_results:
        if "results" in result:
            for attack_name, attack_data in result["results"].items():
                for eps, metrics in attack_data.items():
                    if eps not in attack_results[attack_name]:
                        attack_results[attack_name][eps] = []
                    attack_results[attack_name][eps].append(metrics["accuracy"])

    # Plot each attack
    epsilons = sorted([0, 1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255])
    for attack_name, attack_data in attack_results.items():
        accuracies = []
        for eps in epsilons:
            if eps in attack_data:
                acc = np.mean(attack_data[eps])
            else:
                acc = None
            accuracies.append(acc)

        # Filter out None values for plotting
        valid_eps = [eps for eps, acc in zip(epsilons, accuracies) if acc is not None]
        valid_acc = [acc for acc in accuracies if acc is not None]

        if valid_eps:
            ax.plot(valid_eps, valid_acc, marker="o", label=attack_name.upper(), linewidth=2, markersize=8)

    ax.set_xlabel("Epsilon (ε)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=14, fontweight="bold")
    ax.set_title(f"Attack Comparison: {model_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved attack comparison plot to {save_path}")


def generate_results_table(results_dict, save_path):
    """Generate LaTeX table of results."""
    rows = []

    for model_name, model_results in results_dict.items():
        for result in model_results:
            if "results" in result:
                for attack_name, attack_data in result["results"].items():
                    # Get accuracy at epsilon=8/255
                    eps_key = 8 / 255
                    if eps_key in attack_data:
                        acc = attack_data[eps_key]["accuracy"]
                    else:
                        # Get first available epsilon
                        eps_keys = sorted([k for k in attack_data.keys() if isinstance(k, (int, float))])
                        if eps_keys:
                            acc = attack_data[eps_keys[-1]]["accuracy"]
                        else:
                            acc = None

                    rows.append(
                        {
                            "Model": model_name,
                            "Attack": attack_name.upper(),
                            "Accuracy (%)": f"{acc:.2f}" if acc is not None else "N/A",
                        }
                    )

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = save_path.replace(".tex", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table (CSV) to {csv_path}")

    # Save LaTeX
    tex_path = save_path if save_path.endswith(".tex") else save_path + ".tex"
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.2f", escape=False))
    print(f"Saved results table (LaTeX) to {tex_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate poster results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Directory containing result files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "poster"),
        help="Directory to save poster results",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    print("Loading results...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found! Please run evaluations first.")
        return

    print(f"Found results for {len(results)} models")

    # Generate plots
    print("\nGenerating plots...")

    # Robustness comparison
    comparison_plot = os.path.join(args.output_dir, "robustness_comparison.png")
    plot_robustness_comparison(results, comparison_plot)

    # Attack comparison
    attack_plot = os.path.join(args.output_dir, "attack_comparison.png")
    plot_attack_comparison(results, attack_plot)

    # Generate table
    print("\nGenerating tables...")
    table_path = os.path.join(args.output_dir, "results_table.tex")
    df = generate_results_table(results, table_path)

    print("\n" + "=" * 60)
    print("POSTER RESULTS GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    print(f"  - {comparison_plot}")
    print(f"  - {attack_plot}")
    print(f"  - {table_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
