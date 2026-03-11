#!/usr/bin/env python3
"""Analysis script for Paper B experiment results."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results(path: str) -> List[Dict]:
    """Load JSONL results file."""
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> tuple:
    """Compute confidence interval for mean."""
    if not data or len(data) < 2:
        return (None, None, None)

    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return (mean, ci[0], ci[1])


def analyze_stability(results: List[Dict]) -> pd.DataFrame:
    """Analyze stability metrics by model and condition."""
    rows = []

    for result in results:
        if not result.get("success"):
            continue

        model = result["model_id"]
        condition = result["condition"]
        task = result["task_family"]

        # Get adherence score
        adherence = result.get("mean_adherence")

        # Get stability metrics
        stability = result.get("stability_metrics", {})

        row = {
            "model": model,
            "condition": condition,
            "task": task,
            "adherence": adherence,
            "override_resistance": stability.get("override_resistance"),
            "trial": result["trial_num"]
        }

        # Contradiction update latency
        if stability.get("contradiction_update"):
            row["immediate_update"] = stability["contradiction_update"].get("immediate_update")

        # Persistence
        if stability.get("persistence"):
            row["persisted"] = stability["persistence"].get("persisted")
            row["persistence_turns"] = stability["persistence"].get("persistence_turns")

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_calibration(results: List[Dict]) -> pd.DataFrame:
    """Analyze calibration metrics for T2 tasks."""
    rows = []

    for result in results:
        if not result.get("success"):
            continue
        if result["task_family"] != "T2":
            continue

        model = result["model_id"]
        condition = result["condition"]

        cal = result.get("calibration_metrics", {})

        rows.append({
            "model": model,
            "condition": condition,
            "accuracy": cal.get("accuracy"),
            "mean_confidence": cal.get("mean_confidence"),
            "brier_score": cal.get("brier_score"),
            "log_score": cal.get("log_score"),
            "overconfidence": cal.get("overconfidence"),
            "n_valid": cal.get("n_valid"),
            "trial": result["trial_num"]
        })

    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame, group_cols: List[str], value_col: str) -> pd.DataFrame:
    """Compute mean and CI by group."""
    groups = df.groupby(group_cols)[value_col]

    summary = groups.agg(['mean', 'std', 'count']).reset_index()
    summary['ci_low'] = summary['mean'] - 1.96 * summary['std'] / np.sqrt(summary['count'])
    summary['ci_high'] = summary['mean'] + 1.96 * summary['std'] / np.sqrt(summary['count'])

    return summary


def run_significance_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """Run statistical significance tests."""
    tests = {}

    # C2 vs C0 (telemetry effect)
    c0 = df[df['condition'] == 'C0']['adherence'].dropna()
    c2 = df[df['condition'] == 'C2']['adherence'].dropna()
    if len(c0) > 1 and len(c2) > 1:
        stat, p = stats.mannwhitneyu(c0, c2, alternative='two-sided')
        tests['C2_vs_C0'] = {'statistic': stat, 'p_value': p, 'n_c0': len(c0), 'n_c2': len(c2)}

    # C4 vs C2 (override effect)
    c4 = df[df['condition'] == 'C4']['adherence'].dropna()
    if len(c2) > 1 and len(c4) > 1:
        stat, p = stats.mannwhitneyu(c2, c4, alternative='two-sided')
        tests['C4_vs_C2'] = {'statistic': stat, 'p_value': p, 'n_c2': len(c2), 'n_c4': len(c4)}

    # C1 vs C2 (sham vs real)
    c1 = df[df['condition'] == 'C1']['adherence'].dropna()
    if len(c1) > 1 and len(c2) > 1:
        stat, p = stats.mannwhitneyu(c1, c2, alternative='two-sided')
        tests['C2_vs_C1'] = {'statistic': stat, 'p_value': p, 'n_c1': len(c1), 'n_c2': len(c2)}

    return tests


def plot_stability_by_condition(df: pd.DataFrame, output_dir: Path):
    """Plot adherence by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute summary
    summary = df.groupby('condition')['adherence'].agg(['mean', 'std', 'count']).reset_index()
    summary['ci'] = 1.96 * summary['std'] / np.sqrt(summary['count'])

    # Sort by condition code
    condition_order = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    summary['condition'] = pd.Categorical(summary['condition'], categories=condition_order, ordered=True)
    summary = summary.sort_values('condition')

    # Bar plot
    bars = ax.bar(summary['condition'], summary['mean'], yerr=summary['ci'], capsize=5)

    # Color by condition type
    colors = {
        'C0': '#888888',  # NONE - gray
        'C1': '#FFA500',  # SHAM - orange
        'C2': '#2196F3',  # CONSISTENT - blue
        'C3': '#9C27B0',  # CONTRADICTORY - purple
        'C4': '#F44336',  # OVERRIDE - red
        'C5': '#4CAF50',  # PERSISTENCE - green
    }
    for bar, cond in zip(bars, summary['condition']):
        bar.set_color(colors.get(cond, '#888888'))

    ax.set_xlabel('Condition')
    ax.set_ylabel('Mean Adherence Score')
    ax.set_title('State Adherence by Experimental Condition')
    ax.set_ylim(0, 1.1)

    # Add condition labels
    condition_names = {
        'C0': 'None', 'C1': 'Sham', 'C2': 'Consistent',
        'C3': 'Contradictory', 'C4': 'Override', 'C5': 'Persistence'
    }
    ax.set_xticklabels([f"{c}\n({condition_names.get(c, '')})" for c in summary['condition']])

    plt.tight_layout()
    plt.savefig(output_dir / 'stability_by_condition.png', dpi=150)
    plt.close()


def plot_calibration_by_condition(df: pd.DataFrame, output_dir: Path):
    """Plot calibration metrics by condition."""
    if df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    condition_order = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    # Brier score
    ax = axes[0]
    summary = df.groupby('condition')['brier_score'].agg(['mean', 'std', 'count']).reset_index()
    summary['ci'] = 1.96 * summary['std'] / np.sqrt(summary['count'])
    summary['condition'] = pd.Categorical(summary['condition'], categories=condition_order, ordered=True)
    summary = summary.sort_values('condition').dropna(subset=['mean'])

    if not summary.empty:
        ax.bar(summary['condition'], summary['mean'], yerr=summary['ci'], capsize=5, color='steelblue')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Brier Score (lower is better)')
        ax.set_title('Calibration: Brier Score')

    # Overconfidence
    ax = axes[1]
    summary = df.groupby('condition')['overconfidence'].agg(['mean', 'std', 'count']).reset_index()
    summary['ci'] = 1.96 * summary['std'] / np.sqrt(summary['count'])
    summary['condition'] = pd.Categorical(summary['condition'], categories=condition_order, ordered=True)
    summary = summary.sort_values('condition').dropna(subset=['mean'])

    if not summary.empty:
        colors = ['green' if x < 0 else 'red' for x in summary['mean']]
        ax.bar(summary['condition'], summary['mean'], yerr=summary['ci'], capsize=5, color=colors)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Condition')
        ax.set_ylabel('Overconfidence (mean_p - accuracy)')
        ax.set_title('Calibration: Overconfidence Index')

    # Accuracy vs Confidence
    ax = axes[2]
    for cond in df['condition'].unique():
        cond_df = df[df['condition'] == cond]
        ax.scatter(cond_df['mean_confidence'], cond_df['accuracy'], label=cond, alpha=0.6)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Mean Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Confidence vs Accuracy')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_by_condition.png', dpi=150)
    plt.close()


def plot_reliability_diagram(results: List[Dict], output_dir: Path):
    """Plot reliability diagram from calibration bins."""
    # Aggregate bins across all T2 results
    all_bins = defaultdict(lambda: {'n': 0, 'correct': 0, 'conf_sum': 0})

    for result in results:
        if result.get("task_family") != "T2":
            continue
        if not result.get("success"):
            continue

        cal = result.get("calibration_metrics", {})
        bins = cal.get("calibration_bins", [])

        for b in bins:
            bin_key = (b['bin_start'], b['bin_end'])
            if b['accuracy'] is not None:
                all_bins[bin_key]['n'] += b['n']
                all_bins[bin_key]['correct'] += int(b['accuracy'] * b['n'])
                all_bins[bin_key]['conf_sum'] += b['mean_confidence'] * b['n'] if b['mean_confidence'] else 0

    # Compute final bins
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for (start, end), data in sorted(all_bins.items()):
        if data['n'] > 0:
            bin_centers.append((start + end) / 2)
            bin_accuracies.append(data['correct'] / data['n'])
            bin_counts.append(data['n'])

    if not bin_centers:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Bar plot
    width = 0.08
    ax.bar(bin_centers, bin_accuracies, width=width, alpha=0.7, label='Model')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Add counts as text
    for x, y, n in zip(bin_centers, bin_accuracies, bin_counts):
        ax.annotate(f'n={n}', (x, y + 0.02), ha='center', fontsize=8)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Diagram (All T2 Trials)')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_dir / 'reliability_diagram.png', dpi=150)
    plt.close()


def generate_summary_table(stability_df: pd.DataFrame, calibration_df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary table."""
    rows = []

    conditions = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    condition_names = {
        'C0': 'None', 'C1': 'Sham', 'C2': 'Consistent',
        'C3': 'Contradictory', 'C4': 'Override', 'C5': 'Persistence'
    }

    for cond in conditions:
        cond_stab = stability_df[stability_df['condition'] == cond]
        cond_cal = calibration_df[calibration_df['condition'] == cond] if not calibration_df.empty else pd.DataFrame()

        row = {
            'Condition': cond,
            'Name': condition_names.get(cond, ''),
            'N_Trials': len(cond_stab),
            'Mean_Adherence': cond_stab['adherence'].mean() if len(cond_stab) > 0 else None,
            'Std_Adherence': cond_stab['adherence'].std() if len(cond_stab) > 0 else None,
        }

        if not cond_cal.empty:
            row['Mean_Brier'] = cond_cal['brier_score'].mean()
            row['Mean_Accuracy'] = cond_cal['accuracy'].mean()
            row['Overconfidence'] = cond_cal['overconfidence'].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def print_executive_summary(
    stability_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    tests: Dict[str, Any]
):
    """Print executive summary to stdout."""
    print("\n" + "="*70)
    print("EXECUTIVE RESULTS SUMMARY")
    print("="*70)

    print("\n1. STABILITY METRICS")
    print("-"*40)

    for cond in ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']:
        cond_df = stability_df[stability_df['condition'] == cond]
        if len(cond_df) > 0:
            mean_adh = cond_df['adherence'].mean()
            std_adh = cond_df['adherence'].std()
            n = len(cond_df)
            print(f"  {cond}: adherence = {mean_adh:.3f} ± {std_adh:.3f} (n={n})")

    print("\n2. SIGNIFICANCE TESTS")
    print("-"*40)

    for test_name, test_result in tests.items():
        p = test_result['p_value']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {test_name}: p = {p:.4f} {sig}")

    if not calibration_df.empty:
        print("\n3. CALIBRATION METRICS (T2)")
        print("-"*40)

        for cond in calibration_df['condition'].unique():
            cond_df = calibration_df[calibration_df['condition'] == cond]
            brier = cond_df['brier_score'].mean()
            overconf = cond_df['overconfidence'].mean()
            acc = cond_df['accuracy'].mean()
            print(f"  {cond}: Brier={brier:.3f}, Overconf={overconf:+.3f}, Acc={acc:.3f}")

    print("\n4. KEY FINDINGS")
    print("-"*40)

    # C2 vs C0 effect
    if 'C2_vs_C0' in tests:
        p = tests['C2_vs_C0']['p_value']
        c0_mean = stability_df[stability_df['condition'] == 'C0']['adherence'].mean()
        c2_mean = stability_df[stability_df['condition'] == 'C2']['adherence'].mean()
        diff = c2_mean - c0_mean if c0_mean and c2_mean else 0
        effect = "SIGNIFICANT" if p < 0.05 else "NOT SIGNIFICANT"
        print(f"  Telemetry Effect (C2 vs C0): Δ={diff:+.3f}, p={p:.4f} - {effect}")

    # Override resistance
    if 'C4_vs_C2' in tests:
        p = tests['C4_vs_C2']['p_value']
        c4_mean = stability_df[stability_df['condition'] == 'C4']['adherence'].mean()
        resistance = "RESISTANT" if c4_mean and c4_mean > 0.5 else "COMPLIANT"
        print(f"  Override Resistance (C4): adherence={c4_mean:.3f} - {resistance}")

    # Persistence
    c5_df = stability_df[stability_df['condition'] == 'C5']
    if 'persisted' in c5_df.columns:
        persist_rate = c5_df['persisted'].mean() if len(c5_df) > 0 else None
        if persist_rate is not None:
            print(f"  Persistence (C5): {persist_rate*100:.1f}% persisted after removal")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze Paper B experiment results")
    parser.add_argument(
        "--input",
        type=str,
        default="data/outputs.jsonl",
        help="Path to outputs.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis",
        help="Output directory for figures and tables"
    )
    args = parser.parse_args()

    # Setup paths
    input_path = Path(args.input)
    output_dir = Path(args.output)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading results from {input_path}...")
    results = load_results(str(input_path))
    print(f"Loaded {len(results)} trials")

    # Analyze
    print("Analyzing stability...")
    stability_df = analyze_stability(results)

    print("Analyzing calibration...")
    calibration_df = analyze_calibration(results)

    # Significance tests
    print("Running significance tests...")
    tests = run_significance_tests(stability_df)

    # Generate plots
    print("Generating plots...")
    plot_stability_by_condition(stability_df, figures_dir)
    plot_calibration_by_condition(calibration_df, figures_dir)
    plot_reliability_diagram(results, figures_dir)

    # Generate tables
    print("Generating tables...")
    summary_df = generate_summary_table(stability_df, calibration_df)
    summary_df.to_csv(tables_dir / "summary.csv", index=False)

    stability_df.to_csv(tables_dir / "stability_raw.csv", index=False)
    if not calibration_df.empty:
        calibration_df.to_csv(tables_dir / "calibration_raw.csv", index=False)

    # Print summary
    print_executive_summary(stability_df, calibration_df, tests)

    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to: {tables_dir}")


if __name__ == "__main__":
    main()
