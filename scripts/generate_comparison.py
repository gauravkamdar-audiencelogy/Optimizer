#!/usr/bin/env python3
"""
Generate Comparison Report - Compare optimizer runs across datasets.

Usage:
    # Compare latest runs from each dataset
    python scripts/generate_comparison.py

    # Compare specific runs
    python scripts/generate_comparison.py --drugs-run 20260202_150856 --nativo-run 20260202_151356

    # Output to specific file
    python scripts/generate_comparison.py --output reports/comparison.md
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics_reporter import generate_comparison_report


def get_latest_run(output_dir: Path) -> str:
    """Get the most recent run ID from an output directory."""
    runs = [d.name for d in output_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
    if not runs:
        return None
    return sorted(runs)[-1]


def load_metrics(output_dir: Path, run_id: str) -> dict:
    """Load metrics JSON for a run."""
    metrics_path = output_dir / run_id / f"metrics_{run_id}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}")
    with open(metrics_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison report between optimizer runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/generate_comparison.py
    python scripts/generate_comparison.py --drugs-run 20260202_150856 --nativo-run 20260202_151356
    python scripts/generate_comparison.py --output reports/weekly_comparison.md
        """
    )
    parser.add_argument('--drugs-run', type=str, default=None,
                        help='Specific run ID for drugs_hcp (default: latest)')
    parser.add_argument('--nativo-run', type=str, default=None,
                        help='Specific run ID for nativo_consumer (default: latest)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: output/comparison_{timestamp}.md)')
    args = parser.parse_args()

    # Paths
    drugs_output = Path("output/drugs_hcp")
    nativo_output = Path("output/nativo_consumer")

    # Get run IDs
    drugs_run = args.drugs_run or get_latest_run(drugs_output)
    nativo_run = args.nativo_run or get_latest_run(nativo_output)

    if not drugs_run:
        print("ERROR: No drugs_hcp runs found in output/drugs_hcp/")
        sys.exit(1)
    if not nativo_run:
        print("ERROR: No nativo_consumer runs found in output/nativo_consumer/")
        sys.exit(1)

    print(f"Comparing runs:")
    print(f"  drugs_hcp: {drugs_run}")
    print(f"  nativo_consumer: {nativo_run}")

    # Load metrics
    try:
        drugs_metrics = load_metrics(drugs_output, drugs_run)
        nativo_metrics = load_metrics(nativo_output, nativo_run)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Generate output path
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"output/comparison_{timestamp}.md")

    # Generate comparison report
    generate_comparison_report(
        metrics_a=drugs_metrics,
        metrics_b=nativo_metrics,
        name_a="drugs_hcp",
        name_b="nativo_consumer",
        output_path=str(output_path)
    )

    print(f"\nComparison report saved to: {output_path}")

    # Also print key differences
    print("\n" + "="*60)
    print("KEY DIFFERENCES")
    print("="*60)

    drugs_wr = drugs_metrics.get('global_stats', {}).get('global_win_rate', 0)
    nativo_wr = nativo_metrics.get('global_stats', {}).get('global_win_rate', 0)
    print(f"\nWin Rate: drugs_hcp={drugs_wr:.1%} vs nativo_consumer={nativo_wr:.1%}")

    drugs_bid = drugs_metrics.get('bid_summary', {}).get('bid_median', 0)
    nativo_bid = nativo_metrics.get('bid_summary', {}).get('bid_median', 0)
    print(f"Median Bid: drugs_hcp=${drugs_bid:.2f} vs nativo_consumer=${nativo_bid:.2f}")

    drugs_auc = drugs_metrics.get('model_calibration', {}).get('win_rate_model_empirical', {}).get('auc_roc', 0)
    nativo_auc = nativo_metrics.get('model_calibration', {}).get('win_rate_model_empirical', {}).get('auc_roc', 0)
    print(f"WR Model AUC: drugs_hcp={drugs_auc:.4f} vs nativo_consumer={nativo_auc:.4f}")


if __name__ == '__main__':
    main()
