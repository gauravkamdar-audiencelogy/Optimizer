#!/usr/bin/env python3
"""
Domain Tiering Analysis: Percentiles vs K-Means
================================================
Analyzes domain CTR distribution to determine optimal tiering approach.

Compares:
- Percentile-based tiers (current implementation in domain_value_model.py)
- K-means clustering (data-driven groupings)

Outputs:
- domain_tiering_plots.png: Diagnostic visualizations
- domain_tiering_report.md: Summary with recommendation

Run: python EDA_nativo_consumer/domain_tiering_analysis.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import warnings
import gc

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data_nativo_consumer"
OUTPUT_DIR = PROJECT_DIR / "output_nativo_consumer"

# Output files
PLOT_FILE = SCRIPT_DIR / "domain_tiering_plots.png"
REPORT_FILE = SCRIPT_DIR / "domain_tiering_report.md"

# Config (matching domain_value_model.py defaults)
SHRINKAGE_K = 30
PREMIUM_PERCENTILE = 95.0
STANDARD_PERCENTILE = 50.0
BELOW_AVG_PERCENTILE = 10.0
BLOCKLIST_CTR_FACTOR = 0.1
MIN_BIDS_FOR_BLOCKLIST = 100


def parse_pg_value(val):
    """Parse PostgreSQL array format: {7.50000} -> 7.5"""
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).strip('{}\"'))
    except:
        return np.nan


def load_domain_stats():
    """
    Load domain statistics from optimizer output or compute from raw data.

    Uses chunked reading for large files (>1GB) to avoid memory issues.

    Returns:
        DataFrame with columns: domain, bids, views, clicks, shrunk_rate
    """
    # Option 1: Try to find existing domain_summary from optimizer output
    if OUTPUT_DIR.exists():
        summary_files = sorted(OUTPUT_DIR.glob('*/domain_summary_*.csv'))
        if summary_files:
            latest = summary_files[-1]
            print(f"Loading domain stats from optimizer output: {latest.name}")
            df = pd.read_csv(latest)
            if 'shrunk_rate' in df.columns:
                return df

    # Option 2: Compute fresh from raw data using chunked reading
    print("Computing domain stats from raw data (chunked for large files)...")

    # Find data file
    data_files = sorted(DATA_DIR.glob('data_nativo_consumer*.csv'))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {DATA_DIR}")

    data_file = data_files[-1]
    file_size_gb = data_file.stat().st_size / (1024**3)
    print(f"Using data file: {data_file.name} ({file_size_gb:.1f} GB)")

    # Columns to load (minimal for memory efficiency)
    cols_needed = ['REC_TYPE', 'DOMAIN', 'INTERNAL_TXN_ID']
    CHUNK_SIZE = 500_000  # Process 500K rows at a time

    # Pass 1: Collect view and click transaction IDs
    print("\nPass 1: Collecting view/click transaction IDs...")
    view_txns = set()
    click_txns = set()
    chunk_num = 0

    for chunk in pd.read_csv(data_file, usecols=cols_needed, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1
        if chunk_num % 20 == 0:
            print(f"  Chunk {chunk_num}...")

        # Case-insensitive matching for REC_TYPE
        rec_type_lower = chunk['REC_TYPE'].str.lower()

        views = chunk[rec_type_lower == 'view']['INTERNAL_TXN_ID'].dropna()
        # 'click' or 'link' both indicate engagement
        clicks = chunk[rec_type_lower.isin(['click', 'link'])]['INTERNAL_TXN_ID'].dropna()

        view_txns.update(views.unique())
        click_txns.update(clicks.unique())

    print(f"  Found {len(view_txns):,} view transactions, {len(click_txns):,} click transactions")

    # Pass 2: Aggregate bid statistics by domain
    print("\nPass 2: Aggregating domain statistics...")
    domain_agg = defaultdict(lambda: {'bids': 0, 'views': 0, 'clicks': 0})
    chunk_num = 0
    total_bids = 0

    for chunk in pd.read_csv(data_file, usecols=cols_needed, chunksize=CHUNK_SIZE, low_memory=False):
        chunk_num += 1
        if chunk_num % 20 == 0:
            print(f"  Chunk {chunk_num}...")

        # Filter to bids only (case-insensitive)
        bids = chunk[chunk['REC_TYPE'].str.lower() == 'bid'].copy()
        if len(bids) == 0:
            continue

        total_bids += len(bids)

        # Clean domain names
        bids['domain'] = bids['DOMAIN'].fillna('unknown').str.strip().str.lower()

        # Mark wins and clicks
        bids['won'] = bids['INTERNAL_TXN_ID'].isin(view_txns).astype(int)
        bids['clicked'] = bids['INTERNAL_TXN_ID'].isin(click_txns).astype(int)

        # Aggregate by domain
        for domain, group in bids.groupby('domain'):
            domain_agg[domain]['bids'] += len(group)
            domain_agg[domain]['views'] += group['won'].sum()
            domain_agg[domain]['clicks'] += group['clicked'].sum()

    print(f"  Processed {total_bids:,} bids across {len(domain_agg):,} domains")

    # Convert to DataFrame
    domain_stats = pd.DataFrame([
        {'domain': d, 'bids': s['bids'], 'views': s['views'], 'clicks': s['clicks']}
        for d, s in domain_agg.items()
    ])

    print(f"\nUnique domains: {len(domain_stats):,}")

    # Calculate global rate
    total_views = domain_stats['views'].sum()
    total_clicks = domain_stats['clicks'].sum()

    if total_clicks > 0:
        global_rate = total_clicks / total_views if total_views > 0 else 0
        signal_col = 'clicks'
        signal_name = 'CTR'
    else:
        global_rate = total_views / domain_stats['bids'].sum() if domain_stats['bids'].sum() > 0 else 0
        signal_col = 'views'
        signal_name = 'win_rate'

    print(f"Global {signal_name}: {global_rate:.4%}")

    # Apply Bayesian shrinkage
    if signal_col == 'clicks':
        domain_stats['shrunk_rate'] = (
            (domain_stats['clicks'] + SHRINKAGE_K * global_rate) /
            (domain_stats['views'] + SHRINKAGE_K)
        )
    else:
        domain_stats['shrunk_rate'] = (
            (domain_stats['views'] + SHRINKAGE_K * global_rate) /
            (domain_stats['bids'] + SHRINKAGE_K)
        )

    # Store metadata
    domain_stats.attrs['global_rate'] = global_rate
    domain_stats.attrs['signal_name'] = signal_name

    return domain_stats


def analyze_distribution(shrunk_ctr: np.ndarray) -> dict:
    """
    Compute comprehensive distribution metrics.

    Args:
        shrunk_ctr: Array of shrunk CTR values

    Returns:
        Dictionary of distribution metrics
    """
    p25 = np.percentile(shrunk_ctr, 25)
    p75 = np.percentile(shrunk_ctr, 75)
    iqr = p75 - p25

    metrics = {
        # Basic stats
        'count': len(shrunk_ctr),
        'mean': float(shrunk_ctr.mean()),
        'median': float(np.median(shrunk_ctr)),
        'std': float(shrunk_ctr.std()),
        'min': float(shrunk_ctr.min()),
        'max': float(shrunk_ctr.max()),
        'range': float(shrunk_ctr.max() - shrunk_ctr.min()),

        # Shape metrics
        'skewness': float(pd.Series(shrunk_ctr).skew()),
        'kurtosis': float(pd.Series(shrunk_ctr).kurtosis()),

        # Quantiles
        'p1': float(np.percentile(shrunk_ctr, 1)),
        'p5': float(np.percentile(shrunk_ctr, 5)),
        'p10': float(np.percentile(shrunk_ctr, 10)),
        'p25': float(p25),
        'p50': float(np.percentile(shrunk_ctr, 50)),
        'p75': float(p75),
        'p90': float(np.percentile(shrunk_ctr, 90)),
        'p95': float(np.percentile(shrunk_ctr, 95)),
        'p99': float(np.percentile(shrunk_ctr, 99)),

        # IQR and outliers
        'iqr': float(iqr),
        'outliers_low': int((shrunk_ctr < (p25 - 1.5 * iqr)).sum()),
        'outliers_high': int((shrunk_ctr > (p75 + 1.5 * iqr)).sum()),
    }

    return metrics


def compute_percentile_tiers(shrunk_ctr: np.ndarray) -> tuple:
    """
    Compute percentile-based tier assignments (current implementation).

    Returns:
        (tier_labels, boundaries)
    """
    p95 = np.percentile(shrunk_ctr, PREMIUM_PERCENTILE)
    p50 = np.percentile(shrunk_ctr, STANDARD_PERCENTILE)
    p10 = np.percentile(shrunk_ctr, BELOW_AVG_PERCENTILE)

    boundaries = {'p10': p10, 'p50': p50, 'p95': p95}

    # Assign tiers (0=poor, 1=below_avg, 2=standard, 3=premium)
    tiers = np.zeros(len(shrunk_ctr), dtype=int)
    tiers[shrunk_ctr >= p10] = 1  # below_avg
    tiers[shrunk_ctr >= p50] = 2  # standard
    tiers[shrunk_ctr >= p95] = 3  # premium

    return tiers, boundaries


def compute_kmeans_analysis(shrunk_ctr: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """
    Run k-means clustering analysis.

    Returns:
        Dictionary with clustering results for various k values
    """
    X = shrunk_ctr.reshape(-1, 1)

    results = {
        'k_range': list(k_range),
        'inertias': [],
        'silhouettes': [],
        'calinski': [],
        'cluster_centers': {},
        'cluster_labels': {},
    }

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        results['inertias'].append(km.inertia_)
        results['silhouettes'].append(silhouette_score(X, labels))
        results['calinski'].append(calinski_harabasz_score(X, labels))
        results['cluster_centers'][k] = sorted(km.cluster_centers_.flatten())
        results['cluster_labels'][k] = labels

    return results


def compute_within_between_variance(values: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Compute within-cluster and between-cluster variance.

    Returns:
        (within_variance, between_variance)
    """
    overall_mean = values.mean()

    within_var = 0.0
    between_var = 0.0

    for label in np.unique(labels):
        cluster_values = values[labels == label]
        cluster_mean = cluster_values.mean()

        # Within-cluster variance
        within_var += np.sum((cluster_values - cluster_mean) ** 2)

        # Between-cluster variance
        between_var += len(cluster_values) * (cluster_mean - overall_mean) ** 2

    return within_var / len(values), between_var / len(values)


def find_elbow(k_range: list, inertias: list) -> int:
    """
    Find elbow point using the second derivative method.
    """
    # Compute second derivative
    first_deriv = np.diff(inertias)
    second_deriv = np.diff(first_deriv)

    # Elbow is where second derivative is maximum (most negative to less negative)
    elbow_idx = np.argmax(second_deriv) + 1  # +1 because of diff offset
    return k_range[elbow_idx]


def map_kmeans_to_percentile_order(kmeans_labels: np.ndarray,
                                    shrunk_ctr: np.ndarray,
                                    n_clusters: int) -> np.ndarray:
    """
    Reorder k-means cluster labels to match percentile tier ordering
    (0=lowest, 3=highest).
    """
    # Get mean CTR for each cluster
    cluster_means = []
    for i in range(n_clusters):
        cluster_means.append(shrunk_ctr[kmeans_labels == i].mean())

    # Create mapping from old label to new label (ordered by mean)
    sorted_clusters = np.argsort(cluster_means)
    label_map = {old: new for new, old in enumerate(sorted_clusters)}

    return np.array([label_map[l] for l in kmeans_labels])


def generate_plots(domain_stats: pd.DataFrame,
                   dist_metrics: dict,
                   percentile_tiers: np.ndarray,
                   percentile_bounds: dict,
                   kmeans_results: dict,
                   output_path: Path) -> None:
    """Generate all diagnostic plots."""

    shrunk_ctr = domain_stats['shrunk_rate'].values

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Domain Tiering Analysis: Percentiles vs K-Means', fontsize=14, fontweight='bold')

    # Plot 1: Distribution histogram with tier boundaries
    ax1 = axes[0, 0]
    ax1.hist(shrunk_ctr, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    colors = ['red', 'orange', 'green']
    labels = ['p10 (Below Avg)', 'p50 (Standard)', 'p95 (Premium)']
    for p_name, color, label in zip(['p10', 'p50', 'p95'], colors, labels):
        ax1.axvline(percentile_bounds[p_name], color=color, linestyle='--',
                    linewidth=2, label=f'{label}: {percentile_bounds[p_name]:.4f}')
    ax1.set_xlabel('Shrunk CTR')
    ax1.set_ylabel('Domain Count')
    ax1.set_title('Distribution with Percentile Boundaries')
    ax1.legend(fontsize=8)

    # Plot 2: Log-scale histogram
    ax2 = axes[0, 1]
    ax2.hist(shrunk_ctr, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_yscale('log')
    ax2.set_xlabel('Shrunk CTR')
    ax2.set_ylabel('Domain Count (log scale)')
    ax2.set_title('Distribution (Log Scale)')

    # Plot 3: CDF
    ax3 = axes[0, 2]
    sorted_ctr = np.sort(shrunk_ctr)
    cdf = np.arange(1, len(sorted_ctr) + 1) / len(sorted_ctr)
    ax3.plot(sorted_ctr, cdf, color='steelblue', linewidth=2)
    for p_name, color in zip(['p10', 'p50', 'p95'], colors):
        ax3.axvline(percentile_bounds[p_name], color=color, linestyle='--', alpha=0.7)
        ax3.axhline(float(p_name[1:]) / 100, color=color, linestyle=':', alpha=0.5)
    ax3.set_xlabel('Shrunk CTR')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Box plot
    ax4 = axes[1, 0]
    bp = ax4.boxplot(shrunk_ctr, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    ax4.set_ylabel('Shrunk CTR')
    ax4.set_title(f'Box Plot (Skewness: {dist_metrics["skewness"]:.2f})')
    ax4.set_xticklabels(['All Domains'])

    # Plot 5: Elbow plot
    ax5 = axes[1, 1]
    k_range = kmeans_results['k_range']
    ax5.plot(k_range, kmeans_results['inertias'], 'bo-', linewidth=2, markersize=8)
    elbow_k = find_elbow(k_range, kmeans_results['inertias'])
    elbow_idx = k_range.index(elbow_k)
    ax5.plot(elbow_k, kmeans_results['inertias'][elbow_idx], 'r*', markersize=15,
             label=f'Elbow at k={elbow_k}')
    ax5.axvline(4, color='green', linestyle=':', alpha=0.7, label='Current k=4')
    ax5.set_xlabel('Number of Clusters (k)')
    ax5.set_ylabel('Inertia (Within-cluster SS)')
    ax5.set_title('Elbow Method for Optimal k')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Silhouette scores
    ax6 = axes[1, 2]
    ax6.plot(k_range, kmeans_results['silhouettes'], 'go-', linewidth=2, markersize=8)
    best_k_sil = k_range[np.argmax(kmeans_results['silhouettes'])]
    ax6.axvline(best_k_sil, color='green', linestyle='--', alpha=0.7,
                label=f'Best: k={best_k_sil}')
    ax6.axvline(4, color='orange', linestyle=':', alpha=0.7, label='Current k=4')
    ax6.set_xlabel('Number of Clusters (k)')
    ax6.set_ylabel('Silhouette Score')
    ax6.set_title('Silhouette Score by k')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Plot 7: Tier assignment comparison
    ax7 = axes[2, 0]
    kmeans_4 = map_kmeans_to_percentile_order(
        kmeans_results['cluster_labels'][4], shrunk_ctr, 4
    )
    ax7.scatter(shrunk_ctr, percentile_tiers, alpha=0.3, s=10, label='Percentile', c='blue')
    ax7.scatter(shrunk_ctr, kmeans_4 + 0.15, alpha=0.3, s=10, label='K-Means', c='red')
    ax7.set_xlabel('Shrunk CTR')
    ax7.set_ylabel('Tier (0=Poor, 3=Premium)')
    ax7.set_title('Tier Assignments: Percentile vs K-Means')
    ax7.legend()
    ax7.set_yticks([0, 1, 2, 3])
    ax7.set_yticklabels(['Poor', 'Below Avg', 'Standard', 'Premium'])

    # Plot 8: Tier size comparison (bar chart)
    ax8 = axes[2, 1]
    tier_names = ['Poor', 'Below Avg', 'Standard', 'Premium']
    pct_counts = [np.sum(percentile_tiers == i) for i in range(4)]
    km_counts = [np.sum(kmeans_4 == i) for i in range(4)]

    x = np.arange(4)
    width = 0.35
    bars1 = ax8.bar(x - width/2, pct_counts, width, label='Percentile', color='steelblue', alpha=0.7)
    bars2 = ax8.bar(x + width/2, km_counts, width, label='K-Means', color='coral', alpha=0.7)
    ax8.set_xlabel('Tier')
    ax8.set_ylabel('Domain Count')
    ax8.set_title('Tier Size Comparison')
    ax8.set_xticks(x)
    ax8.set_xticklabels(tier_names)
    ax8.legend()

    # Add count labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax8.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Plot 9: K-means cluster centers comparison
    ax9 = axes[2, 2]
    for k in [3, 4, 5]:
        centers = kmeans_results['cluster_centers'][k]
        ax9.scatter([k] * len(centers), centers, s=100, alpha=0.7,
                   label=f'k={k}' if k == 3 else None)
        for c in centers:
            ax9.annotate(f'{c:.4f}', xy=(k, c), xytext=(5, 0),
                        textcoords='offset points', fontsize=8)

    # Add percentile boundaries for reference
    for p_name, color in zip(['p10', 'p50', 'p95'], colors):
        ax9.axhline(percentile_bounds[p_name], color=color, linestyle='--',
                   alpha=0.5, label=p_name)

    ax9.set_xlabel('Number of Clusters')
    ax9.set_ylabel('Cluster Center (CTR)')
    ax9.set_title('K-Means Cluster Centers')
    ax9.set_xticks([3, 4, 5])
    ax9.legend(fontsize=8, loc='upper right')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to: {output_path}")


def generate_report(domain_stats: pd.DataFrame,
                    dist_metrics: dict,
                    percentile_tiers: np.ndarray,
                    percentile_bounds: dict,
                    kmeans_results: dict,
                    output_path: Path) -> None:
    """Generate markdown report with analysis and recommendation."""

    shrunk_ctr = domain_stats['shrunk_rate'].values

    # Compute comparison metrics
    kmeans_4 = map_kmeans_to_percentile_order(
        kmeans_results['cluster_labels'][4], shrunk_ctr, 4
    )

    agreement = (percentile_tiers == kmeans_4).mean() * 100

    pct_within, pct_between = compute_within_between_variance(shrunk_ctr, percentile_tiers)
    km_within, km_between = compute_within_between_variance(shrunk_ctr, kmeans_4)

    elbow_k = find_elbow(kmeans_results['k_range'], kmeans_results['inertias'])
    best_k_sil = kmeans_results['k_range'][np.argmax(kmeans_results['silhouettes'])]

    # Tier counts
    tier_names = ['Poor', 'Below Avg', 'Standard', 'Premium']
    pct_counts = {tier_names[i]: np.sum(percentile_tiers == i) for i in range(4)}
    km_counts = {tier_names[i]: np.sum(kmeans_4 == i) for i in range(4)}

    # Interpretation
    skew_interp = "right-skewed (long tail of high performers)" if dist_metrics['skewness'] > 0.5 else \
                  "left-skewed (long tail of low performers)" if dist_metrics['skewness'] < -0.5 else \
                  "approximately symmetric"

    kurt_interp = "heavy tails (more outliers than normal)" if dist_metrics['kurtosis'] > 1 else \
                  "light tails (fewer outliers than normal)" if dist_metrics['kurtosis'] < -1 else \
                  "approximately normal tails"

    # Recommendation logic
    if agreement > 90:
        recommendation = "Keep Percentiles"
        reasoning = f"Methods agree {agreement:.1f}% of the time. Percentiles are simpler and equivalent."
    elif km_within < pct_within * 0.9 and km_between > pct_between * 1.1:
        recommendation = "Consider K-Means"
        reasoning = f"K-means creates more homogeneous tiers (within-var: {km_within:.6f} vs {pct_within:.6f}) with better separation."
    elif elbow_k != 4 and best_k_sil != 4:
        recommendation = f"Consider {elbow_k} Tiers"
        reasoning = f"Elbow method suggests k={elbow_k}, silhouette suggests k={best_k_sil}. Current 4-tier may be over-segmenting."
    else:
        recommendation = "Keep Percentiles"
        reasoning = "No strong evidence that k-means would improve tiering. Percentiles are simpler."

    # Generate report
    report = f"""# Domain Tiering Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Recommendation: {recommendation}**

{reasoning}

---

## Distribution Summary

| Metric | Value |
|--------|-------|
| Total domains | {dist_metrics['count']:,} |
| Mean CTR | {dist_metrics['mean']:.4%} |
| Median CTR | {dist_metrics['median']:.4%} |
| Std Dev | {dist_metrics['std']:.4%} |
| Range | {dist_metrics['min']:.4%} - {dist_metrics['max']:.4%} |
| Skewness | {dist_metrics['skewness']:.2f} |
| Kurtosis | {dist_metrics['kurtosis']:.2f} |

### Shape Interpretation

- **Skewness ({dist_metrics['skewness']:.2f})**: Distribution is {skew_interp}
- **Kurtosis ({dist_metrics['kurtosis']:.2f})**: Distribution has {kurt_interp}

### Key Percentiles

| Percentile | CTR Value |
|------------|-----------|
| 1st | {dist_metrics['p1']:.4%} |
| 5th | {dist_metrics['p5']:.4%} |
| 10th | {dist_metrics['p10']:.4%} |
| 25th | {dist_metrics['p25']:.4%} |
| 50th (median) | {dist_metrics['p50']:.4%} |
| 75th | {dist_metrics['p75']:.4%} |
| 90th | {dist_metrics['p90']:.4%} |
| 95th | {dist_metrics['p95']:.4%} |
| 99th | {dist_metrics['p99']:.4%} |

### Outliers

- Low outliers (< Q1 - 1.5×IQR): {dist_metrics['outliers_low']:,}
- High outliers (> Q3 + 1.5×IQR): {dist_metrics['outliers_high']:,}

---

## Percentile Tiers (Current Implementation)

| Tier | Threshold | Domain Count | % of Total |
|------|-----------|--------------|------------|
| Premium (Top 5%) | >= {percentile_bounds['p95']:.4%} | {pct_counts['Premium']:,} | {pct_counts['Premium']/dist_metrics['count']*100:.1f}% |
| Standard (5-50%) | >= {percentile_bounds['p50']:.4%} | {pct_counts['Standard']:,} | {pct_counts['Standard']/dist_metrics['count']*100:.1f}% |
| Below Avg (50-90%) | >= {percentile_bounds['p10']:.4%} | {pct_counts['Below Avg']:,} | {pct_counts['Below Avg']/dist_metrics['count']*100:.1f}% |
| Poor (Bottom 10%) | < {percentile_bounds['p10']:.4%} | {pct_counts['Poor']:,} | {pct_counts['Poor']/dist_metrics['count']*100:.1f}% |

---

## K-Means Analysis

### Optimal k Detection

| Method | Suggested k | Notes |
|--------|-------------|-------|
| Elbow | {elbow_k} | Point where adding clusters yields diminishing returns |
| Silhouette | {best_k_sil} | Highest cluster separation score |
| Current | 4 | Matches 4-tier percentile system |

### Clustering Quality by k

| k | Inertia | Silhouette | Calinski-Harabasz |
|---|---------|------------|-------------------|
"""

    for i, k in enumerate(kmeans_results['k_range']):
        report += f"| {k} | {kmeans_results['inertias'][i]:,.0f} | {kmeans_results['silhouettes'][i]:.3f} | {kmeans_results['calinski'][i]:,.0f} |\n"

    report += f"""
### K-Means Cluster Centers (k=4)

| Cluster | Center CTR | Domain Count |
|---------|------------|--------------|
"""

    centers_4 = sorted(kmeans_results['cluster_centers'][4])
    for i, center in enumerate(centers_4):
        count = np.sum(kmeans_4 == i)
        report += f"| {i} | {center:.4%} | {count:,} |\n"

    report += f"""
---

## Method Comparison

| Metric | Percentile | K-Means | Winner |
|--------|------------|---------|--------|
| Within-tier variance | {pct_within:.6f} | {km_within:.6f} | {'K-Means' if km_within < pct_within else 'Percentile'} |
| Between-tier separation | {pct_between:.6f} | {km_between:.6f} | {'K-Means' if km_between > pct_between else 'Percentile'} |
| Agreement | {agreement:.1f}% | - | - |

### Interpretation

- **Within-tier variance**: Lower is better (domains in same tier are more similar)
- **Between-tier separation**: Higher is better (tiers are more distinct)
- **Agreement**: How often both methods assign the same tier

---

## Tier Assignment Comparison

| Tier | Percentile | K-Means | Difference |
|------|------------|---------|------------|
"""

    for tier in tier_names:
        diff = km_counts[tier] - pct_counts[tier]
        sign = '+' if diff > 0 else ''
        report += f"| {tier} | {pct_counts[tier]:,} | {km_counts[tier]:,} | {sign}{diff:,} |\n"

    report += f"""
---

## Recommendation

### **{recommendation}**

**Reasoning:** {reasoning}

### Decision Criteria Applied

| Criterion | Result |
|-----------|--------|
| Agreement > 90% | {'✓' if agreement > 90 else '✗'} ({agreement:.1f}%) |
| K-means better within-var | {'✓' if km_within < pct_within * 0.9 else '✗'} |
| K-means better separation | {'✓' if km_between > pct_between * 1.1 else '✗'} |
| Elbow at k=4 | {'✓' if elbow_k == 4 else '✗'} (elbow at k={elbow_k}) |
| Silhouette at k=4 | {'✓' if best_k_sil == 4 else '✗'} (best at k={best_k_sil}) |

---

## Files Generated

- `domain_tiering_plots.png` - Visual diagnostics
- `domain_tiering_report.md` - This report

## Next Steps

"""

    if "Keep Percentiles" in recommendation:
        report += """1. No changes needed to `domain_value_model.py`
2. Current percentile-based tiering is appropriate for this data
3. Re-run this analysis periodically as data accumulates
"""
    else:
        report += """1. Consider updating `domain_value_model.py` to support k-means mode
2. Add `tiering_method: kmeans` config option
3. Test impact on bid distribution and win rates
4. Monitor tier stability over time with k-means
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Saved report to: {output_path}")


def main():
    """Run complete domain tiering analysis."""

    print("=" * 60)
    print("Domain Tiering Analysis: Percentiles vs K-Means")
    print("=" * 60)

    # Load data
    domain_stats = load_domain_stats()
    shrunk_ctr = domain_stats['shrunk_rate'].values

    print(f"\nAnalyzing {len(domain_stats):,} domains...")

    # Analyze distribution
    print("\n1. Computing distribution metrics...")
    dist_metrics = analyze_distribution(shrunk_ctr)
    print(f"   Skewness: {dist_metrics['skewness']:.2f}, Kurtosis: {dist_metrics['kurtosis']:.2f}")

    # Compute percentile tiers
    print("\n2. Computing percentile tiers...")
    percentile_tiers, percentile_bounds = compute_percentile_tiers(shrunk_ctr)
    print(f"   Boundaries: p10={percentile_bounds['p10']:.4%}, p50={percentile_bounds['p50']:.4%}, p95={percentile_bounds['p95']:.4%}")

    # Run k-means analysis
    print("\n3. Running k-means analysis (k=2 to k=8)...")
    kmeans_results = compute_kmeans_analysis(shrunk_ctr)
    elbow_k = find_elbow(kmeans_results['k_range'], kmeans_results['inertias'])
    best_sil_k = kmeans_results['k_range'][np.argmax(kmeans_results['silhouettes'])]
    print(f"   Elbow method suggests k={elbow_k}")
    print(f"   Best silhouette at k={best_sil_k}")

    # Generate plots
    print("\n4. Generating diagnostic plots...")
    generate_plots(domain_stats, dist_metrics, percentile_tiers,
                   percentile_bounds, kmeans_results, PLOT_FILE)

    # Generate report
    print("\n5. Generating analysis report...")
    generate_report(domain_stats, dist_metrics, percentile_tiers,
                    percentile_bounds, kmeans_results, REPORT_FILE)

    # Quick summary
    kmeans_4 = map_kmeans_to_percentile_order(
        kmeans_results['cluster_labels'][4], shrunk_ctr, 4
    )
    agreement = (percentile_tiers == kmeans_4).mean() * 100

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Domains analyzed: {len(domain_stats):,}")
    print(f"Distribution: {'right-skewed' if dist_metrics['skewness'] > 0.5 else 'symmetric'}")
    print(f"Optimal k (elbow): {elbow_k}")
    print(f"Optimal k (silhouette): {best_sil_k}")
    print(f"Percentile vs K-means agreement: {agreement:.1f}%")
    print(f"\nOutputs:")
    print(f"  - {PLOT_FILE}")
    print(f"  - {REPORT_FILE}")
    print("=" * 60)


if __name__ == '__main__':
    main()
