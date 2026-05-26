\
\
\
\
\
\
\
\
\
\
\
\
\
   

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy import stats


def load_results(json_path):
                                      
    with open(json_path, 'r') as f:
        return json.load(f)


def create_dataframe(data):
                                                 
    traces = data["per_trace"]

    rows = []
    for trace in traces:
                         
        if trace.get('astar_status') == 'optimal':
            rows.append({
                'trace_idx': trace['trace_idx'],
                'trace_length': trace['trace_length'],
                'method': 'astar',
                'computation_time': trace['astar_time'],
                'alignment_cost': trace.get('astar_cost'),
                'status': 'optimal'
            })

                         
        if trace.get('lp_status') == 'optimal':
            rows.append({
                'trace_idx': trace['trace_idx'],
                'trace_length': trace['trace_length'],
                'method': 'lp',
                'computation_time': trace['lp_total_time'],
                'alignment_cost': trace.get('lp_objective'),
                'status': 'optimal',
                'rg_build_time': trace.get('lp_rg_build_time'),
                'solve_time': trace.get('lp_solve_time'),
                'num_nodes': trace.get('lp_num_nodes'),
                'num_edges': trace.get('lp_num_edges')
            })

    return pd.DataFrame(rows)


def print_dataset_summary(data, df):
                                        
    print("=" * 100)
    print("📊 DATASET SUMMARY")
    print("=" * 100)
    print(f"Dataset: {data['dataset']}")
    print(f"Total traces: {data['n_traces']:,}")
    print(f"Average trace length: {data['avg_trace_length']:.2f} events")

    if 'fitness' in data:
        print(f"Model fitness: {data['fitness']:.4f}")
    if 'precision' in data:
        print(f"Model precision: {data['precision']:.4f}")

    if 'cache_stats' in data:
        print(f"Cache hit rate: {data['cache_stats']['hit_rate']:.2f}%")

    print(f"\nSolved traces per method:")
    method_counts = df.groupby('method').size()
    for method, count in method_counts.items():
        print(f"  {method.upper()}: {count}/{data['n_traces']} ({count/data['n_traces']*100:.1f}%)")


def find_crossover_point(df):
\
\
\
       
                                                    
    comparison = df.groupby(['trace_length', 'method'])['computation_time'].mean().unstack()

    if 'astar' not in comparison.columns or 'lp' not in comparison.columns:
        return None

                                               
    comparison['lp_faster'] = comparison['lp'] < comparison['astar']

                                                                              
    for i in range(len(comparison) - 5):
        if comparison['lp_faster'].iloc[i:i+5].all():
            return comparison.index[i]

    return None


def print_performance_summary(df):
                                                                  
    print("\n" + "=" * 100)
    print("⏱️  PERFORMANCE SUMMARY")
    print("=" * 100)

    summary_data = []

    for method in ['astar', 'lp']:
        method_data = df[df['method'] == method]

        if len(method_data) > 0:
            times = method_data['computation_time']

            summary_data.append({
                'Method': method.upper(),
                'Count': len(method_data),
                'Mean': times.mean(),
                'Std': times.std(),
                'CV (%)': (times.std() / times.mean() * 100),
                'Median': times.median(),
                'Min': times.min(),
                'Max': times.max(),
                'Q1': times.quantile(0.25),
                'Q3': times.quantile(0.75)
            })

    summary_df = pd.DataFrame(summary_data)

                 
    print("\n" + summary_df.to_string(
        index=False,
        float_format=lambda x: f'{x:.4f}' if x < 1000 else f'{x:.0f}'
    ))

                      
    if len(summary_df) == 2:
        astar_mean = summary_df[summary_df['Method'] == 'ASTAR']['Mean'].values[0]
        lp_mean = summary_df[summary_df['Method'] == 'LP']['Mean'].values[0]
        speedup = astar_mean / lp_mean

        print(f"\n{'─' * 100}")
        print(f"Overall Speedup:", end=" ")
        if speedup > 1:
            print(f"A* is {speedup:.2f}x SLOWER than LP")
        else:
            print(f"LP is {1/speedup:.2f}x SLOWER than A*")
        print(f"{'─' * 100}")

                            
    if len(summary_df) == 2:
        astar_cv = summary_df[summary_df['Method'] == 'ASTAR']['CV (%)'].values[0]
        lp_cv = summary_df[summary_df['Method'] == 'LP']['CV (%)'].values[0]

        print(f"\nConsistency (Coefficient of Variation):")
        print(f"  A*: {astar_cv:.2f}%")
        print(f"  LP: {lp_cv:.2f}%")

        if lp_cv < astar_cv:
            print(f"  → LP is {astar_cv/lp_cv:.2f}x MORE CONSISTENT")
        else:
            print(f"  → A* is {lp_cv/astar_cv:.2f}x MORE CONSISTENT")


def analyze_by_trace_length(df):
                                                                   
    print("\n" + "=" * 100)
    print("📏 PERFORMANCE BY TRACE LENGTH")
    print("=" * 100)

    max_length = df['trace_length'].max()

                                                
    quartiles = df['trace_length'].quantile([0.25, 0.5, 0.75]).values
    ranges = [
        (0, int(quartiles[0]), "Q1: Short"),
        (int(quartiles[0]), int(quartiles[1]), "Q2: Medium"),
        (int(quartiles[1]), int(quartiles[2]), "Q3: Long"),
        (int(quartiles[2]), int(max_length) + 1, "Q4: Very Long")
    ]

    for min_len, max_len, label in ranges:
        range_traces = df[(df['trace_length'] >= min_len) & (df['trace_length'] < max_len)]

        if len(range_traces) > 0:
            lp_data = range_traces[range_traces['method'] == 'lp']
            astar_data = range_traces[range_traces['method'] == 'astar']

            if len(lp_data) > 0 and len(astar_data) > 0:
                lp_mean = lp_data['computation_time'].mean()
                lp_std = lp_data['computation_time'].std()
                astar_mean = astar_data['computation_time'].mean()
                astar_std = astar_data['computation_time'].std()
                speedup = astar_mean / lp_mean

                print(f"\n{label} ({min_len}-{max_len-1} events, n={len(astar_data)}):")
                print(f"  A*: {astar_mean:.4f}s ± {astar_std:.4f}s")
                print(f"  LP: {lp_mean:.4f}s ± {lp_std:.4f}s")
                print(f"  Speedup: ", end="")
                if speedup > 1:
                    print(f"A* is {speedup:.2f}x SLOWER")
                else:
                    print(f"LP is {1/speedup:.2f}x SLOWER")


def analyze_crossover(df):
                                                                           
    print("\n" + "=" * 100)
    print("🔄 CROSSOVER POINT ANALYSIS")
    print("=" * 100)

    crossover = find_crossover_point(df)

    if crossover:
        print(f"\nCrossover point detected at trace length: {crossover}")
        print(f"→ LP consistently faster for traces with ≥{crossover} activities")

                                                        
        before = df[df['trace_length'] < crossover]
        after = df[df['trace_length'] >= crossover]

        print(f"\nBefore crossover (<{crossover} activities):")
        for method in ['astar', 'lp']:
            method_data = before[before['method'] == method]
            if len(method_data) > 0:
                print(f"  {method.upper()}: {method_data['computation_time'].mean():.4f}s ± {method_data['computation_time'].std():.4f}s")

        print(f"\nAfter crossover (≥{crossover} activities):")
        for method in ['astar', 'lp']:
            method_data = after[after['method'] == method]
            if len(method_data) > 0:
                print(f"  {method.upper()}: {method_data['computation_time'].mean():.4f}s ± {method_data['computation_time'].std():.4f}s")

                                           
        astar_after = after[after['method'] == 'astar']['computation_time'].mean()
        lp_after = after[after['method'] == 'lp']['computation_time'].mean()
        if astar_after and lp_after:
            speedup = astar_after / lp_after
            print(f"\n→ For traces ≥{crossover}: A* is {speedup:.2f}x SLOWER than LP")
    else:
        print("\nNo clear crossover point detected in this dataset.")
        print("This could mean:")
        print("  • Dataset is too small or homogeneous")
        print("  • One method dominates across all trace lengths")
        print("  • Crossover occurs outside the observed range")


def analyze_correlation(df):
                                                                        
    print("\n" + "=" * 100)
    print("🔗 CORRELATION ANALYSIS")
    print("=" * 100)

    print("\nPearson Correlation (trace length vs computation time):")

    correlation_data = []

    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        if len(method_data) > 1:
            corr, p_value = stats.pearsonr(
                method_data['trace_length'],
                method_data['computation_time']
            )

            correlation_data.append({
                'Method': method.upper(),
                'Correlation': corr,
                'P-value': p_value,
                'Significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            })

    corr_df = pd.DataFrame(correlation_data)
    print("\n" + corr_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\nInterpretation:")
    print("  0.7-1.0: Strong positive correlation")
    print("  0.3-0.7: Moderate positive correlation")
    print("  0.0-0.3: Weak positive correlation")
    print("  Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


def verify_alignment_costs(df):
                                                                  
    print("\n" + "=" * 100)
    print("✓ ALIGNMENT COST VERIFICATION")
    print("=" * 100)

                            
    cost_pivot = df.pivot_table(
        index='trace_idx',
        columns='method',
        values='alignment_cost',
        aggfunc='first'
    )

    if 'astar' in cost_pivot.columns and 'lp' in cost_pivot.columns:
                           
        valid_costs = cost_pivot.dropna()

        if len(valid_costs) > 0:
                                                     
            costs_match = np.allclose(
                valid_costs['astar'],
                valid_costs['lp'],
                rtol=1e-4
            )

            print(f"\nTraces compared: {len(valid_costs)}")
            print(f"Cost agreement: {'✓ ALL COSTS MATCH' if costs_match else '✗ COSTS DIFFER'}")

            if costs_match:
                print(f"\nCost statistics:")
                print(f"  Range: {valid_costs['astar'].min():.2f} - {valid_costs['astar'].max():.2f}")
                print(f"  Mean: {valid_costs['astar'].mean():.2f}")
                print(f"  Median: {valid_costs['astar'].median():.2f}")

                             
                print(f"\nSample costs (first 10 traces):")
                sample = valid_costs.head(10)
                for idx, row in sample.iterrows():
                    print(f"  Trace {idx}: A*={row['astar']:.4f}, LP={row['lp']:.4f}")
            else:
                print("\n⚠️  WARNING: Cost mismatch detected!")
                mismatches = valid_costs[
                    ~np.isclose(valid_costs['astar'], valid_costs['lp'], rtol=1e-4)
                ]
                print(f"\nMismatched traces ({len(mismatches)}):")
                print(mismatches.to_string())
        else:
            print("\n⚠️  No overlapping traces with both A* and LP costs")
    else:
        print("\n⚠️  Cannot verify costs - missing data for one or both methods")


def create_publication_visualizations(data, df, output_dir):
                                                  
    dataset_name = Path(data['dataset']).stem

                           
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 11

                                    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Conformance Checking Performance Analysis: {dataset_name}',
                 fontsize=15, fontweight='bold')

                                           
    ax = axes[0, 0]
    for method in ['astar', 'lp']:
        method_data = df[df['method'] == method]
        ax.scatter(method_data['trace_length'], method_data['computation_time'],
                  label=method.upper(), alpha=0.5, s=30)

                        
        z = np.polyfit(method_data['trace_length'], method_data['computation_time'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(method_data['trace_length'].min(),
                             method_data['trace_length'].max(), 100)
        ax.plot(x_trend, p(x_trend), '--', alpha=0.8, linewidth=2)

    ax.set_xlabel('Trace Length (events)')
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Computation Time vs Trace Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

                                              
    ax = axes[0, 1]
    quartiles = df['trace_length'].quantile([0.25, 0.5, 0.75]).values

    box_data = []
    labels = []
    for i, (q_low, q_high) in enumerate([(0, quartiles[0]),
                                           (quartiles[0], quartiles[1]),
                                           (quartiles[1], quartiles[2]),
                                           (quartiles[2], df['trace_length'].max())]):
        q_data = df[(df['trace_length'] >= q_low) & (df['trace_length'] <= q_high)]

        for method in ['astar', 'lp']:
            method_q_data = q_data[q_data['method'] == method]
            if len(method_q_data) > 0:
                box_data.append(method_q_data['computation_time'].values)
                labels.append(f'Q{i+1}\n{method.upper()}')

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('lightblue' if i % 2 == 0 else 'lightcoral')

    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Performance Distribution by Trace Length Quartiles')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                                     
    ax = axes[1, 0]
    for method in ['astar', 'lp']:
        method_data = df[df['method'] == method].sort_values('computation_time')
        cumulative = np.arange(1, len(method_data) + 1) / len(method_data) * 100
        ax.plot(method_data['computation_time'], cumulative,
               label=method.upper(), linewidth=2)

    ax.set_xlabel('Computation Time (seconds)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Distribution of Computation Times')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

                                                                 
    ax = axes[1, 1]

    for method in ['astar', 'lp']:
        method_data = df[df['method'] == method]
        grouped = method_data.groupby('trace_length')['computation_time']

        means = grouped.mean()
        stds = grouped.std()
        counts = grouped.count()

                                           
        confidence = 1.96 * stds / np.sqrt(counts)

        ax.plot(means.index, means.values, marker='o', label=method.upper(), linewidth=2)
        ax.fill_between(means.index,
                        means.values - confidence.values,
                        means.values + confidence.values,
                        alpha=0.2)

    ax.set_xlabel('Trace Length (events)')
    ax.set_ylabel('Mean Computation Time (seconds)')
    ax.set_title('Mean Time by Trace Length (95% CI)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

               
    output_path = Path(output_dir) / f"{dataset_name}_analysis_v2.1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Analyze conformance checking experiment results (V2.1)'
    )
    parser.add_argument(
        'results_path',
        type=str,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save visualizations (default: same as results file)'
    )

    args = parser.parse_args()

                                  
    if not Path(args.results_path).exists():
        print(f"❌ Error: Results file not found: {args.results_path}")
        return

                          
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(args.results_path).parent

                  
    print("\n" + "=" * 100)
    print("CONFORMANCE CHECKING ANALYSIS V2.1")
    print("=" * 100)
    print(f"\nLoading results from: {args.results_path}")

    data = load_results(args.results_path)

                      
    df = create_dataframe(data)

    if df.empty:
        print("❌ Error: No valid data found in results file")
        return

                  
    print_dataset_summary(data, df)
    print_performance_summary(df)
    analyze_by_trace_length(df)
    analyze_crossover(df)
    analyze_correlation(df)
    verify_alignment_costs(df)

                           
    print("\n" + "=" * 100)
    print("📊 GENERATING VISUALIZATIONS")
    print("=" * 100)
    create_publication_visualizations(data, df, output_dir)

                   
    print("\n" + "=" * 100)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nAll results saved to: {output_dir}")
    print("\nFor summary tables across multiple datasets, run:")
    print("  python generate_summary_table.py")
    print("=" * 100)


if __name__ == "__main__":
    main()