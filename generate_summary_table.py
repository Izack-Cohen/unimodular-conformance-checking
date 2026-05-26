\
\
\
\
   

import json
import pandas as pd
import os
import glob
import numpy as np
from pathlib import Path


def load_results(json_path):
                                      
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_summary_statistics(data):
\
\
\
       

                            
    traces = data["per_trace"]

                                  
    rows = []
    for trace in traces:
        row = {
            'trace_idx': trace['trace_idx'],
            'trace_length': trace['trace_length'],
        }

                 
        if trace.get('astar_status') == 'optimal':
            row['astar_optimal'] = True
            row['astar_time'] = trace['astar_time']
            row['astar_cost'] = trace.get('astar_cost')
        else:
            row['astar_optimal'] = False
            row['astar_time'] = None
            row['astar_cost'] = None

                 
        if trace.get('lp_status') == 'optimal':
            row['lp_optimal'] = True
            row['lp_total_time'] = trace['lp_total_time']
            row['lp_rg_build_time'] = trace.get('lp_rg_build_time')
            row['lp_solve_time'] = trace.get('lp_solve_time')
            row['lp_cost'] = trace.get('lp_objective')
        else:
            row['lp_optimal'] = False
            row['lp_total_time'] = None
            row['lp_rg_build_time'] = None
            row['lp_solve_time'] = None
            row['lp_cost'] = None

        rows.append(row)

    df = pd.DataFrame(rows)

                                  
    summary = {
        'dataset': data['dataset'],
        'n_traces': data['n_traces'],
        'min_trace_length': df['trace_length'].min(),
        'max_trace_length': df['trace_length'].max(),
        'avg_trace_length': df['trace_length'].mean(),
        'median_trace_length': df['trace_length'].median(),
    }

                                            
    if 'fitness' in data:
        summary['fitness'] = data['fitness']
    if 'precision' in data:
        summary['precision'] = data['precision']

                   
    astar_optimal = df[df['astar_optimal'] == True]
    summary['astar_n_optimal'] = len(astar_optimal)
    summary['astar_pct_optimal'] = (len(astar_optimal) / len(df)) * 100

    if len(astar_optimal) > 0:
        summary['astar_avg_cpu'] = astar_optimal['astar_time'].mean()
        summary['astar_std_cpu'] = astar_optimal['astar_time'].std()
        summary['astar_median_cpu'] = astar_optimal['astar_time'].median()
        summary['astar_min_cpu'] = astar_optimal['astar_time'].min()
        summary['astar_max_cpu'] = astar_optimal['astar_time'].max()
        summary['astar_cv'] = (summary['astar_std_cpu'] / summary['astar_avg_cpu']) * 100
    else:
        summary['astar_avg_cpu'] = None
        summary['astar_std_cpu'] = None
        summary['astar_median_cpu'] = None
        summary['astar_min_cpu'] = None
        summary['astar_max_cpu'] = None
        summary['astar_cv'] = None

                   
    lp_optimal = df[df['lp_optimal'] == True]
    summary['lp_n_optimal'] = len(lp_optimal)
    summary['lp_pct_optimal'] = (len(lp_optimal) / len(df)) * 100

    if len(lp_optimal) > 0:
        summary['lp_avg_cpu'] = lp_optimal['lp_total_time'].mean()
        summary['lp_std_cpu'] = lp_optimal['lp_total_time'].std()
        summary['lp_median_cpu'] = lp_optimal['lp_total_time'].median()
        summary['lp_min_cpu'] = lp_optimal['lp_total_time'].min()
        summary['lp_max_cpu'] = lp_optimal['lp_total_time'].max()
        summary['lp_cv'] = (summary['lp_std_cpu'] / summary['lp_avg_cpu']) * 100

                                
        summary['lp_avg_rg_build'] = lp_optimal['lp_rg_build_time'].mean()
        summary['lp_std_rg_build'] = lp_optimal['lp_rg_build_time'].std()
        summary['lp_pct_rg_build'] = (summary['lp_avg_rg_build'] / summary['lp_avg_cpu']) * 100

        summary['lp_avg_solve'] = lp_optimal['lp_solve_time'].mean()
        summary['lp_std_solve'] = lp_optimal['lp_solve_time'].std()
        summary['lp_pct_solve'] = (summary['lp_avg_solve'] / summary['lp_avg_cpu']) * 100
    else:
        summary['lp_avg_cpu'] = None
        summary['lp_std_cpu'] = None
        summary['lp_median_cpu'] = None
        summary['lp_min_cpu'] = None
        summary['lp_max_cpu'] = None
        summary['lp_cv'] = None
        summary['lp_avg_rg_build'] = None
        summary['lp_std_rg_build'] = None
        summary['lp_pct_rg_build'] = None
        summary['lp_avg_solve'] = None
        summary['lp_std_solve'] = None
        summary['lp_pct_solve'] = None

                         
    if summary['astar_avg_cpu'] and summary['lp_avg_cpu']:
        summary['speedup'] = summary['astar_avg_cpu'] / summary['lp_avg_cpu']
    else:
        summary['speedup'] = None

                                      
    if len(astar_optimal) > 0 and len(lp_optimal) > 0:
                            
        common_traces = set(astar_optimal['trace_idx']) & set(lp_optimal['trace_idx'])
        if common_traces:
            astar_costs = df[df['trace_idx'].isin(common_traces)]['astar_cost']
            lp_costs = df[df['trace_idx'].isin(common_traces)]['lp_cost']

                                                     
            costs_match = np.allclose(astar_costs.dropna(), lp_costs.dropna(), rtol=1e-4)
            summary['costs_verified'] = costs_match
            summary['n_verified_traces'] = len(common_traces)
        else:
            summary['costs_verified'] = None
            summary['n_verified_traces'] = 0
    else:
        summary['costs_verified'] = None
        summary['n_verified_traces'] = 0

                                         
    if len(astar_optimal) > 1:
        astar_corr = astar_optimal[['trace_length', 'astar_time']].corr().iloc[0, 1]
        summary['astar_length_time_corr'] = astar_corr
    else:
        summary['astar_length_time_corr'] = None

    if len(lp_optimal) > 1:
        lp_corr = lp_optimal[['trace_length', 'lp_total_time']].corr().iloc[0, 1]
        summary['lp_length_time_corr'] = lp_corr
    else:
        summary['lp_length_time_corr'] = None

    return summary


def create_summary_table(results_dir='results'):
                                                                              

                                
    json_files = glob.glob(os.path.join(results_dir, '*_optimized.json'))

                                      
    json_files.extend(glob.glob(os.path.join(results_dir, 'quickstart_*.json')))

                       
    json_files = list(set(json_files))

    if not json_files:
        print(f"No result files found in {results_dir}")
        return None

                       
    summaries = []
    for json_file in sorted(json_files):
        print(f"Processing: {os.path.basename(json_file)}")
        try:
            data = load_results(json_file)
            summary = extract_summary_statistics(data)
            summaries.append(summary)
        except Exception as e:
            print(f"  Error processing {json_file}: {e}")

    if not summaries:
        print("No valid summaries generated")
        return None

                      
    df = pd.DataFrame(summaries)

                          
    df = df.sort_values('dataset')

    return df


def print_summary_table(df):
                                                        

    print("\n" + "=" * 120)
    print("CONFORMANCE CHECKING EXPERIMENT SUMMARY V2.1")
    print("=" * 120)

    for idx, row in df.iterrows():
        print(f"\n{'='*120}")
        print(f"Dataset: {row['dataset']}")
        print(f"{'='*120}")

                            
        print(f"\nDataset Statistics:")
        print(f"  Number of traces:        {row['n_traces']:,}")
        print(f"  Trace length:            {row['min_trace_length']:.0f} - {row['max_trace_length']:.0f} (min-max)")
        print(f"  Average trace length:    {row['avg_trace_length']:.2f}")
        print(f"  Median trace length:     {row['median_trace_length']:.2f}")

        if pd.notna(row.get('fitness')):
            print(f"  Model fitness:           {row['fitness']:.4f}")
        if pd.notna(row.get('precision')):
            print(f"  Model precision:         {row['precision']:.4f}")

                      
        print(f"\nA* Algorithm:")
        print(f"  Traces solved:           {row['astar_n_optimal']}/{row['n_traces']} ({row['astar_pct_optimal']:.1f}%)")
        if pd.notna(row['astar_avg_cpu']):
            print(f"  Average CPU time:        {row['astar_avg_cpu']:.4f}s ± {row['astar_std_cpu']:.4f}s")
            print(f"  Median CPU time:         {row['astar_median_cpu']:.4f}s")
            print(f"  CPU time range:          {row['astar_min_cpu']:.4f}s - {row['astar_max_cpu']:.4f}s")
            print(f"  Coefficient of Variation: {row['astar_cv']:.2f}%")
            if pd.notna(row.get('astar_length_time_corr')):
                print(f"  Length-Time Correlation: {row['astar_length_time_corr']:.3f}")
        else:
            print(f"  Average CPU time:        N/A")

                          
        print(f"\nLP (URC2) Method:")
        print(f"  Traces solved:           {row['lp_n_optimal']}/{row['n_traces']} ({row['lp_pct_optimal']:.1f}%)")
        if pd.notna(row['lp_avg_cpu']):
            print(f"  Average CPU time:        {row['lp_avg_cpu']:.4f}s ± {row['lp_std_cpu']:.4f}s")
            print(f"  Median CPU time:         {row['lp_median_cpu']:.4f}s")
            print(f"  CPU time range:          {row['lp_min_cpu']:.4f}s - {row['lp_max_cpu']:.4f}s")
            print(f"  Coefficient of Variation: {row['lp_cv']:.2f}%")
            if pd.notna(row.get('lp_length_time_corr')):
                print(f"  Length-Time Correlation: {row['lp_length_time_corr']:.3f}")

                                 
            print(f"\n  Time Breakdown:")
            print(f"    RG build time:         {row['lp_avg_rg_build']:.4f}s ± {row['lp_std_rg_build']:.4f}s ({row['lp_pct_rg_build']:.1f}%)")
            print(f"    LP solve time:         {row['lp_avg_solve']:.4f}s ± {row['lp_std_solve']:.4f}s ({row['lp_pct_solve']:.1f}%)")
        else:
            print(f"  Average CPU time:        N/A")

                              
        if pd.notna(row.get('speedup')):
            print(f"\nRelative Performance:")
            if row['speedup'] > 1:
                print(f"  ⚡ A* is {row['speedup']:.2f}x SLOWER than LP")
            else:
                print(f"  ⚡ LP is {1/row['speedup']:.2f}x SLOWER than A*")

                           
        if pd.notna(row.get('costs_verified')):
            print(f"\nCost Verification:")
            if row['costs_verified']:
                print(f"  ✓ ALL COSTS VERIFIED ({row['n_verified_traces']} traces)")
            else:
                print(f"  ✗ COST MISMATCH DETECTED ({row['n_verified_traces']} traces checked)")

                                
        if pd.notna(row['astar_cv']) and pd.notna(row['lp_cv']):
            print(f"\nConsistency Comparison (Coefficient of Variation):")
            print(f"  A*: {row['astar_cv']:.2f}%")
            print(f"  LP: {row['lp_cv']:.2f}%")
            if row['lp_cv'] < row['astar_cv']:
                ratio = row['astar_cv'] / row['lp_cv']
                print(f"  → LP is {ratio:.2f}x MORE CONSISTENT than A*")
            else:
                ratio = row['lp_cv'] / row['astar_cv']
                print(f"  → A* is {ratio:.2f}x MORE CONSISTENT than LP")


def save_summary_table(df, output_path='results/summary_table.csv'):
                                         
                                 
    column_order = [
        'dataset',
        'n_traces',
        'min_trace_length',
        'max_trace_length',
        'avg_trace_length',
        'median_trace_length',
        'fitness',
        'precision',
        'astar_n_optimal',
        'astar_pct_optimal',
        'astar_avg_cpu',
        'astar_std_cpu',
        'astar_median_cpu',
        'astar_cv',
        'astar_length_time_corr',
        'lp_n_optimal',
        'lp_pct_optimal',
        'lp_avg_cpu',
        'lp_std_cpu',
        'lp_median_cpu',
        'lp_cv',
        'lp_length_time_corr',
        'lp_avg_rg_build',
        'lp_pct_rg_build',
        'lp_avg_solve',
        'lp_pct_solve',
        'speedup',
        'costs_verified',
        'n_verified_traces'
    ]

                                               
    column_order = [col for col in column_order if col in df.columns]
    df_ordered = df[column_order]

    df_ordered.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\nSummary table saved to: {output_path}")


def save_latex_table(df, output_path='results/summary_table.tex'):
\
\
\
       

                                                 
    pub_df = pd.DataFrame({
        'Dataset': df['dataset'].apply(lambda x: Path(x).stem),
        'N': df['n_traces'],
        'Avg Length': df['avg_trace_length'],
        'A* Time (s)': df['astar_avg_cpu'],
        'A* Std': df['astar_std_cpu'],
        'A* CV (%)': df['astar_cv'],
        'LP Time (s)': df['lp_avg_cpu'],
        'LP Std': df['lp_std_cpu'],
        'LP CV (%)': df['lp_cv'],
        'Speedup': df['speedup']
    })

                            
    for col in ['Avg Length', 'A* Time (s)', 'A* Std', 'LP Time (s)', 'LP Std']:
        pub_df[col] = pub_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')

    for col in ['A* CV (%)', 'LP CV (%)']:
        pub_df[col] = pub_df[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')

    pub_df['Speedup'] = pub_df['Speedup'].apply(lambda x: f'{x:.2f}x' if pd.notna(x) else 'N/A')

                      
    latex_str = pub_df.to_latex(
        index=False,
        escape=False,
        column_format='l' + 'r' * (len(pub_df.columns) - 1),
        caption='Conformance Checking Performance Summary: A* vs LP (URC2)',
        label='tab:conformance_summary',
    )

                                      
    latex_str = latex_str.replace('\\toprule', '\\hline\\hline')
    latex_str = latex_str.replace('\\midrule', '\\hline')
    latex_str = latex_str.replace('\\bottomrule', '\\hline\\hline')

                  
    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"LaTeX table saved to: {output_path}")


def main():
                                                   

                          
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')

                                       
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        print("Please run experiments first or specify correct path.")
        return

                            
    print("=" * 120)
    print("GENERATING SUMMARY TABLE V2.1")
    print("=" * 120)
    print(f"\nSearching for results in: {results_dir}")

    df = create_summary_table(results_dir)

    if df is None:
        return

                   
    print_summary_table(df)

                 
    csv_path = os.path.join(results_dir, 'summary_table_v2.1.csv')
    save_summary_table(df, csv_path)

                   
    latex_path = os.path.join(results_dir, 'summary_table_v2.1.tex')
    save_latex_table(df, latex_path)

    print("\n" + "=" * 120)
    print("✓ SUMMARY GENERATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()