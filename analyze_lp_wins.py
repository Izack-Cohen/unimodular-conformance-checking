#!/usr/bin/env python3
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

def load_results(json_path):
                                      
    with open(json_path) as f:
        return json.load(f)

def normalize_astar_costs(results):
                                                                     
    for model in results:
        for trace in model['per_trace']:
            if trace.get('astar_cost') is not None:
                                                                
                trace['astar_cost'] = trace['astar_cost'] / 10000.0
    return results

def extract_trace_data(results):
                                                             
    rows = []
    
    for model in results:
        model_info = model['model_info']
        
        for trace in model['per_trace']:
                              
            if trace['astar_status'] == 'optimal' and trace['lp_status'] == 'optimal':
                astar_time = trace['astar_time']
                lp_time = trace['lp_time']
                
                if lp_time < astar_time * 0.95:
                    winner = 'LP'
                elif astar_time < lp_time * 0.95:
                    winner = 'A*'
                else:
                    winner = 'Similar'
            else:
                winner = 'Error'
            
                                                
            if trace['lp_time'] > 0:
                sync_pct = (trace['lp_sync_time'] / trace['lp_time']) * 100
                solve_pct = (trace['lp_solve_time'] / trace['lp_time']) * 100
            else:
                sync_pct = 0
                solve_pct = 0
            
            rows.append({
                'model_file': model_info['filename'],
                'fitness': model_info['fitness'],
                'precision': model_info['precision'],
                'method': model_info['method'],
                'trace_idx': trace['trace_idx'],
                'trace_length': trace['trace_length'],
                'astar_cost': trace['astar_cost'],
                'lp_cost': trace['lp_cost'],
                'cost_match': abs(trace['astar_cost'] - trace['lp_cost']) < 0.01,
                'astar_time': trace['astar_time'],
                'lp_time': trace['lp_time'],
                'lp_sync_time': trace['lp_sync_time'],
                'lp_solve_time': trace['lp_solve_time'],
                'lp_sync_pct': sync_pct,
                'lp_solve_pct': solve_pct,
                'rg_nodes': trace['lp_nodes'],
                'rg_edges': trace['lp_edges'],
                'winner': winner,
                'speedup': trace['astar_time'] / trace['lp_time'] if trace['lp_time'] > 0 else 0
            })
    
    return pd.DataFrame(rows)

def analyze_lp_wins(df):
                               
    
    print("="*80)
    print("LP WINS ANALYSIS")
    print("="*80)
    
                        
    total = len(df[df['winner'] != 'Error'])
    lp_wins = len(df[df['winner'] == 'LP'])
    astar_wins = len(df[df['winner'] == 'A*'])
    similar = len(df[df['winner'] == 'Similar'])
    
    print(f"\n📊 Overall Performance:")
    print(f"   Total traces: {total}")
    print(f"   A* wins: {astar_wins} ({100*astar_wins/total:.1f}%)")
    print(f"   LP wins: {lp_wins} ({100*lp_wins/total:.1f}%)")
    print(f"   Similar: {similar} ({100*similar/total:.1f}%)")
    
                       
    cost_matches = df['cost_match'].sum()
    print(f"\n✅ Cost Verification:")
    print(f"   Matching costs: {cost_matches}/{len(df)} ({100*cost_matches/len(df):.1f}%)")
    if cost_matches < len(df):
        mismatches = df[~df['cost_match']]
        print(f"   ⚠️  Found {len(mismatches)} mismatches!")
        print(f"   Max difference: {(mismatches['astar_cost'] - mismatches['lp_cost']).abs().max():.4f}")
    
                             
    print(f"\n📏 LP Wins by Trace Length:")
    df['length_bin'] = pd.cut(df['trace_length'], bins=[0, 20, 40, 60, 80, 100, 200])
    length_analysis = df.groupby('length_bin').agg({
        'winner': lambda x: (x == 'LP').sum(),
        'trace_length': 'count'
    })
    length_analysis['lp_win_rate'] = length_analysis['winner'] / length_analysis['trace_length'] * 100
    print(length_analysis[['trace_length', 'winner', 'lp_win_rate']])
    
                                
    print(f"\n🎯 LP Wins by Model Precision:")
    precision_analysis = df.groupby('model_file').agg({
        'precision': 'first',
        'fitness': 'first',
        'winner': lambda x: (x == 'LP').sum(),
        'trace_length': 'count'
    })
    precision_analysis['lp_win_rate'] = precision_analysis['winner'] / precision_analysis['trace_length'] * 100
    precision_analysis = precision_analysis.sort_values('precision')
    print(precision_analysis[['precision', 'fitness', 'winner', 'lp_win_rate']])
    
                        
    print(f"\n🔄 LP Wins by RG Size:")
    df['rg_size_bin'] = pd.cut(df['rg_nodes'], bins=[0, 1000, 2000, 5000, 10000, 100000])
    rg_analysis = df.groupby('rg_size_bin').agg({
        'winner': lambda x: (x == 'LP').sum(),
        'trace_length': 'count'
    })
    rg_analysis['lp_win_rate'] = rg_analysis['winner'] / rg_analysis['trace_length'] * 100
    print(rg_analysis[['trace_length', 'winner', 'lp_win_rate']])
    
    return df

def analyze_lp_time_breakdown(df):
                                    
    
    print(f"\n{'='*80}")
    print("LP TIME BREAKDOWN ANALYSIS")
    print("="*80)
    
                       
    print(f"\n⏱️  Average LP Time Breakdown:")
    print(f"   Total LP time: {df['lp_time'].mean():.3f}s")
    print(f"   Sync time: {df['lp_sync_time'].mean():.3f}s ({df['lp_sync_pct'].mean():.1f}%)")
    print(f"   Solve time: {df['lp_solve_time'].mean():.3f}s ({df['lp_solve_pct'].mean():.1f}%)")
    
                                                                  
                                                                 
    print(f"\n   Note: 'Solve time' includes:")
    print(f"   - Reachability graph building: ~60-80% of solve time")
    print(f"   - LP solving (Gurobi): ~20-40% of solve time")
    
                          
    print(f"\n📊 LP Time vs RG Size:")
    size_breakdown = df.groupby('rg_size_bin').agg({
        'rg_nodes': 'mean',
        'lp_time': 'mean',
        'lp_sync_time': 'mean',
        'lp_solve_time': 'mean'
    })
    print(size_breakdown)
    
                        
    print(f"\n📊 LP Time by Model:")
    model_breakdown = df.groupby('model_file').agg({
        'precision': 'first',
        'rg_nodes': 'mean',
        'lp_time': 'mean',
        'lp_sync_time': 'mean',
        'lp_solve_time': 'mean',
        'lp_sync_pct': 'mean',
        'lp_solve_pct': 'mean'
    }).sort_values('precision')
    print(model_breakdown)
    
    return df

def create_visualizations(df, output_dir='.'):
                                     
    
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
               
    sns.set_style("whitegrid")
    
                                 
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
                                      
    model_summary = df.groupby('model_file').agg({
        'precision': 'first',
        'winner': lambda x: (x == 'LP').sum() / len(x) * 100,
        'rg_nodes': 'mean'
    }).reset_index()
    
    axes[0, 0].scatter(model_summary['precision'], model_summary['winner'], s=100)
    axes[0, 0].set_xlabel('Model Precision')
    axes[0, 0].set_ylabel('LP Win Rate (%)')
    axes[0, 0].set_title('LP Win Rate vs Model Precision')
    axes[0, 0].grid(True)
    
                                         
    length_summary = df.groupby('trace_length').agg({
        'winner': lambda x: (x == 'LP').sum() / len(x) * 100 if len(x) > 0 else 0
    }).reset_index()
    
    axes[0, 1].scatter(length_summary['trace_length'], length_summary['winner'], alpha=0.6)
    axes[0, 1].set_xlabel('Trace Length')
    axes[0, 1].set_ylabel('LP Win Rate (%)')
    axes[0, 1].set_title('LP Win Rate vs Trace Length')
    axes[0, 1].grid(True)
    
                                 
    lp_data = df[df['winner'] == 'LP']
    astar_data = df[df['winner'] == 'A*']
    
    if len(lp_data) > 0:
        axes[1, 0].scatter(lp_data['rg_nodes'], lp_data['lp_time'], 
                          alpha=0.6, label='LP wins', color='green')
    if len(astar_data) > 0:
        axes[1, 0].scatter(astar_data['rg_nodes'], astar_data['lp_time'], 
                          alpha=0.6, label='A* wins', color='red')
    axes[1, 0].set_xlabel('RG Nodes')
    axes[1, 0].set_ylabel('LP Time (s)')
    axes[1, 0].set_title('LP Time vs RG Size (by winner)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
                             
    valid_df = df[df['winner'] != 'Error']
    axes[1, 1].scatter(valid_df['astar_time'], valid_df['lp_time'], alpha=0.5)
    max_time = max(valid_df['astar_time'].max(), valid_df['lp_time'].max())
    axes[1, 1].plot([0, max_time], [0, max_time], 'r--', label='Equal time')
    axes[1, 1].set_xlabel('A* Time (s)')
    axes[1, 1].set_ylabel('LP Time (s)')
    axes[1, 1].set_title('A* vs LP Time Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    output_path = f'{output_dir}/lp_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()
    
                                            
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_breakdown = df.groupby('model_file').agg({
        'precision': 'first',
        'lp_sync_time': 'mean',
        'lp_solve_time': 'mean'
    }).sort_values('precision')
    
    x = range(len(model_breakdown))
    ax.bar(x, model_breakdown['lp_sync_time'], label='Sync Time')
    ax.bar(x, model_breakdown['lp_solve_time'], 
           bottom=model_breakdown['lp_sync_time'], label='Solve Time (RG+LP)')
    
    ax.set_xlabel('Model (sorted by precision)')
    ax.set_ylabel('Time (s)')
    ax.set_title('LP Time Breakdown by Model')
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p:.2f}" for p in model_breakdown['precision']], rotation=45)
    ax.legend()
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    output_path = f'{output_dir}/lp_time_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_lp_wins.py <results.json>")
        print("Example: python analyze_lp_wins.py BPI_Challenge_2012_model_variants_comparison.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)
    
    print(f"Normalizing A* costs (dividing by 10,000)...")
    results = normalize_astar_costs(results)
    
    print(f"Extracting trace data...")
    df = extract_trace_data(results)
    
    print(f"Total traces: {len(df)}")
    print(f"Total models: {df['model_file'].nunique()}")
    
                  
    df = analyze_lp_wins(df)
    df = analyze_lp_time_breakdown(df)
    
                           
    create_visualizations(df)
    
                         
    output_csv = json_path.replace('.json', '_analyzed.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed analysis saved to: {output_csv}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
