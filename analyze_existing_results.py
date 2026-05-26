#!/usr/bin/env python3
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
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def analyze_lp_time_breakdown(results_data):
                                                          
    
    print(f"\n{'='*80}")
    print(f"LP TIME BREAKDOWN ANALYSIS")
    print(f"{'='*80}")
    
    all_traces = []
    
    for model_result in results_data:
        model_info = model_result["model_info"]
        precision = model_info.get("precision", 0)
        
        for trace in model_result["per_trace"]:
            if (trace.get("lp_status") == "optimal" and 
                trace.get("lp_sync_time") is not None and 
                trace.get("lp_time", 0) > 0):
                
                sync_pct = (trace["lp_sync_time"] / trace["lp_time"]) * 100
                solve_pct = (trace["lp_solve_time"] / trace["lp_time"]) * 100
                
                all_traces.append({
                    "precision": precision,
                    "sync_time": trace["lp_sync_time"],
                    "solve_time": trace["lp_solve_time"],
                    "total_time": trace["lp_time"],
                    "sync_pct": sync_pct,
                    "solve_pct": solve_pct,
                    "rg_nodes": trace.get("lp_nodes", 0),
                    "rg_edges": trace.get("lp_edges", 0)
                })
    
    if not all_traces:
        print("  No LP timing data found")
        return
    
    df = pd.DataFrame(all_traces)
    
                       
    print(f"\nOverall LP Time Breakdown:")
    print(f"  Average Sync/RG time: {df['sync_time'].mean():.4f}s ({df['sync_pct'].mean():.1f}%)")
    print(f"  Average LP solve time: {df['solve_time'].mean():.4f}s ({df['solve_pct'].mean():.1f}%)")
    print(f"  Average total LP time: {df['total_time'].mean():.4f}s")
    
                         
    if df['sync_pct'].mean() > 70:
        print(f"\n  → RG building is the bottleneck ({df['sync_pct'].mean():.1f}% of time)")
        print(f"     Optimization focus: Reachability graph construction")
    elif df['solve_pct'].mean() > 70:
        print(f"\n  → LP solving is the bottleneck ({df['solve_pct'].mean():.1f}% of time)")
        print(f"     Optimization focus: Linear programming solver")
    else:
        print(f"\n  → Balanced time distribution")
    
                            
    print(f"\nBreakdown by Model Precision:")
    df['precision_bin'] = pd.cut(df['precision'], 
                                   bins=[0, 0.3, 0.6, 1.0],
                                   labels=['Low (0-0.3)', 'Medium (0.3-0.6)', 'High (0.6-1.0)'])
    
    breakdown = df.groupby('precision_bin').agg({
        'sync_time': 'mean',
        'solve_time': 'mean',
        'sync_pct': 'mean',
        'rg_nodes': 'mean',
        'rg_edges': 'mean'
    }).round(4)
    
    print(breakdown)


def analyze_trace_length_effect(results_data):
                                                                
    
    print(f"\n{'='*80}")
    print(f"TRACE LENGTH EFFECT ANALYSIS")
    print(f"{'='*80}")
    
    trace_data = []
    
    for model_result in results_data:
        model_info = model_result["model_info"]
        precision = model_info.get("precision", 0)
        
        for trace in model_result["per_trace"]:
            if (trace.get("astar_status") == "optimal" and 
                trace.get("lp_status") == "optimal"):
                
                astar_time = trace.get("astar_time", 0)
                lp_time = trace.get("lp_time", 0)
                
                                  
                if astar_time < lp_time * 0.95:
                    winner = "astar"
                elif lp_time < astar_time * 0.95:
                    winner = "lp"
                else:
                    winner = "similar"
                
                trace_data.append({
                    "trace_length": trace.get("trace_length", 0),
                    "precision": precision,
                    "astar_time": astar_time,
                    "lp_time": lp_time,
                    "winner": winner
                })
    
    if not trace_data:
        print("  No comparable trace data found")
        return
    
    df = pd.DataFrame(trace_data)
    
                                
    df['length_bin'] = pd.cut(df['trace_length'], 
                                bins=[0, 20, 40, 60, 80, 100, 1000],
                                labels=['0-20', '20-40', '40-60', '60-80', '80-100', '100+'])
    
    length_analysis = df.groupby('length_bin').agg({
        'astar_time': 'mean',
        'lp_time': 'mean',
        'winner': lambda x: (x == 'lp').sum() / len(x) * 100 if len(x) > 0 else 0,
        'trace_length': 'count'
    }).round(4)
    
    length_analysis.columns = ['Avg A* Time', 'Avg LP Time', 'LP Win %', 'Count']
    
    print(f"\nPerformance by Trace Length:")
    print(length_analysis)
    
                          
    print(f"\n🔍 Searching for crossover point:")
    for length_threshold in range(20, 100, 10):
        short_traces = df[df['trace_length'] < length_threshold]
        long_traces = df[df['trace_length'] >= length_threshold]
        
        if len(short_traces) > 10 and len(long_traces) > 10:
            short_lp_win_pct = (short_traces['winner'] == 'lp').sum() / len(short_traces) * 100
            long_lp_win_pct = (long_traces['winner'] == 'lp').sum() / len(long_traces) * 100
            
            if short_lp_win_pct < 50 and long_lp_win_pct > 50:
                print(f"   Crossover detected at trace length ~{length_threshold}")
                print(f"   - Traces < {length_threshold}: LP wins {short_lp_win_pct:.1f}%")
                print(f"   - Traces ≥ {length_threshold}: LP wins {long_lp_win_pct:.1f}%")
                break


def check_cost_validation(results_data):
                                                      
    
    print(f"\n{'='*80}")
    print(f"COST VALIDATION CHECK")
    print(f"{'='*80}")
    
    total_compared = 0
    mismatches = []
    
    for model_result in results_data:
        model_file = model_result["model_info"].get("filename", "unknown")
        
        for trace in model_result["per_trace"]:
            if (trace.get("astar_status") == "optimal" and 
                trace.get("lp_status") == "optimal"):
                
                total_compared += 1
                astar_cost = trace.get("astar_cost")
                lp_cost = trace.get("lp_cost")
                
                if astar_cost is not None and lp_cost is not None:
                    if abs(astar_cost - lp_cost) > 0.01:
                        mismatches.append({
                            "model": model_file,
                            "trace_idx": trace.get("trace_idx"),
                            "astar_cost": astar_cost,
                            "lp_cost": lp_cost,
                            "difference": abs(astar_cost - lp_cost)
                        })
    
    print(f"\nTotal traces compared: {total_compared}")
    print(f"Cost mismatches: {len(mismatches)}")
    
    if len(mismatches) == 0:
        print(f"✅ All costs match! LP produces optimal solutions.")
    else:
        print(f"⚠️  {len(mismatches)} cost mismatches detected!")
        print(f"\nFirst 5 mismatches:")
        for i, m in enumerate(mismatches[:5], 1):
            print(f"  {i}. Model: {m['model']}, Trace: {m['trace_idx']}")
            print(f"     A* cost: {m['astar_cost']:.4f}, LP cost: {m['lp_cost']:.4f}, Diff: {m['difference']:.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_existing_results.py <results_json>")
        print("\nExample:")
        print("  python analyze_existing_results.py experiment_results/BPI_Challenge_2012_model_variants_comparison.json")
        return 1
    
    results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"❌ Error: File not found: {results_file}")
        return 1
    
    print(f"{'='*80}")
    print(f"ANALYZING EXISTING EXPERIMENT RESULTS")
    print(f"{'='*80}")
    print(f"File: {results_file}")
    print(f"{'='*80}")
    
                  
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    print(f"\nLoaded {len(results_data)} model results")
    
                  
    analyze_lp_time_breakdown(results_data)
    analyze_trace_length_effect(results_data)
    check_cost_validation(results_data)
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
