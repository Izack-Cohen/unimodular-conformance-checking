#!/usr/bin/env python3
\
\
\
   

import argparse
import logging
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.log.importer.xes import importer as xes_importer

                      
from experiments.enhanced_runner import EnhancedExperimentRunner
from experiments.datasets import DatasetManager


def discover_optimal_model(log, target_fitness=0.9, target_precision=0.6):
\
\
\
       
    print(f"\nDiscovering optimal model (target: fitness≥{target_fitness}, precision≥{target_precision})")

    best_model = None
    best_fitness = 0
    best_precision = 0
    best_noise = None

                                                        
    noise_low, noise_high = 0.0, 0.5
    max_iterations = 15
    iteration = 0

                                             
    all_attempts = []

    while iteration < max_iterations and noise_high - noise_low > 0.01:
                                              
        noise_values = [
            noise_low,
            (noise_low + noise_high) / 2,
            noise_high
        ]

        for noise in noise_values:
            iteration += 1
            print(f"  Iteration {iteration}: Testing noise={noise:.3f}...", end="")

            try:
                                                                              
                net, im, fm = inductive_miner.apply(
                    log,
                    variant=inductive_miner.Variants.IMf,
                    parameters={
                        inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: noise
                    }
                )

                                
                fitness = fitness_evaluator.apply(
                    log, net, im, fm,
                    variant=fitness_evaluator.Variants.TOKEN_BASED
                )['average_trace_fitness']

                precision = precision_evaluator.apply(
                    log, net, im, fm,
                    variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                )

                print(f" fitness={fitness:.3f}, precision={precision:.3f}")

                                    
                all_attempts.append({
                    'noise': noise,
                    'fitness': fitness,
                    'precision': precision,
                    'model': (net, im, fm)
                })

                                                 
                if fitness >= target_fitness and precision >= target_precision:
                    if precision < 0.8:                                
                        best_model = (net, im, fm)
                        best_fitness = fitness
                        best_precision = precision
                        best_noise = noise
                        print(f"  ✓ Found model meeting targets!")
                        noise_low = noise
                        break

                                                                           
                if fitness < target_fitness:
                    if noise == noise_values[1]:
                        noise_high = noise
                elif precision < target_precision:
                    if noise == noise_values[1]:
                        noise_low = noise
                else:
                    score = fitness * 0.6 + precision * 0.4
                    if best_model is None or score > (best_fitness * 0.6 + best_precision * 0.4):
                        best_model = (net, im, fm)
                        best_fitness = fitness
                        best_precision = precision
                        best_noise = noise

            except Exception as e:
                print(f" failed: {e}")
                continue

                             
        if best_model is None:
            if noise_high < 0.8:
                noise_high = min(noise_high + 0.1, 0.8)
            else:
                break
        else:
            noise_low = max(best_noise - 0.05, 0.0)
            noise_high = min(best_noise + 0.05, 1.0)

                                                                   
    if best_model is None:
        print("\n  No model met both targets. Finding best compromise...")
        for attempt in all_attempts:
            if attempt['fitness'] >= target_fitness:
                attempt['score'] = attempt['fitness'] + attempt['precision'] * 0.5
            else:
                attempt['score'] = attempt['fitness'] * 2

        all_attempts.sort(key=lambda x: x['score'], reverse=True)
        if all_attempts:
            best_attempt = all_attempts[0]
            best_model = best_attempt['model']
            best_fitness = best_attempt['fitness']
            best_precision = best_attempt['precision']
            best_noise = best_attempt['noise']

                                                   
    if best_model and best_fitness > target_fitness - 0.05:
        print(f"\n  Refining model around noise={best_noise:.3f}...")
        for delta in [-0.02, -0.01, 0.01, 0.02]:
            test_noise = best_noise + delta
            if 0 <= test_noise <= 1:
                try:
                    net, im, fm = inductive_miner.apply(
                        log,
                        variant=inductive_miner.Variants.IMf,
                        parameters={
                            inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: test_noise
                        }
                    )

                    fitness = fitness_evaluator.apply(
                        log, net, im, fm,
                        variant=fitness_evaluator.Variants.TOKEN_BASED
                    )['average_trace_fitness']

                    precision = precision_evaluator.apply(
                        log, net, im, fm,
                        variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN
                    )

                    if fitness >= target_fitness and precision >= target_precision:
                        if precision <= 0.8:
                            best_model = (net, im, fm)
                            best_fitness = fitness
                            best_precision = precision
                            best_noise = test_noise
                            print(
                                f"    Improved: noise={test_noise:.3f}, fitness={fitness:.3f}, precision={precision:.3f}")
                except:
                    continue

    if best_model:
        print(f"\n✓ Selected model: noise={best_noise:.3f}, fitness={best_fitness:.3f}, precision={best_precision:.3f}")
    else:
        print("\n✗ Could not find suitable model, using default")
        best_model = inductive_miner.apply(log)
        best_fitness = 0.5
        best_precision = 0.5

    return best_model, best_fitness, best_precision


def print_results_table(results, dataset_name):
\
\
\
       
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

                  
    print(f"\n📊 Dataset: {dataset_name}")
    print(f"   Traces analyzed: {results['n_traces']}")
    print(f"   Average trace length: {results['avg_trace_length']:.2f} events")

    if 'fitness' in results:
        print(f"   Model fitness: {results['fitness']:.4f}")
    if 'precision' in results:
        print(f"   Model precision: {results['precision']:.4f}")

                      
    if 'cache_stats' in results:
        print(f"   Cache hit rate: {results['cache_stats']['hit_rate']:.1f}%")

                                             
    if not results.get('per_trace'):
        print("\n⚠️  No per-trace results available")
        return

    traces = results['per_trace']
    rows = []

    for trace in traces:
                 
        if trace.get('astar_status') == 'optimal':
            rows.append({
                'trace_idx': trace['trace_idx'],
                'trace_length': trace['trace_length'],
                'method': 'astar',
                'time': trace['astar_time'],
                'cost': trace.get('astar_cost'),
                'status': 'optimal'
            })

                 
        if trace.get('lp_status') == 'optimal':
            rows.append({
                'trace_idx': trace['trace_idx'],
                'trace_length': trace['trace_length'],
                'method': 'lp',
                'time': trace['lp_total_time'],
                'cost': trace.get('lp_objective'),
                'status': 'optimal',
                'rg_build_time': trace.get('lp_rg_build_time'),
                'solve_time': trace.get('lp_solve_time')
            })

    df = pd.DataFrame(rows)

    if df.empty:
        print("\n⚠️  No optimal solutions found")
        return

                                  
    print("\n" + "=" * 100)
    print("⏱️  PERFORMANCE COMPARISON")
    print("=" * 100)

    comparison_data = []

    for method in ['astar', 'lp']:
        method_data = df[df['method'] == method]

        if len(method_data) > 0:
            mean_time = method_data['time'].mean()
            std_time = method_data['time'].std()
            median_time = method_data['time'].median()
            min_time = method_data['time'].min()
            max_time = method_data['time'].max()
            cv = (std_time / mean_time * 100) if mean_time > 0 else 0

            comparison_data.append({
                'Method': method.upper(),
                'Traces': len(method_data),
                'Mean (s)': mean_time,
                'Std (s)': std_time,
                'CV (%)': cv,
                'Median (s)': median_time,
                'Min (s)': min_time,
                'Max (s)': max_time
            })

    comp_df = pd.DataFrame(comparison_data)
    print("\n" + comp_df.to_string(index=False, float_format=lambda x: f'{x:.4f}' if x < 1000 else f'{x:.1f}'))

                      
    if len(comp_df) == 2:
        astar_mean = comp_df[comp_df['Method'] == 'ASTAR']['Mean (s)'].values[0]
        lp_mean = comp_df[comp_df['Method'] == 'LP']['Mean (s)'].values[0]
        speedup = astar_mean / lp_mean

        print(f"\n{'─' * 100}")
        if speedup > 1:
            print(f"⚡ A* is {speedup:.2f}x SLOWER than LP (LP is faster)")
        else:
            print(f"⚡ LP is {1 / speedup:.2f}x SLOWER than A* (A* is faster)")
        print(f"{'─' * 100}")

                          
    print("\n" + "=" * 100)
    print("📈 CONSISTENCY ANALYSIS")
    print("=" * 100)

    astar_data = df[df['method'] == 'astar']
    lp_data = df[df['method'] == 'lp']

    if len(astar_data) > 0 and len(lp_data) > 0:
        astar_cv = (astar_data['time'].std() / astar_data['time'].mean()) * 100
        lp_cv = (lp_data['time'].std() / lp_data['time'].mean()) * 100

        print(f"\nCoefficient of Variation (CV = std/mean):")
        print(f"  A*: {astar_cv:.2f}%")
        print(f"  LP: {lp_cv:.2f}%")

        if lp_cv < astar_cv:
            ratio = astar_cv / lp_cv
            print(f"\n  ➜ LP is {ratio:.2f}x MORE CONSISTENT than A* (lower is better)")
        else:
            ratio = lp_cv / astar_cv
            print(f"\n  ➜ A* is {ratio:.2f}x MORE CONSISTENT than LP (lower is better)")

                          
    print("\n" + "=" * 100)
    print("🔗 CORRELATION: Trace Length vs Computation Time")
    print("=" * 100)

    for method in ['astar', 'lp']:
        method_data = df[df['method'] == method]
        if len(method_data) > 1:
            corr = method_data[['trace_length', 'time']].corr().iloc[0, 1]
            print(f"  {method.upper()}: {corr:.3f}")

                                 
    print("\n" + "=" * 100)
    print("📏 PERFORMANCE BY TRACE LENGTH RANGES")
    print("=" * 100)

    max_length = df['trace_length'].max()
    ranges = [
        (0, 20, "Short"),
        (20, 40, "Medium"),
        (40, 60, "Long"),
        (60, 80, "Very Long"),
        (80, max_length + 1, "Extremely Long")
    ]

    for min_len, max_len, label in ranges:
        range_traces = df[(df['trace_length'] >= min_len) & (df['trace_length'] < max_len)]

        if len(range_traces) > 0:
            lp_data = range_traces[range_traces['method'] == 'lp']
            astar_data = range_traces[range_traces['method'] == 'astar']

            if len(lp_data) > 0 and len(astar_data) > 0:
                lp_mean = lp_data['time'].mean()
                astar_mean = astar_data['time'].mean()
                speedup = astar_mean / lp_mean

                print(f"\n{label} traces ({min_len}-{max_len - 1} events, n={len(astar_data)}):")
                print(f"  A*: {astar_mean:.4f}s ± {astar_data['time'].std():.4f}s")
                print(f"  LP: {lp_mean:.4f}s ± {lp_data['time'].std():.4f}s")
                if speedup > 1:
                    print(f"  → A* is {speedup:.2f}x SLOWER")
                else:
                    print(f"  → LP is {1 / speedup:.2f}x SLOWER")

                       
    print("\n" + "=" * 100)
    print("✓ COST VERIFICATION")
    print("=" * 100)

                            
    cost_pivot = df.pivot_table(
        index='trace_idx',
        columns='method',
        values='cost',
        aggfunc='first'
    )

    if 'astar' in cost_pivot.columns and 'lp' in cost_pivot.columns:
        costs_match = (cost_pivot['astar'].round(4) == cost_pivot['lp'].round(4)).all()

        print(f"\nAlignment cost agreement: {'✓ ALL COSTS MATCH' if costs_match else '✗ COSTS DIFFER'}")

        if not costs_match:
            print("\n⚠️  WARNING: Some alignment costs don't match!")
            mismatches = cost_pivot[cost_pivot['astar'].round(4) != cost_pivot['lp'].round(4)]
            print("\nMismatched traces:")
            print(mismatches.to_string())
        else:
                               
            sample_costs = cost_pivot.head(10)
            print(f"\nSample alignment costs (first 10 traces):")
            print(f"  Cost range: {cost_pivot['astar'].min():.2f} - {cost_pivot['astar'].max():.2f}")
            print(f"  Mean cost: {cost_pivot['astar'].mean():.2f}")

    print("\n" + "=" * 100)


def main():
                                  
                           
    parser = argparse.ArgumentParser(
        description='Alignment Checking Quickstart - Run conformance checking experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (BPI 2012, 100 traces)
  python quickstart.py

  # Specify dataset and number of traces
  python quickstart.py data/BPI_Challenge_2012.xes --max-traces 50

  # Full custom run
  python quickstart.py data/my_log.xes --max-traces 200 --time-limit 180 --output-dir my_results

  # Run all traces in dataset
  python quickstart.py data/BPI_Challenge_2012.xes --max-traces -1
        """
    )

    parser.add_argument(
        'dataset',
        nargs='?',
        default='BPI_Challenge_2012.xes',
        help='Path to event log file (.xes) or dataset name (default: BPI_Challenge_2012.xes)'
    )

    parser.add_argument(
        '--max-traces',
        type=int,
        default=100,
        help='Maximum number of traces to process (default: 100, use -1 for all traces)'
    )

    parser.add_argument(
        '--time-limit',
        type=float,
        default=120.0,
        help='Time limit per trace in seconds (default: 120.0)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )

    parser.add_argument(
        '--target-fitness',
        type=float,
        default=0.9,
        help='Target model fitness (default: 0.9)'
    )

    parser.add_argument(
        '--target-precision',
        type=float,
        default=0.6,
        help='Target model precision (default: 0.6)'
    )

    args = parser.parse_args()

                   
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

                                  
    dataset_input = args.dataset
    n_traces = None if args.max_traces == -1 else args.max_traces
    time_limit = args.time_limit
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    target_fitness = args.target_fitness
    target_precision = args.target_precision

    print("=" * 100)
    print("ALIGNMENT CHECKING QUICKSTART V2.1")
    print("=" * 100)

                  
    print(f"\n1. Loading dataset: {dataset_input}")
    dataset_manager = DatasetManager()

                                                   
    if Path(dataset_input).exists():
                          
        log_path = Path(dataset_input)
        dataset_name = log_path.name
        print(f"   Using dataset file: {log_path}")
    else:
                                           
        dataset_name = dataset_input
        if not dataset_manager.is_downloaded(dataset_name):
            print(f"   Downloading {dataset_name}...")
            dataset_manager.download_dataset(dataset_name)
        log_path = dataset_manager.get_dataset_path(dataset_name)
        print(f"   Using dataset from: {log_path}")

    log = xes_importer.apply(str(log_path))

                   
    if n_traces and n_traces < len(log):
        print(f"   Sampling {n_traces} traces from {len(log)} total traces")
        log = log[:n_traces]

                                    
    avg_trace_length = sum(len(trace) for trace in log) / len(log)
    print(f"   Loaded {len(log)} traces (avg length: {avg_trace_length:.1f} events)")

                            
    print(f"\n2. Discovering process model")
    start_time = time.time()
    (net, im, fm), fitness, precision = discover_optimal_model(
        log,
        target_fitness=target_fitness,
        target_precision=target_precision
    )
    discovery_time = time.time() - start_time

    print(f"   Model discovery took {discovery_time:.2f} seconds")
    print(f"   Model has {len(net.places)} places and {len(net.transitions)} transitions")

                                                
    print(f"\n3. Running alignment experiments")
    print(f"   Time limit per trace: {time_limit}s")

    runner = EnhancedExperimentRunner(
        enable_cache=True,
        cache_dir=output_dir / "cache",
        time_limit=time_limit
    )

                     
    experiment_start = time.time()
    results = runner.run_comparison(
        log=log,
        net=net,
        im=im,
        fm=fm,
        methods=['astar', 'lp'],                       
        dataset_name=dataset_name,
        lp_solver='gurobi',
        parallel=False
    )
    experiment_time = time.time() - experiment_start

                  
    results['dataset'] = dataset_name
    results['n_traces'] = len(log)
    results['avg_trace_length'] = avg_trace_length
    results['fitness'] = fitness
    results['precision'] = precision
    results['model_places'] = len(net.places)
    results['model_transitions'] = len(net.transitions)
    results['time_limit'] = time_limit
    results['total_experiment_time'] = experiment_time

                  
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"quickstart_{dataset_name.replace('.xes', '')}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

                                                           
    print_results_table(results, dataset_name)

                   
    print("\n" + "=" * 100)
    print("✓ EXPERIMENT COMPLETE")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Traces analyzed: {len(log)}")
    print(f"  Time limit: {time_limit}s per trace")
    print(f"  Model targets: fitness≥{target_fitness}, precision≥{target_precision}")
    print(f"\nTotal experiment time: {experiment_time:.2f} seconds")
    print(f"Results saved to: {output_file}")
    print(f"\nFor detailed analysis and visualizations, run:")
    print(f"  python analyze_results.py {output_file}")
    print("=" * 100)


if __name__ == "__main__":
    main()
