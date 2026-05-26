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
\
\
\
\
\
\
\
\
   

import argparse
import json
import time
from pathlib import Path
import pickle
import random

import pm4py
import numpy as np
from models.model_cache import ModelCache


                                                                             
                                    
                                                                             

                                                       
                                                                    
                                                                   
                                                                 
DISCOVERY_FITNESS_TARGET = 0.90
DISCOVERY_PRECISION_TARGET = 0.60

                                                                     
DISCOVERY_MAX_EVAL_TRACES = 2000                                                  

                                                                                     
DISCOVERY_NOISE_GRID_COARSE = [
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05,
    0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095,                    
    0.10, 0.11, 0.12, 0.13, 0.14, 0.15,
    0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40
]

                                                       
DISCOVERY_FINE_STEP = 0.0025                                      
DISCOVERY_FINE_RADIUS = 0.08                                  
DISCOVERY_FINE_MIN_NOISE = 0.0
DISCOVERY_FINE_MAX_NOISE = 0.8


                                                                             
                                         
                                                                             

def load_event_log(log_path):
\
\
\
\
       
    from pm4py.objects.log.obj import EventLog
    from pm4py.objects.conversion.log import converter as log_converter
    import pandas as pd

    log_path = Path(log_path)

                                      
    if log_path.suffix.lower() == ".xes":
        pkl_path = log_path.with_suffix(log_path.suffix + ".pkl")                 
    else:
        pkl_path = log_path.with_suffix(".pkl")

    def _summarize(log, label):
        try:
            if isinstance(log, EventLog):
                n_traces = len(log)
                n_events = sum(len(tr) for tr in log)
                print(f"  Loaded {n_traces} traces ({n_events} events) [{label} EventLog]")
                return
        except Exception:
            pass
        print(f"  Loaded {len(log)} items (type: {type(log).__name__}) [{label}]")

                           
    if pkl_path.exists():
        print(f"Loading pickled event log from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            log = pickle.load(f)

                                                                             
        if isinstance(log, pd.DataFrame):
            print("  PKL contains DataFrame; converting to EventLog...")
            log = log_converter.apply(
                log,
                variant=log_converter.Variants.TO_EVENT_LOG
            )
                                                         
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(log, f)
                print("  Overwrote PKL with EventLog representation.")
            except Exception as e:
                print(f"  Warning: could not overwrite PKL with EventLog: {e}")

        _summarize(log, "PKL")
        return log

                          
    print(f"Loading event log from XES: {log_path}")
    log = pm4py.read_xes(str(log_path))

                                                          
    if isinstance(log, pd.DataFrame):
        print("  XES parsed as DataFrame; converting to EventLog...")
        log = log_converter.apply(
            log,
            variant=log_converter.Variants.TO_EVENT_LOG
        )

    _summarize(log, "XES/EventLog")

                       
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(log, f)
        print(f"  Cached EventLog to: {pkl_path}")
    except Exception as e:
        print(f"  Warning: could not cache log to {pkl_path}: {e}")

    return log


                                                                             
                                                      
                                                                             

def _sample_log_for_evaluation(log, max_traces=DISCOVERY_MAX_EVAL_TRACES, seed=42):
\
\
\
       
    from pm4py.objects.log.obj import EventLog

                                      
    if not isinstance(log, EventLog):
        from pm4py.objects.conversion.log import converter as log_converter
        log = log_converter.apply(
            log,
            variant=log_converter.Variants.TO_EVENT_LOG
        )

    if max_traces is None or len(log) <= max_traces:
        return log

    rng = random.Random(seed)
    idxs = list(range(len(log)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:max_traces])

    sampled = EventLog()
    for i in idxs:
        sampled.append(log[i])

    return sampled


def calculate_quality_metrics(log, net, im, fm):
                                                                            
    print("Calculating quality metrics...")
    fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
    precision = pm4py.precision_token_based_replay(log, net, im, fm)
    print(f"  Fitness: {fitness['average_trace_fitness']:.4f}")
    print(f"  Precision: {precision:.4f}")
    return fitness["average_trace_fitness"], precision


def discover_process_model(
    log,
    model_cache_path=None,
    fitness_target: float = DISCOVERY_FITNESS_TARGET,
    precision_target: float = DISCOVERY_PRECISION_TARGET,
):
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
\
\
\
\
\
\
\
       
                         
    if model_cache_path and Path(model_cache_path).exists():
        print(f"Loading cached model from: {model_cache_path}")
        with open(model_cache_path, "rb") as f:
            cached = pickle.load(f)
        net, im, fm = cached["net"], cached["im"], cached["fm"]
        print(f"  Model has {len(net.places)} places and {len(net.transitions)} transitions")
        return net, im, fm

                                               
    print("No cached model found. Discovering process model with improved noise search...")
    history = []                                                            
    evaluated_noises = set()

    def eval_noise(noise_value: float):
                                                                                  
        nonlocal history, evaluated_noises
        if noise_value in evaluated_noises:
            return
        evaluated_noises.add(noise_value)

        print(f"  -> Testing noise={noise_value:.4f}")
        try:
            net, im, fm = pm4py.discover_petri_net_inductive(
                log,
                noise_threshold=noise_value
            )
        except TypeError:
            print("     (noise_threshold not supported; falling back to default discovery)")
            net, im, fm = pm4py.discover_petri_net_inductive(log)

                                         
        fitness = pm4py.fitness_token_based_replay(log, net, im, fm)
        precision = pm4py.precision_token_based_replay(log, net, im, fm)
        f_val = fitness["average_trace_fitness"]
        p_val = precision
        print(f"     fitness={f_val:.4f}, precision={p_val:.4f}")

        history.append((noise_value, f_val, p_val, net, im, fm))

                                   
    print("\n=== Stage 1: Improved coarse noise grid search ===")
    print(f"Target: fitness >= {fitness_target:.2f}, precision >= {precision_target:.2f}")
    print(f"Coarse noise thresholds: {DISCOVERY_NOISE_GRID_COARSE}")
    for noise in DISCOVERY_NOISE_GRID_COARSE:
        eval_noise(noise)

    def choose_best(candidates):
\
\
\
\
\
\
           
                                              
        feasible = [
            (noise, fit, prec, net, im, fm)
            for (noise, fit, prec, net, im, fm) in candidates
            if fit >= fitness_target and prec >= precision_target
        ]
        if feasible:
            return max(feasible, key=lambda tpl: tpl[2])                     

                                                                                      
        near_feasible = [
            (noise, fit, prec, net, im, fm)
            for (noise, fit, prec, net, im, fm) in candidates
            if fit >= fitness_target - 0.05 and prec >= precision_target - 0.05
        ]
        if near_feasible:
                                                              
            def distance2(tpl):
                _, fit, prec, _, _, _ = tpl
                df = max(0.0, fitness_target - fit)
                dp = max(0.0, precision_target - prec)
                return df * df + dp * dp
            return min(near_feasible, key=distance2)

                                                                    
        def shortfall2(tpl):
            _noise, fit, prec, _net, _im, _fm = tpl
            df = max(0.0, fitness_target - fit)
            dp = max(0.0, precision_target - prec)
            return df * df + dp * dp

        return min(candidates, key=shortfall2)

    best_noise, best_fit, best_prec, best_net, best_im, best_fm = choose_best(history)

                                                                               
    if not (best_fit >= fitness_target and best_prec >= precision_target):
        print(f"\n✗ No model from coarse grid satisfied both constraints.")
        print(f"  Best coarse: noise={best_noise:.4f}, fitness={best_fit:.4f}, precision={best_prec:.4f}")
        print(f"\n=== Stage 2: Enhanced local refinement around best noise ===")

        fine_noises = []
        k_max = int(DISCOVERY_FINE_RADIUS / DISCOVERY_FINE_STEP)
        for k in range(-k_max, k_max + 1):
            n_val = round(best_noise + k * DISCOVERY_FINE_STEP, 4)
            if (DISCOVERY_FINE_MIN_NOISE <= n_val <= DISCOVERY_FINE_MAX_NOISE and
                    n_val not in evaluated_noises):
                fine_noises.append(n_val)

        print(f"  Refinement step: {DISCOVERY_FINE_STEP}, radius: {DISCOVERY_FINE_RADIUS}")
        print(f"  Testing {len(fine_noises)} additional noise values")
        for noise in sorted(fine_noises):
            eval_noise(noise)

        best_noise, best_fit, best_prec, best_net, best_im, best_fm = choose_best(history)
    else:
        print("\n✓ Found model satisfying both constraints in coarse grid.")

                  
    if best_fit >= fitness_target and best_prec >= precision_target:
        print("\n=== Final Model (meets both constraints) ===")
    else:
        print("\n=== Final Model (best available, does not meet both constraints) ===")

    print(f"  noise={best_noise:.4f}, fitness={best_fit:.4f}, precision={best_prec:.4f}")
    print(f"  Places: {len(best_net.places)}, Transitions: {len(best_net.transitions)}")

                            
    if model_cache_path:
        Path(model_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_cache_path, "wb") as f:
            pickle.dump({"net": best_net, "im": best_im, "fm": best_fm}, f)
        print(f"  Model cached to: {model_cache_path}")

    return best_net, best_im, best_fm



                                                                             
                                     
                                                                             

def extract_trace_labels(trace):
                                                                                 
    trace_labels = []
    for event in trace:
        try:
            if isinstance(event, dict):
                trace_labels.append(event.get("concept:name", str(event)))
            elif hasattr(event, "_dict"):
                trace_labels.append(event._dict.get("concept:name", str(event)))
            elif hasattr(event, "__getitem__"):
                try:
                    trace_labels.append(event["concept:name"])
                except (TypeError, KeyError):
                    trace_labels.append(str(event))
            else:
                trace_labels.append(str(event))
        except Exception:
            trace_labels.append(str(event))
    return trace_labels


def run_astar_alignment(trace_labels, net, im, fm, time_limit=30):
                                                                                   

    start_time = time.time()
    try:
        from solvers.a_star_solver import run_a_star_alignment as astar_func

                                                              
        result = astar_func(trace_labels, net, im, fm)

        elapsed_time = time.time() - start_time

                                
        if elapsed_time > time_limit:
            return None, elapsed_time, "timeout"

                                             
        if result and "cost" in result:
            cost = result["cost"]
        else:
            cost = None

        return cost, elapsed_time, "optimal"

    except ImportError as e:
        print(f"    A* solver not available: {e}")
        return None, 0.0, "skipped"
    except Exception as e:
        print(f"    A* error: {e}")
        return None, 0.0, "error"


def run_lp_alignment(trace_labels, net, im, fm, time_limit=30, model_cache=None, model_fitness=0.9):
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
\
       

    start_time = time.time()

    try:
        from models.sync_product_incidence import build_sync_product_incidence
        from solvers.netflow_lp_solver_optimized import solve_alignment_netflow_lp_optimized

                                                                         
                                                                                
        sync_start = time.time()
        I, c, m_i, m_f, meta = build_sync_product_incidence(
            net, im, fm, trace_labels, model_cache=model_cache
        )
        sync_build_time = time.time() - sync_start

                                                
        def calculate_max_depth_fitness_based(trace_length, model_fitness):
\
\
\
\
\
\
\
\
               
            expected_extra_moves = int((1 - model_fitness) * trace_length * 1.5)
            safety_buffer = max(10, int(trace_length * 0.1))
            depth = trace_length + expected_extra_moves + safety_buffer
            return max(depth, 15)                                 

        n = calculate_max_depth_fitness_based(len(trace_labels), model_fitness)

                                                                            
        lp_start = time.time()
        try:
            X, info = solve_alignment_netflow_lp_optimized(
                I, c, m_i, m_f,
                max_depth=n,
                time_limit=time_limit,
                enable_optimizations=True
            )

            cost = info.get("objective", np.sum(X * c.reshape(-1, 1)))
            rg_nodes = info.get("num_nodes", 0)
            rg_edges = info.get("num_edges", 0)
            rg_build_time = info.get("build_time", 0.0)
            lp_solve_time = info.get("solve_time", 0.0)

            status = "optimal"

        except Exception as e:
            print(f"    LP solver error: {e}")
            cost = None
            status = "error"
            lp_solve_time = 0.0
            rg_nodes = 0
            rg_edges = 0
            rg_build_time = 0.0

        total_time = time.time() - start_time

        return cost, total_time, sync_build_time, rg_build_time, lp_solve_time, rg_nodes, rg_edges, status

    except ImportError as e:
        print(f"    LP solver not available: {e}")
        return None, 0.0, 0.0, 0.0, 0.0, 0, 0, "skipped"
    except Exception as e:
        print(f"    Unexpected LP error: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0, 0.0, 0.0, 0.0, 0, 0, "error"


                                                                             
                        
                                                                             

def print_comprehensive_results(results):
\
\
\
       
    import pandas as pd
    import numpy as np

    print("\n" + "=" * 100)
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 100)

                  
    print(f"\nDataset: {results['dataset']}")
    print(f"Traces analyzed: {results['n_traces']}")
    print(f"Average trace length: {results['avg_trace_length']:.2f} events")
    print(f"Model fitness: {results['fitness']:.4f}")
    print(f"Model precision: {results['precision']:.4f}")
    print(f"Model: {results['model_places']} places, {results['model_transitions']} transitions")
    print(f"Cache hit rate: {results['cache_stats']['hit_rate']:.1f}%")

                                             
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
                'sync_build_time': trace.get('lp_sync_build_time'),
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

            row = {
                'Method': method.upper(),
                'Traces': len(method_data),
                'Mean (s)': mean_time,
                'Std (s)': std_time,
                'CV (%)': cv,
                'Median (s)': median_time,
                'Min (s)': min_time,
                'Max (s)': max_time
            }

                                        
            if method == 'lp':
                rg_mean = method_data['rg_build_time'].mean()
                rg_std = method_data['rg_build_time'].std()
                solve_mean = method_data['solve_time'].mean()
                solve_std = method_data['solve_time'].std()

                row['RG Build (s)'] = rg_mean
                row['RG Std (s)'] = rg_std
                row['Solve (s)'] = solve_mean
                row['Solve Std (s)'] = solve_std
                row['RG %'] = (rg_mean / mean_time * 100) if mean_time > 0 else 0

            comparison_data.append(row)

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
            print(f"⚡ LP is {1/speedup:.2f}x SLOWER than A* (A* is faster)")
        print(f"{'─' * 100}")

                       
    lp_data = df[df['method'] == 'lp']
    if len(lp_data) > 0:
        print("\n" + "=" * 100)
        print("🔧 LP TIME BREAKDOWN")
        print("=" * 100)

        sync_mean = lp_data['sync_build_time'].mean()
        sync_std = lp_data['sync_build_time'].std()
        rg_mean = lp_data['rg_build_time'].mean()
        rg_std = lp_data['rg_build_time'].std()
        solve_mean = lp_data['solve_time'].mean()
        solve_std = lp_data['solve_time'].std()
        total_mean = lp_data['time'].mean()

        sync_pct = (sync_mean / total_mean * 100) if total_mean > 0 else 0
        rg_pct = (rg_mean / total_mean * 100) if total_mean > 0 else 0
        solve_pct = (solve_mean / total_mean * 100) if total_mean > 0 else 0
        accounted = sync_pct + rg_pct + solve_pct
        other_pct = 100 - accounted

        print(f"\nSynchronous Product Construction:")
        print(f"  Mean: {sync_mean:.4f}s ± {sync_std:.4f}s ({sync_pct:.1f}% of total LP time)")
        print(f"\nReachability Graph Building:")
        print(f"  Mean: {rg_mean:.4f}s ± {rg_std:.4f}s ({rg_pct:.1f}% of total LP time)")
        print(f"\nLP Solving:")
        print(f"  Mean: {solve_mean:.4f}s ± {solve_std:.4f}s ({solve_pct:.1f}% of total LP time)")

        if other_pct > 1.0:
            print(f"\nOther/Overhead:")
            print(f"  {other_pct:.1f}% (Python overhead, data conversion, etc.)")

        print(f"\nTotal LP Time:")
        print(f"  Mean: {total_mean:.4f}s")
        print(f"  Breakdown: SP {sync_pct:.1f}% + RG {rg_pct:.1f}% + Solve {solve_pct:.1f}% + Other {other_pct:.1f}%")

                          
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

                print(f"\n{label} traces ({min_len}-{max_len-1} events, n={len(astar_data)}):")
                print(f"  A*: {astar_mean:.4f}s ± {astar_data['time'].std():.4f}s")
                print(f"  LP: {lp_mean:.4f}s ± {lp_data['time'].std():.4f}s")
                if speedup > 1:
                    print(f"  → A* is {speedup:.2f}x SLOWER")
                else:
                    print(f"  → LP is {1/speedup:.2f}x SLOWER")

                       
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
                               
            print(f"\nSample alignment costs (first 10 traces):")
            print(f"  Cost range: {cost_pivot['astar'].min():.2f} - {cost_pivot['astar'].max():.2f}")
            print(f"  Mean cost: {cost_pivot['astar'].mean():.2f}")

    print("\n" + "=" * 100)


def run_experiments(
    log_path,
    output_dir="results",
    time_limit=30,
    max_traces=None,
    use_cache=True,
):
                                                               

              
    log = load_event_log(log_path)

                    
    dataset_name = Path(log_path).stem
    model_cache_path = Path(output_dir) / f"{dataset_name}_model.pkl"

    print("\n" + "=" * 80)
    print("DISCOVERING PROCESS MODEL")
    print("=" * 80)

    net, im, fm = discover_process_model(
        log,
        model_cache_path=model_cache_path,
        fitness_target=DISCOVERY_FITNESS_TARGET,
        precision_target=DISCOVERY_PRECISION_TARGET,
    )

                               
    print("\n" + "=" * 80)
    print("EVALUATING MODEL QUALITY")
    print("=" * 80)
    fitness, precision = calculate_quality_metrics(log, net, im, fm)

                                 
    try:
        total_items = len(log)
        avg_trace_length = sum(len(trace) for trace in log) / total_items
    except Exception:
        total_items = None
        avg_trace_length = None

    if max_traces and total_items and max_traces < total_items:
        traces_to_process = max_traces
    elif max_traces:
        traces_to_process = max_traces
    else:
        traces_to_process = total_items if total_items is not None else None

                                  
    results = {
        "dataset": Path(log_path).name,
        "n_traces": traces_to_process,
        "avg_trace_length": avg_trace_length,
        "fitness": fitness,
        "precision": precision,
        "model_places": len(net.places),
        "model_transitions": len(net.transitions),
        "time_limit": time_limit,
        "solvers": {
            "milp": "none",
            "lp": "gurobi_netflow_optimized",
        },
        "cache_stats": {
            "total_traces": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate": 0.0,
        },
        "methods": {},
        "per_trace": [],
    }

    trace_cache = {}
    cache_hits = 0
    cache_misses = 0
    processed_traces = 0

    if traces_to_process is not None and traces_to_process < (total_items or traces_to_process):
        print(f"\nProcessing first {traces_to_process} of {total_items} traces...")
    else:
        print(f"\nProcessing {traces_to_process if traces_to_process is not None else 'all available'} traces...")

    astar_times = []
    lp_times = []
    astar_solved = 0
    lp_solved = 0

                                                                           
                                                                              
    print("\n🚀 Initializing ModelCache for optimized LP alignment...")
    model_cache = ModelCache(net)
    print(f"   ✓ Cached {len(model_cache.T_list)} transitions, {len(model_cache.P_list)} places")
    print(f"   ✓ Found {len(model_cache.tau_indices)} tau transitions")

                                                                               
    print("\n⚡ Pre-compiling Numba functions (eliminates ~1.5s penalty on first trace)...")
    try:
                                                               
        warmup_trace = ["A", "B"]                 
        from models.sync_product_incidence import build_sync_product_incidence
        from solvers.netflow_lp_solver_optimized import solve_alignment_netflow_lp_optimized

        I_warmup, c_warmup, m_i_warmup, m_f_warmup, _ = build_sync_product_incidence(
            net, im, fm, warmup_trace, model_cache=model_cache
        )
                                                      
        X_warmup, info_warmup = solve_alignment_netflow_lp_optimized(
            I_warmup, c_warmup, m_i_warmup, m_f_warmup,
            max_depth=10, time_limit=5, enable_optimizations=True
        )
        print(f"   ✓ Numba functions pre-compiled successfully")
    except Exception as e:
        print(f"   ⚠ Warmup failed (continuing anyway): {e}")

                           
    for idx, trace in enumerate(log):
        if traces_to_process is not None and processed_traces >= traces_to_process:
            break

        if (processed_traces + 1) % 100 == 0 or (
            traces_to_process is not None and (processed_traces + 1) == traces_to_process
        ):
            print(f"  Processed {processed_traces + 1}/{traces_to_process} traces...")

        try:
            trace_length = len(trace)
        except Exception:
            trace_length = None

        trace_labels = extract_trace_labels(trace)
        trace_key = tuple(trace_labels)

        processed_traces += 1

                    
        if use_cache and trace_key in trace_cache:
            cached_result = trace_cache[trace_key]
            results["per_trace"].append({
                "trace_idx": idx,
                "trace_length": trace_length,
                "cache_hit": True,
                **cached_result,
            })
            cache_hits += 1

            if cached_result.get("astar_status") == "optimal":
                astar_solved += 1
                astar_times.append(cached_result.get("astar_time", 0.0))
            if cached_result.get("lp_status") == "optimal":
                lp_solved += 1
                lp_times.append(cached_result.get("lp_total_time", 0.0))

            continue

        cache_misses += 1

                
        astar_cost, astar_time, astar_status = run_astar_alignment(
            trace_labels, net, im, fm, time_limit
        )
        if astar_status == "optimal":
            astar_solved += 1
            astar_times.append(astar_time)

                                                                           
        lp_cost, lp_total_time, lp_sync_build, lp_rg_build, lp_solve, lp_nodes, lp_edges, lp_status = run_lp_alignment(
            trace_labels, net, im, fm, time_limit, model_cache=model_cache, model_fitness=fitness
        )
        if lp_status == "optimal":
            lp_solved += 1
            lp_times.append(lp_total_time)

        trace_result = {
            "astar_cost": astar_cost,
            "astar_time": astar_time,
            "astar_status": astar_status,
            "milp_status": "skipped",
            "lp_status": lp_status,
            "lp_objective": lp_cost,
            "lp_sync_build_time": lp_sync_build,
            "lp_rg_build_time": lp_rg_build,
            "lp_solve_time": lp_solve,
            "lp_total_time": lp_total_time,
            "lp_num_nodes": lp_nodes,
            "lp_num_edges": lp_edges,
        }

        results["per_trace"].append({
            "trace_idx": idx,
            "trace_length": trace_length,
            "cache_hit": False,
            **trace_result,
        })

        if use_cache:
            trace_cache[trace_key] = trace_result

                             
    results["cache_stats"]["cache_hits"] = cache_hits
    results["cache_stats"]["cache_misses"] = cache_misses
    results["cache_stats"]["total_traces"] = processed_traces
    results["cache_stats"]["hit_rate"] = (
        (cache_hits / processed_traces) * 100.0 if processed_traces > 0 else 0.0
    )

                       
    results["methods"]["A*"] = {
        "method": "A*",
        "pct_optimal": (astar_solved / processed_traces) * 100.0 if processed_traces > 0 else 0.0,
        "avg_solve_time": (sum(astar_times) / len(astar_times)) if astar_times else None,
        "n_traces": processed_traces,
        "n_solved": astar_solved,
    }

    results["methods"]["LP"] = {
        "method": "LP(gurobi_netflow_optimized)",
        "pct_optimal": (lp_solved / processed_traces) * 100.0 if processed_traces > 0 else 0.0,
        "avg_solve_time": (sum(lp_times) / len(lp_times)) if lp_times else None,
        "n_traces": processed_traces,
        "n_solved": lp_solved,
    }

                  
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"{dataset_name}_optimized.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\nExperiment Summary:")
    print(
        f"  A*: {astar_solved}/{processed_traces} solved "
        f"({results['methods']['A*']['pct_optimal']:.2f}%)"
    )
    print(
        f"  LP: {lp_solved}/{processed_traces} solved "
        f"({results['methods']['LP']['pct_optimal']:.2f}%)"
    )
    print(f"  Cache hit rate: {results['cache_stats']['hit_rate']:.2f}%")

                                                
    print_comprehensive_results(results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run conformance checking experiments on event logs"
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to XES event log file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Time limit per trace in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum number of traces to process (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of duplicate traces",
    )

    args = parser.parse_args()

    if not Path(args.log_path).exists():
        print(f"Error: Log file not found: {args.log_path}")
        return

    print("=" * 80)
    print("CONFORMANCE CHECKING EXPERIMENT RUNNER V2.1")
    print("With Integrated Comprehensive Results Analysis")
    print("=" * 80)
    print(f"Dataset: {args.log_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Time limit: {args.time_limit}s per trace")
    print(f"Max traces: {args.max_traces if args.max_traces else 'all'}")
    print(f"Caching: {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 80)

    start_time = time.time()
    run_experiments(
        args.log_path,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        max_traces=args.max_traces,
        use_cache=not args.no_cache,
    )
    elapsed_time = time.time() - start_time

    print(f"\nTotal execution time: {elapsed_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()