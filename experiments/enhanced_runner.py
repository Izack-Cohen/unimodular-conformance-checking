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
   
from __future__ import annotations
import argparse
import os
import time
import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

                 
from models.petri_net_loader import (
    load_trace_from_xes,
    load_petri_net,
    discover_petri_from_log,
    compute_fitness_precision,
)
from models.sync_product_incidence import build_sync_product_incidence
from solvers.solver_selector import solve_alignment
from solvers.a_star_solver import run_a_star_alignment

               
from experiments.cache_manager import (
    load_log_from_cache,
    save_log_to_cache,
    load_model_from_cache,
    save_model_to_cache,
    load_trace_results_from_cache,
    save_trace_results_to_cache
)

           
DEFAULT_TIME_LIMIT = 30.0           
DEFAULT_SAMPLE_SIZE = None                             
DEFAULT_SOLVER = "gurobi"                                  

                                                  
DEFAULT_COST_SYNC = 0.0
DEFAULT_COST_LOG = 1.0
DEFAULT_COST_MODEL = 1.0
DEFAULT_COST_TAU = 0.0001                                              


@dataclass
class MethodStats:
                                                              
    name: str
    total_traces: int = 0
    solved_optimal: int = 0
    times_solved: List[float] = None
    rg_build_times: List[float] = None               
    lp_solve_times: List[float] = None               

    def __post_init__(self):
        if self.times_solved is None:
            self.times_solved = []
        if self.rg_build_times is None:
            self.rg_build_times = []
        if self.lp_solve_times is None:
            self.lp_solve_times = []

    def add_solved(self, solve_time: float, rg_time: float = None, lp_time: float = None):
                                                 
        self.solved_optimal += 1
        self.times_solved.append(solve_time)
        if rg_time is not None:
            self.rg_build_times.append(rg_time)
        if lp_time is not None:
            self.lp_solve_times.append(lp_time)

    def get_metrics(self) -> Dict[str, Any]:
                                               
        pct_optimal = 100.0 * self.solved_optimal / max(1, self.total_traces)

        if self.times_solved:
            avg_time = np.mean(self.times_solved)
        else:
            avg_time = None

        result = {
            'method': self.name,
            'pct_optimal': pct_optimal,
            'avg_solve_time': avg_time,
            'n_traces': self.total_traces,
            'n_solved': self.solved_optimal,
        }

                                 
        if self.rg_build_times:
            result['avg_rg_build_time'] = np.mean(self.rg_build_times)
        if self.lp_solve_times:
            result['avg_lp_solve_time'] = np.mean(self.lp_solve_times)
        if self.rg_build_times and self.lp_solve_times:
            total_times = [rg + lp for rg, lp in zip(self.rg_build_times, self.lp_solve_times)]
            result['avg_total_time'] = np.mean(total_times)

        return result


@dataclass
class CacheStats:
                                                
    total_traces: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def hit_rate(self) -> float:
                                                     
        if self.total_traces == 0:
            return 0.0
        return 100.0 * self.cache_hits / self.total_traces

    def summary(self) -> str:
                                 
        return (f"Cache: {self.cache_hits}/{self.total_traces} hits "
                f"({self.hit_rate():.1f}% speedup)")


def _trace_to_event_labels(trace) -> List[str]:
                                                   
    try:
        return [e["concept:name"] for e in trace]
    except Exception:
        return [str(list(e.values())[0]) for e in trace]


def load_event_log_cached(dataset_path: str) -> Any:
\
\
\
\
       
                     
    log = load_log_from_cache(dataset_path)
    if log is not None:
        return log

                    
    print(f"  Loading log from file (first time, will be cached)...")
    from pm4py.objects.log.importer.xes import importer as xes_importer

    if dataset_path.lower().endswith(".xes"):
        log = xes_importer.apply(dataset_path)
    else:
        raise ValueError("Only .xes datasets are supported")

                          
    save_log_to_cache(dataset_path, log)

    return log


def discover_model_with_quality(log, min_fitness: float = 0.9, target_precision: float = 0.7,
                                noise_thresholds: List[float] = None) -> Tuple[Any, Any, Any, float, float]:
\
\
\
\
\
\
\
\
       
    if noise_thresholds is None:
                                                              
        noise_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    best_model = None
    best_quality = None
    best_metrics = (0.0, 0.0)

    print(f"  Discovering model with min_fitness={min_fitness}, target_precision={target_precision}")

    for noise in noise_thresholds:
        try:
            net, im, fm = discover_petri_from_log(log, variant="imf", noise_threshold=noise)
            fitness, precision = compute_fitness_precision(log, net, im, fm)

            print(f"    noise={noise:.1f}: fitness={fitness:.3f}, precision={precision:.3f}")

                                                    
            if fitness >= min_fitness:
                                                                                   
                if best_model is None or abs(precision - target_precision) < abs(best_metrics[1] - target_precision):
                    best_model = (net, im, fm)
                    best_metrics = (fitness, precision)

                                                                               
                    if abs(precision - target_precision) < 0.1:
                        print(f"    → Found good model with fitness={fitness:.3f}, precision={precision:.3f}")
                        break

        except Exception as e:
            print(f"    noise={noise:.1f}: failed - {e}")
            continue

    if best_model is None:
                                                                                
        print(f"  Warning: No model met criteria (fitness >= {min_fitness})")
        print(f"  Using best available model with fitness={best_metrics[0]:.3f}, precision={best_metrics[1]:.3f}")
        if best_model is None:
                                                        
            net, im, fm = discover_petri_from_log(log, variant="imf", noise_threshold=0.5)
            fitness, precision = compute_fitness_precision(log, net, im, fm)
            return net, im, fm, fitness, precision

    net, im, fm = best_model
    fitness, precision = best_metrics
    return net, im, fm, fitness, precision


def discover_model_cached(
        log: Any,
        dataset_path: str,
        min_fitness: float,
        target_precision: float
) -> Tuple[Any, Any, Any, float, float]:
\
\
\
\
       
                     
    cached_model = load_model_from_cache(dataset_path, min_fitness, target_precision)
    if cached_model is not None:
        return cached_model

                    
    print(f"  Discovering model (first time, will be cached)...")
    net, im, fm, fitness, precision = discover_model_with_quality(
        log, min_fitness, target_precision
    )

                          
    save_model_to_cache(dataset_path, min_fitness, target_precision,
                        net, im, fm, fitness, precision)

    return net, im, fm, fitness, precision


def run_dataset_experiment(
        dataset_path: str,
        time_limit: float = DEFAULT_TIME_LIMIT,
        sample_size: int = None,
        min_fitness: float = 0.9,
        target_precision: float = 0.7,
        cost_sync: float = DEFAULT_COST_SYNC,
        cost_model: float = DEFAULT_COST_MODEL,
        cost_log: float = DEFAULT_COST_LOG,
        cost_tau: float = DEFAULT_COST_TAU,
        milp_solver: str = DEFAULT_SOLVER,
        lp_solver: str = DEFAULT_SOLVER,
        use_trace_cache: bool = True,
) -> Dict[str, Any]:
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
       
    print(f"\n{'=' * 70}")
    print(f"Processing: {os.path.basename(dataset_path)}")
    print(f"{'=' * 70}")
    print(f"Solver Configuration: MILP={milp_solver}, LP={lp_solver}")
    print(f"Cost Configuration: sync={cost_sync}, log={cost_log}, model={cost_model}, tau={cost_tau}")
    print(f"Trace Caching: {'Enabled' if use_trace_cache else 'Disabled'}")

                                   
    print(f"\nLoading event log...")
    log = load_event_log_cached(dataset_path)
    print(f"Loaded log with {len(log)} traces")

                                    
    trace_lengths = [len(trace) for trace in log]
    avg_trace_length = np.mean(trace_lengths)
    print(f"Average trace length: {avg_trace_length:.1f} events")

                                                             
    print(f"\nDiscovering/loading process model...")
    net, im, fm, fitness, precision = discover_model_cached(
        log, dataset_path, min_fitness, target_precision
    )

    print(f"\nModel quality metrics:")
    print(f"  Fitness: {fitness:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Places: {len(net.places)}")
    print(f"  Transitions: {len(net.transitions)}")


                                      
    stats_astar = MethodStats(name="A*")
    stats_milp = MethodStats(name=f"MILP({milp_solver})")
    stats_lp = MethodStats(name=f"LP({lp_solver})")
    cache_stats = CacheStats()

                           
    if sample_size is None:
        n_traces = len(log)
    else:
        n_traces = min(sample_size, len(log))

    print(f"\nRunning conformance checking on {n_traces} traces...")
    print(f"Time limit: {time_limit}s per trace (MILP gets 3x)")

    per_trace_results = []

                        
    for idx in range(n_traces):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Processing trace {idx + 1}/{n_traces}... {cache_stats.summary()}")

        trace = log[idx]
        labels = _trace_to_event_labels(trace)

                                               
        trace_result = {
            'trace_idx': idx,
            'trace_length': len(labels),
        }

                           
        cache_stats.total_traces += 1
        cached_results = None
        if use_trace_cache:
            cached_results = load_trace_results_from_cache(
                dataset_path, labels, min_fitness, target_precision
            )

        if cached_results is not None:
                                           
            cache_stats.cache_hits += 1
            trace_result.update(cached_results)
            trace_result['cache_hit'] = True

                          
            stats_astar.total_traces += 1
            if cached_results.get('astar_status') == 'optimal':
                stats_astar.add_solved(cached_results.get('astar_time', 0.0))

                                                    
                                          
                                                                
                                                                             

            stats_lp.total_traces += 1
            if cached_results.get('lp_status') == 'optimal':
                stats_lp.add_solved(
                    cached_results.get('lp_total_time', 0.0),
                    cached_results.get('lp_rg_build_time'),
                    cached_results.get('lp_solve_time')
                )

            per_trace_results.append(trace_result)
            continue

                                      
        cache_stats.cache_misses += 1
        trace_result['cache_hit'] = False

                               
        stats_astar.total_traces += 1
        try:
            t0 = time.time()
            res_astar = run_a_star_alignment(labels, net, im, fm)
            t_astar = time.time() - t0

            cost_astar = float(res_astar.get("cost", math.inf))

            if math.isfinite(cost_astar):
                stats_astar.add_solved(t_astar)
                trace_result['astar_cost'] = cost_astar
                trace_result['astar_time'] = t_astar
                trace_result['astar_status'] = 'optimal'
            else:
                trace_result['astar_status'] = 'failed'

        except Exception as e:
            print(f"  A* ERROR: {e}")
            trace_result['astar_status'] = 'error'
            trace_result['astar_error'] = str(e)
            cost_astar = math.inf

                                                   
        try:
            I_sp, c_sp, m_i_sp, m_f_sp, meta = build_sync_product_incidence(
                net, im, fm, labels,
                cost_sync=cost_sync,
                cost_log=cost_log,
                cost_model=cost_model,
                cost_tau=cost_tau
            )
            n_bound = max(50, len(labels) * 2)

        except Exception as e:
            print(f"  Sync Product ERROR: {e}")
            trace_result['sp_error'] = str(e)
            per_trace_results.append(trace_result)
            continue

                                                        
                                                             
        if False:
            stats_milp.total_traces += 1
            try:
                t0 = time.time()
                milp_time_limit = time_limit * 3

                X_milp, milp_model = solve_alignment(
                    I=I_sp, c=c_sp, m_i=m_i_sp, m_f=m_f_sp, n=n_bound,
                    method=milp_solver,
                    model="milp",
                    time_limit=milp_time_limit
                )
                t_milp = time.time() - t0

                if hasattr(milp_model, 'ObjVal'):
                    obj_milp = float(milp_model.ObjVal)
                elif hasattr(milp_model, 'objective'):
                    obj_milp = float(milp_model.objective.value())
                else:
                    obj_milp = math.inf

                if math.isfinite(obj_milp) and math.isfinite(cost_astar):
                    if abs(obj_milp - cost_astar) < 0.01:
                        stats_milp.add_solved(t_milp)
                        trace_result['milp_status'] = 'optimal'
                    else:
                        trace_result['milp_status'] = 'suboptimal'

                    trace_result['milp_objective'] = obj_milp
                    trace_result['milp_time'] = t_milp
                else:
                    trace_result['milp_status'] = 'failed'

            except Exception as e:
                print(f"  MILP ERROR: {e}")
                trace_result['milp_status'] = 'error'
                trace_result['milp_error'] = str(e)

                                     
        stats_milp.total_traces += 1
        trace_result['milp_status'] = 'skipped'

                               
        stats_lp.total_traces += 1
        try:
            t0_total = time.time()

            X_lp, info_lp = solve_alignment(
                I=I_sp, c=c_sp, m_i=m_i_sp, m_f=m_f_sp, n=n_bound,
                method=lp_solver,
                model="netflow_lp",
                time_limit=time_limit
            )

            t_total = time.time() - t0_total

            obj_lp = float(info_lp.get('objective', math.inf))
            t_rg_build = float(info_lp.get('build_time', 0.0))
            t_lp_solve = float(info_lp.get('solve_time', 0.0))

            if math.isfinite(obj_lp) and math.isfinite(cost_astar):
                if abs(obj_lp - cost_astar) < 0.01:
                    stats_lp.add_solved(
                        solve_time=t_total,
                        rg_time=t_rg_build,
                        lp_time=t_lp_solve
                    )
                    trace_result['lp_status'] = 'optimal'
                else:
                    trace_result['lp_status'] = 'suboptimal'

                trace_result['lp_objective'] = obj_lp
                trace_result['lp_rg_build_time'] = t_rg_build
                trace_result['lp_solve_time'] = t_lp_solve
                trace_result['lp_total_time'] = t_total
                trace_result['lp_num_nodes'] = info_lp.get('num_nodes', 0)
                trace_result['lp_num_edges'] = info_lp.get('num_edges', 0)
            else:
                trace_result['lp_status'] = 'failed'

        except Exception as e:
            print(f"  LP ERROR: {e}")
            trace_result['lp_status'] = 'error'
            trace_result['lp_error'] = str(e)

                             
        if use_trace_cache:
            save_trace_results_to_cache(
                dataset_path, labels, min_fitness, target_precision, trace_result
            )

        per_trace_results.append(trace_result)
                             
    print(f"\n{'=' * 70}")
    print(f"Results Summary")
    print(f"{'=' * 70}")

    summary = {
        'dataset': os.path.basename(dataset_path),
        'n_traces': n_traces,
        'avg_trace_length': avg_trace_length,
        'fitness': fitness,
        'precision': precision,
        'model_places': len(net.places),
        'model_transitions': len(net.transitions),
        'time_limit': time_limit,
        'solvers': {
            'milp': milp_solver,
            'lp': lp_solver,
        },
        'cache_stats': {
            'total_traces': cache_stats.total_traces,
            'cache_hits': cache_stats.cache_hits,
            'cache_misses': cache_stats.cache_misses,
            'hit_rate': cache_stats.hit_rate(),
        },
        'methods': {
            'A*': stats_astar.get_metrics(),
            'MILP': stats_milp.get_metrics(),
            'LP': stats_lp.get_metrics(),
        },
        'per_trace': per_trace_results,
    }

                         
    print(f"\nDataset: {summary['dataset']}")
    print(f"Traces: {n_traces}, Avg length: {avg_trace_length:.1f}")
    print(f"Model: Fitness={fitness:.3f}, Precision={precision:.3f}")
    print(f"Solvers: MILP={milp_solver}, LP={lp_solver}")

    print(f"\nCache Performance:")
    print(f"  Total traces: {cache_stats.total_traces}")
    print(f"  Cache hits: {cache_stats.cache_hits} ({cache_stats.hit_rate():.1f}%)")
    print(f"  Cache misses: {cache_stats.cache_misses}")

    print(f"\nMethod Performance:")
    print(f"{'Method':<15} {'% Optimal':<12} {'Avg Time (s)':<15}")
    print(f"{'-' * 45}")

    for method_name, metrics in summary['methods'].items():
        pct = metrics['pct_optimal']
        avg_t = metrics.get('avg_solve_time')
        if avg_t is not None:
            print(f"{method_name:<15} {pct:>6.1f}%      {avg_t:>8.4f}")
        else:
            print(f"{method_name:<15} {pct:>6.1f}%      {'N/A':>8}")

                               
    lp_metrics = summary['methods']['LP']
    if 'avg_rg_build_time' in lp_metrics:
        print(f"\nLP Timing Breakdown:")
        print(f"  Avg RG build time: {lp_metrics['avg_rg_build_time']:.4f}s")
        print(f"  Avg LP solve time: {lp_metrics['avg_lp_solve_time']:.4f}s")
        print(f"  Avg total time: {lp_metrics['avg_total_time']:.4f}s")

    return summary


def format_results_table(all_results: List[Dict[str, Any]]) -> pd.DataFrame:
\
\
\
\
\
\
       
    rows = []

    for result in all_results:
        astar_metrics = result['methods']['A*']
        milp_metrics = result['methods']['MILP']
        lp_metrics = result['methods']['LP']
        cache_stats = result.get('cache_stats', {})

        row = {
            'Dataset': result['dataset'],
            'N_Traces': result['n_traces'],
            'Avg_Trace_Length': f"{result['avg_trace_length']:.1f}",
            'Model_Fitness': f"{result['fitness']:.3f}",
            'Model_Precision': f"{result['precision']:.3f}",
            'Cache_Hit_Rate': f"{cache_stats.get('hit_rate', 0):.1f}%",

            'A*_Pct_Optimal': f"{astar_metrics['pct_optimal']:.1f}%",
            'A*_Avg_Time': f"{astar_metrics['avg_solve_time']:.4f}" if astar_metrics['avg_solve_time'] else "N/A",

            'MILP_Pct_Optimal': f"{milp_metrics['pct_optimal']:.1f}%",
            'MILP_Avg_Time': f"{milp_metrics['avg_solve_time']:.4f}" if milp_metrics['avg_solve_time'] else "N/A",

            'LP_Pct_Optimal': f"{lp_metrics['pct_optimal']:.1f}%",
        }

                                 
        if 'avg_rg_build_time' in lp_metrics:
            row['LP_Avg_RG_Build_Time'] = f"{lp_metrics['avg_rg_build_time']:.4f}"
            row['LP_Avg_LP_Solve_Time'] = f"{lp_metrics['avg_lp_solve_time']:.4f}"
            row['LP_Avg_Total_Time'] = f"{lp_metrics['avg_total_time']:.4f}"
        else:
            row['LP_Avg_RG_Build_Time'] = "N/A"
            row['LP_Avg_LP_Solve_Time'] = "N/A"
            row['LP_Avg_Total_Time'] = "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced conformance checking benchmark with comprehensive caching"
    )
    parser.add_argument("datasets", nargs="+", help="Paths to .xes event logs")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                        help=f"Time limit per trace in seconds (default: {DEFAULT_TIME_LIMIT})")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Number of traces to sample (default: use all)")
    parser.add_argument("--min-fitness", type=float, default=0.9,
                        help="Minimum fitness requirement for discovered model (default: 0.9)")
    parser.add_argument("--target-precision", type=float, default=0.7,
                        help="Target precision for discovered model (default: 0.7)")
    parser.add_argument("--milp-solver", type=str, default=DEFAULT_SOLVER,
                        choices=["gurobi", "pulp"],
                        help=f"Solver for MILP (default: {DEFAULT_SOLVER})")
    parser.add_argument("--lp-solver", type=str, default=DEFAULT_SOLVER,
                        choices=["gurobi", "pulp"],
                        help=f"Solver for LP (default: {DEFAULT_SOLVER})")
    parser.add_argument("--no-trace-cache", action="store_true",
                        help="Disable trace-level caching")
    parser.add_argument("--out", type=str, default="benchmark_results.json",
                        help="Output JSON file path")
    parser.add_argument("--csv", type=str, default="benchmark_summary.csv",
                        help="Output CSV file path")
    parser.add_argument("--table", type=str, default="benchmark_table.csv",
                        help="Output formatted table CSV path")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear cache before running")

    args = parser.parse_args()

                              
    if args.clear_cache:
        from experiments.cache_manager import clear_cache
        print("Clearing cache...")
        clear_cache("all")
        print()

                                
    print(f"Solver Configuration:")
    print(f"  MILP solver: {args.milp_solver}")
    print(f"  LP solver: {args.lp_solver}")

                                         
    if args.milp_solver == "gurobi" or args.lp_solver == "gurobi":
        try:
            import gurobipy as gp
            m = gp.Model()
            m.Params.OutputFlag = 0
            print(f"  ✓ Gurobi available and licensed")
        except ImportError:
            print(f"  ✗ ERROR: Gurobi not installed")
            print(f"     Install: pip install gurobipy --break-system-packages")
            return
        except Exception as e:
            print(f"  ✗ ERROR: Gurobi license issue: {e}")
            print(f"     Get license: https://www.gurobi.com/academia/")
            return

                     
    all_results = []

    for dataset_path in args.datasets:
        try:
            result = run_dataset_experiment(
                dataset_path=dataset_path,
                time_limit=args.time_limit,
                sample_size=args.sample_size,
                min_fitness=args.min_fitness,
                target_precision=args.target_precision,
                milp_solver=args.milp_solver,
                lp_solver=args.lp_solver,
                use_trace_cache=not args.no_trace_cache,
            )
            all_results.append(result)

        except Exception as e:
            print(f"\nError processing {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("No results to save!")
        return

                                
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to: {args.out}")

                                     
    table_df = format_results_table(all_results)
    table_df.to_csv(args.table, index=False)
    print(f"Saved formatted table to: {args.table}")

                       
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS TABLE")
    print(f"{'=' * 70}\n")
    print(table_df.to_string(index=False))

    print(f"\n{'=' * 70}")
    print("Experiment completed successfully!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()