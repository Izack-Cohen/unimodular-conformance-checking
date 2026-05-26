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
\
\
\
\
\
   

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List
import sys

import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter

                                     
sys.path.append(str(Path(__file__).parent.parent))
from models.model_cache import ModelCache

                                                        
try:
    from models.model_transition_cache import ModelTransitionCache
    HAS_TRANSITION_CACHE = True
except ImportError:
    HAS_TRANSITION_CACHE = False
    ModelTransitionCache = None


def load_event_log(log_path):
                                                   
    log_path = Path(log_path)

    if log_path.suffix.lower() == ".xes":
        pkl_path = log_path.with_suffix(log_path.suffix + ".pkl")
    else:
        pkl_path = log_path.with_suffix(".pkl")

    if pkl_path.exists():
        print(f"Loading from cache: {pkl_path}")
        with open(pkl_path, "rb") as f:
            log = pickle.load(f)

        if isinstance(log, pd.DataFrame):
            log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)

        print(f"  Loaded {len(log)} traces")
        return log

    print(f"Loading from XES: {log_path}")
    log = pm4py.read_xes(str(log_path))

    if isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)

    print(f"  Loaded {len(log)} traces")

    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(log, f)
    except Exception:
        pass                                  

    return log


def find_model_files(models_dir: Path, dataset_name: str) -> List[Path]:
                                                          
    pattern = f"{dataset_name}_f*_p*_model.pkl"
    model_files = sorted(models_dir.glob(pattern))
    return model_files


def load_model(model_path: Path) -> tuple:
                                        
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    return model_data["net"], model_data["im"], model_data["fm"]


def extract_model_info_from_filename(filename: str) -> Dict:
\
\
\
\
\
       
    parts = filename.replace(".pkl", "").split("_")

    info = {
        "filename": filename,
        "fitness": None,
        "precision": None,
        "method": "Unknown"
    }

    for part in parts:
        if part.startswith("f") and "." in part:
            try:
                info["fitness"] = float(part[1:])
            except:
                pass
        elif part.startswith("p") and "." in part:
            try:
                info["precision"] = float(part[1:])
            except:
                pass
        elif part in ["IMf", "IMd", "IM", "Heuristic", "Alpha"]:
            info["method"] = part

    return info


def extract_trace_labels(trace):
                                               
    labels = []
    for event in trace:
        if "concept:name" in event:
            labels.append(event["concept:name"])
        elif "Activity" in event:
            labels.append(event["Activity"])
    return labels


def calculate_adaptive_max_depth(trace_length: int, model_fitness: float, model_precision: float) -> int:
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
       
    base_depth = trace_length

                                                                
    fitness_multiplier = 1.0 + (1.0 - model_fitness)

                                                                  
    if model_precision >= 0.75:
                                                                    
        precision_multiplier = 1.15
    elif model_precision >= 0.65:
                                                      
        precision_multiplier = 1.35
    else:
                                                             
        precision_multiplier = 1.7

                     
    max_depth = int(base_depth * fitness_multiplier * precision_multiplier)

                   
    min_depth = trace_length + max(5, int(trace_length * 0.2))
    max_safe = trace_length * 3

    max_depth = min(max(max_depth, min_depth), max_safe)

    return max_depth


def calculate_max_depth_fitness_based(trace_length: int, model_fitness: float) -> int:
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
       
                                                                             
    max_depth = trace_length * 10
    
                                   
    return max(max_depth, 100)


def run_astar_alignment(trace_labels, net, im, fm, time_limit=30.0):
                           
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

                                   
    from pm4py.objects.log.obj import Trace, Event
    trace_obj = Trace()
    for label in trace_labels:
        event = Event()
        event["concept:name"] = label
        trace_obj.append(event)

    try:
        start_time = time.time()
        alignment = alignments.apply(
            trace_obj,
            net,
            im,
            fm,
            parameters={
                alignments.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: False
            }
        )
        elapsed = time.time() - start_time

                                                                   
        if alignment is None:
            print(f"      A* returned None (model may have structural issues)")
            return None, elapsed, "failed"

        cost = alignment.get('cost')
        if cost is None:
            print(f"      A* result missing 'cost' field")
            return None, elapsed, "failed"

                                                 
                                                     
                                                
                                                          
        normalized_cost = cost / 10000.0

        return normalized_cost, elapsed, "optimal"

    except Exception as e:
        print(f"      A* failed: {e}")
        return None, 0.0, "error"


def run_lp_alignment(trace_labels, net, im, fm, model_fitness, model_precision=None, time_limit=30.0,
                     model_cache=None, model_transition_cache_obj=None, rg_timeout=None):
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
       
    from models.sync_product_incidence import build_sync_product_incidence
    from solvers.netflow_lp_solver_optimized import solve_alignment_netflow_lp_optimized

    try:
                                                                   
        sync_start = time.time()
        I, c, m_i, m_f, num_states = build_sync_product_incidence(
            net, im, fm, trace_labels, model_cache=model_cache
        )
        sync_time = time.time() - sync_start

                                                                                        
        max_depth = calculate_max_depth_fitness_based(len(trace_labels), model_fitness)

                                                                                  
        X, info = solve_alignment_netflow_lp_optimized(
            I, c, m_i, m_f,
            max_depth=max_depth,
            time_limit=time_limit,
            rg_timeout=rg_timeout,
            enable_optimizations=True,
            model_cache=model_transition_cache_obj,
            trace_labels=trace_labels
        )

                                                              
        if info.get("status") in ("rg_timeout", "rg_no_path"):
            rg_build_time = info.get("build_time", 0.0)
            rg_nodes = info.get("num_nodes", 0)
            rg_edges = info.get("num_edges", 0)
            return None, sync_time + rg_build_time, sync_time, rg_build_time, 0.0, rg_nodes, rg_edges, info.get("status")

                         
        cost = info.get("objective", None)
        rg_nodes = info.get("num_nodes", 0)
        rg_edges = info.get("num_edges", 0)

                                                                    
        rg_build_time = info.get("build_time", 0.0)
        lp_solve_time = info.get("solve_time", 0.0)

                                                                      
        total_time = sync_time + rg_build_time + lp_solve_time

        if cost is not None:
            return cost, total_time, sync_time, rg_build_time, lp_solve_time, rg_nodes, rg_edges, "optimal"
        else:
            return None, total_time, sync_time, rg_build_time, lp_solve_time, 0, 0, "failed"

    except Exception as e:
        print(f"      LP failed: {e}")
        return None, 0.0, 0.0, 0.0, 0.0, 0, 0, "error"


def run_experiment_with_model(
    log,
    net,
    im,
    fm,
    model_info: Dict,
    max_traces: int = None,
    time_limit: float = 30.0,
    rg_timeout: float = None
) -> Dict:
                                                                           

    print(f"\n  Testing model: f={model_info.get('fitness', 0):.3f} "
          f"p={model_info.get('precision', 0):.3f} "
          f"({model_info.get('method', 'unknown')})")

                                                                 
    model_cache = ModelCache(net)

                                                                                       
    model_transition_cache_obj = None
    if HAS_TRANSITION_CACHE:
        model_transition_cache_obj = ModelTransitionCache(net, im, fm)

                                                                             
    model_fitness = model_info.get('fitness', 0.9)                                   
    model_precision = model_info.get('precision', 0.7)                                   

    results = {
        "model_info": model_info,
        "model_stats": {
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs)
        },
        "per_trace": []
    }

    traces_to_process = min(len(log), max_traces) if max_traces else len(log)

    astar_wins = 0
    lp_wins = 0
    similar = 0
    rg_timeouts = 0

    for idx, trace in enumerate(log):
        if idx >= traces_to_process:
            break

        trace_labels = extract_trace_labels(trace)
        trace_length = len(trace_labels)

                
        astar_cost, astar_time, astar_status = run_astar_alignment(
            trace_labels, net, im, fm, time_limit
        )

                
        lp_cost, lp_time, lp_sync, lp_rg_build, lp_solve, lp_nodes, lp_edges, lp_status = run_lp_alignment(
            trace_labels, net, im, fm, model_fitness, model_precision, time_limit, model_cache,
            model_transition_cache_obj,
            rg_timeout=rg_timeout
        )

                                                    
        if lp_status in ("rg_timeout", "rg_no_path"):
            rg_timeouts += 1
            if idx == 0:
                if lp_status == "rg_no_path":
                    print(f"    First trace (len={trace_length}): A*={astar_time:.3f}s, LP={lp_status} - final marking unreachable ({lp_nodes} nodes in {lp_rg_build:.2f}s)")
                else:
                    print(f"    First trace (len={trace_length}): A*={astar_time:.3f}s, LP={lp_status} after {lp_rg_build:.2f}s ({lp_nodes} nodes)")
        elif idx == 0:
            print(f"    First trace (len={trace_length}): A*={astar_time:.3f}s, LP={lp_time:.3f}s "
                  f"(RG: {lp_rg_build:.3f}s, Solve: {lp_solve:.3f}s, {lp_nodes} nodes)")

                                         
        if (idx + 1) % 100 == 0:
            print(f"    Processed {idx + 1}/{traces_to_process} traces "
                  f"(A* wins: {astar_wins}, LP wins: {lp_wins}, RG timeouts: {rg_timeouts})")

                 
        if astar_status == "optimal" and lp_status == "optimal":
            if astar_time < lp_time * 0.95:
                astar_wins += 1
            elif lp_time < astar_time * 0.95:
                lp_wins += 1
            else:
                similar += 1

        results["per_trace"].append({
            "trace_idx": idx,
            "trace_length": trace_length,
            "astar_cost": astar_cost,
            "astar_time": astar_time,
            "astar_status": astar_status,
            "lp_cost": lp_cost,
            "lp_time": lp_time,
            "lp_sync_time": lp_sync,
            "lp_rg_build_time": lp_rg_build,
            "lp_solve_time": lp_solve,
            "lp_nodes": lp_nodes,
            "lp_edges": lp_edges,
            "lp_status": lp_status
        })

    results["summary"] = {
        "traces_processed": traces_to_process,
        "astar_wins": astar_wins,
        "lp_wins": lp_wins,
        "similar": similar,
        "rg_timeouts": rg_timeouts,
        "astar_win_rate": astar_wins / traces_to_process * 100 if traces_to_process > 0 else 0,
        "lp_win_rate": lp_wins / traces_to_process * 100 if traces_to_process > 0 else 0
    }

    if rg_timeouts > 0:
        print(f"    Results: A* wins {astar_wins}, LP wins {lp_wins}, Similar {similar}, RG timeouts {rg_timeouts}")
    else:
        print(f"    Results: A* wins {astar_wins}, LP wins {lp_wins}, Similar {similar}")

    return results


def run_all_model_experiments(
    log,
    models_dir: Path,
    dataset_name: str,
    output_dir: Path,
    max_traces: int = None,
    time_limit: float = 30.0,
    min_precision: float = 0.4,
    min_fitness: float = 0.0,
    rg_timeout: float = None,
    single_model: bool = False
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
       

    print(f"\n{'='*80}")
    if single_model:
        print(f"RUNNING EXPERIMENT FOR SINGLE MODEL")
    else:
        print(f"RUNNING EXPERIMENTS FOR ALL MODEL VARIANTS")
    print(f"{'='*80}")

                     
    if single_model:
        model_files = [(models_dir, {})]
        print(f"Using single model: {models_dir.name}")
    else:
        model_files = find_model_files(models_dir, dataset_name)
        
        if not model_files:
            print(f"❌ No model files found for dataset: {dataset_name}")
            print(f"   Looking in: {models_dir}")
            print(f"   Pattern: {dataset_name}_f*_p*_model.pkl")
            return

        print(f"Found {len(model_files)} models")
        model_files = [(f, {}) for f in model_files]

                                                                       
    if not single_model:
        filtered_models = []
        skipped_by_precision = []
        skipped_by_fitness = []

        for model_path, _ in model_files:
            model_info = extract_model_info_from_filename(model_path.name)
            precision = model_info.get('precision')
            fitness = model_info.get('fitness')
            
            precision_ok = precision is None or precision >= min_precision
            fitness_ok = fitness is None or fitness >= min_fitness
            
            if precision_ok and fitness_ok:
                filtered_models.append((model_path, model_info))
            else:
                if not precision_ok:
                    skipped_by_precision.append((model_path.name, precision))
                if not fitness_ok:
                    skipped_by_fitness.append((model_path.name, fitness))

        print(f"✅ Models passing filters: {len(filtered_models)}/{len(model_files)}")
        print(f"   Precision ≥ {min_precision}, Fitness ≥ {min_fitness}")
        
        if skipped_by_precision:
            print(f"⏭️  Skipped {len(skipped_by_precision)} models with precision < {min_precision}:")
            for name, prec in skipped_by_precision:
                print(f"   - {name} (p={prec:.3f})")
        
        if skipped_by_fitness:
            print(f"⏭️  Skipped {len(skipped_by_fitness)} models with fitness < {min_fitness}:")
            for name, fit in skipped_by_fitness:
                print(f"   - {name} (f={fit:.3f})")

        model_files = filtered_models
    else:
        model_path, _ = model_files[0]
        model_info = extract_model_info_from_filename(model_path.name)
        model_files = [(model_path, model_info)]

    print(f"\nTesting {len(model_files)} model(s)...")

    all_results = []

    for i, (model_path, model_info) in enumerate(model_files, 1):
        if single_model:
            print(f"\n[1/1] Testing: {model_path.name}")
        else:
            print(f"\n[{i}/{len(model_files)}] Testing: {model_path.name}")

        model_info["model_file"] = model_path.name

        try:
            net, im, fm = load_model(model_path)
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            continue

        try:
            result = run_experiment_with_model(
                log, net, im, fm, model_info,
                max_traces=max_traces,
                time_limit=time_limit,
                rg_timeout=rg_timeout
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ❌ Experiment failed: {e}")
            continue

                  
    output_path = output_dir / f"{dataset_name}_model_variants_comparison.json"

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"RESULTS SAVED")
    print(f"{'='*80}")
    print(f"📄 Detailed JSON (per-trace data): {output_path}")
    print(f"   Contains:")
    print(f"   - Per-trace costs, times, and statuses for A* and LP")
    print(f"   - LP time breakdown (sync_time, solve_time)")
    print(f"   - Reachability graph statistics (nodes, edges)")
    print(f"   - Model information (fitness, precision, method)")

                             
    generate_summary_report(all_results, output_dir, dataset_name)

    return all_results


def generate_summary_report(results: List[Dict], output_dir: Path, dataset_name: str):
                                                               

    print(f"\n{'='*80}")
    print(f"SUMMARY REPORT")
    print(f"{'='*80}")

    summary_data = []

    for result in results:
        model_info = result["model_info"]
        summary = result["summary"]
        model_stats = result["model_stats"]
        per_trace = result.get("per_trace", [])

                                              
        lp_sync_times = [t["lp_sync_time"] for t in per_trace if t.get("lp_status") == "optimal"]
        lp_rg_build_times = [t.get("lp_rg_build_time", 0) for t in per_trace if t.get("lp_status") == "optimal"]
        lp_solve_times = [t["lp_solve_time"] for t in per_trace if t.get("lp_status") == "optimal"]
        lp_total_times = [t["lp_time"] for t in per_trace if t.get("lp_status") == "optimal"]

        avg_lp_sync = sum(lp_sync_times) / len(lp_sync_times) if lp_sync_times else 0
        avg_lp_rg_build = sum(lp_rg_build_times) / len(lp_rg_build_times) if lp_rg_build_times else 0
        avg_lp_solve = sum(lp_solve_times) / len(lp_solve_times) if lp_solve_times else 0
        avg_lp_total = sum(lp_total_times) / len(lp_total_times) if lp_total_times else 0

        lp_sync_pct = (avg_lp_sync / avg_lp_total * 100) if avg_lp_total > 0 else 0
        lp_rg_build_pct = (avg_lp_rg_build / avg_lp_total * 100) if avg_lp_total > 0 else 0
        lp_solve_pct = (avg_lp_solve / avg_lp_total * 100) if avg_lp_total > 0 else 0

                                   
        avg_rg_nodes = sum(t.get("lp_nodes", 0) for t in per_trace) / len(per_trace) if per_trace else 0
        avg_rg_edges = sum(t.get("lp_edges", 0) for t in per_trace) / len(per_trace) if per_trace else 0

        summary_data.append({
            "model_file": model_info.get("filename", "unknown"),
            "method": model_info.get("method", "unknown"),
            "fitness": model_info.get("fitness", 0),
            "precision": model_info.get("precision", 0),
            "places": model_stats.get("places", 0),
            "transitions": model_stats.get("transitions", 0),
            "complexity": model_stats.get("places", 0) * model_stats.get("transitions", 0),
            "astar_wins": summary["astar_wins"],
            "lp_wins": summary["lp_wins"],
            "similar": summary["similar"],
            "astar_win_rate": summary["astar_win_rate"],
            "lp_win_rate": summary["lp_win_rate"],
            "traces_tested": summary["traces_processed"],
            "avg_lp_sync_time": avg_lp_sync,
            "avg_lp_rg_build_time": avg_lp_rg_build,
            "avg_lp_solve_time": avg_lp_solve,
            "avg_lp_total_time": avg_lp_total,
            "lp_sync_pct": lp_sync_pct,
            "lp_rg_build_pct": lp_rg_build_pct,
            "lp_solve_pct": lp_solve_pct,
            "avg_rg_nodes": avg_rg_nodes,
            "avg_rg_edges": avg_rg_edges
        })

    df = pd.DataFrame(summary_data)

                                     
    df = df.sort_values("precision")

              
    csv_path = output_dir / f"{dataset_name}_model_variants_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Summary CSV saved to: {csv_path}")

                 
    print(f"\n📊 Performance Summary by Model:")
    print(f"\n{df.to_string(index=False)}")

                  
    print(f"\n🔍 Key Insights:")

                                
    print(f"\n   LP Time Breakdown (average across all models):")
    avg_sync_pct = df["lp_sync_pct"].mean()
    avg_rg_build_pct = df["lp_rg_build_pct"].mean()
    avg_solve_pct = df["lp_solve_pct"].mean()
    print(f"      Sync product building: {avg_sync_pct:.1f}% of LP time")
    print(f"      RG building: {avg_rg_build_pct:.1f}% of LP time")
    print(f"      LP solving: {avg_solve_pct:.1f}% of LP time")

    if avg_rg_build_pct > 50:
        print(f"      → RG building is the bottleneck (optimize reachability graph construction)")
    elif avg_solve_pct > 70:
        print(f"      → LP solving is the bottleneck (optimize linear programming)")
    else:
        print(f"      → Balanced time distribution")

                      
    print(f"\n   Average Reachability Graph Sizes:")
    print(f"      Nodes: {df['avg_rg_nodes'].mean():.0f}")
    print(f"      Edges: {df['avg_rg_edges'].mean():.0f}")

                                               
    if len(df) >= 3:
        import numpy as np
        corr = np.corrcoef(df["precision"], df["lp_win_rate"])[0, 1]
        print(f"\n   Correlation (precision vs LP win rate): {corr:.3f}")

        if corr > 0.5:
            print(f"      → Strong positive correlation: Higher precision favors LP")
        elif corr < -0.5:
            print(f"      → Strong negative correlation: Higher precision favors A*")
        else:
            print(f"      → Weak correlation: No clear precision effect")

                       
    best_lp = df.loc[df["lp_win_rate"].idxmax()]
    print(f"\n   Best model for LP:")
    print(f"      {best_lp['model_file']}")
    print(f"      f={best_lp['fitness']:.3f}, p={best_lp['precision']:.3f}")
    print(f"      LP wins: {best_lp['lp_win_rate']:.1f}%")

                       
    best_astar = df.loc[df["astar_win_rate"].idxmax()]
    print(f"\n   Best model for A*:")
    print(f"      {best_astar['model_file']}")
    print(f"      f={best_astar['fitness']:.3f}, p={best_astar['precision']:.3f}")
    print(f"      A* wins: {best_astar['astar_win_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run A* vs LP experiments with discovered model variants"
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to XES event log file"
    )
    parser.add_argument(
        "models_dir",
        type=str,
        help="Directory containing model .pkl files, or path to single .pkl file (use with --single-model)"
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Treat models_dir as a single model file path instead of a directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="alignment_checker/results",
        help="Directory to save results (default: alignment_checker/results)"
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=None,
        help="Maximum traces to process (default: all)"
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Time limit per trace in seconds (default: 30)"
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.4,
        help="Minimum model precision to include (default: 0.4, skips low-quality models)"
    )
    parser.add_argument(
        "--min-fitness",
        type=float,
        default=0.0,
        help="Minimum model fitness to include (default: 0.0, no fitness filtering)"
    )
    parser.add_argument(
        "--rg-timeout",
        type=float,
        default=None,
        help="Maximum seconds for RG building per trace. Skips LP if exceeded. (default: no limit)"
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {log_path}")
        return

    models_path = Path(args.models_dir)
    
    if args.single_model:
        if not models_path.exists():
            print(f"❌ Error: Model file not found: {models_path}")
            return
        if not models_path.is_file():
            print(f"❌ Error: Expected a file in --single-model mode, got: {models_path}")
            return
        if models_path.suffix != '.pkl':
            print(f"❌ Error: Model file must be a .pkl file, got: {models_path}")
            return
    else:
        if not models_path.exists():
            print(f"❌ Error: Models directory not found: {models_path}")
            return
        if not models_path.is_dir():
            print(f"❌ Error: Expected a directory, got file: {models_path}")
            print(f"   Hint: Use --single-model flag if you want to test a single model file")
            return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = log_path.stem

    print(f"{'='*80}")
    print(f"MODEL VARIANT EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Dataset: {log_path}")
    if args.single_model:
        print(f"Model file: {models_path}")
        print(f"Mode: Single model")
    else:
        print(f"Models directory: {models_path}")
        print(f"Mode: All models in directory")
    print(f"Output directory: {output_dir}")
    print(f"Max traces: {args.max_traces if args.max_traces else 'all'}")
    print(f"Time limit: {args.time_limit}s per trace")
    print(f"RG timeout: {args.rg_timeout}s per trace" if args.rg_timeout else "RG timeout: none")
    print(f"Min precision filter: ≥{args.min_precision}")
    print(f"Min fitness filter: ≥{args.min_fitness}")
    print(f"{'='*80}")

              
    start_time = time.time()
    log = load_event_log(log_path)
    load_time = time.time() - start_time
    print(f"✅ Log loaded in {load_time:.2f}s")

                     
    experiment_start = time.time()
    results = run_all_model_experiments(
        log,
        models_path,
        dataset_name,
        output_dir,
        max_traces=args.max_traces,
        time_limit=args.time_limit,
        min_precision=args.min_precision,
        min_fitness=args.min_fitness,
        rg_timeout=args.rg_timeout,
        single_model=args.single_model
    )
    experiment_time = time.time() - experiment_start

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETE - Total time: {total_time:.2f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
