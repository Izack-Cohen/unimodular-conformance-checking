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
from dataclasses import dataclass, asdict
from typing import List, Tuple
import random

import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter


@dataclass
class DiscoveredModel:
                                                                    
    dataset_name: str
    discovery_method: str
    noise_threshold: float
    dependency_threshold: float                       
    places: int
    transitions: int
    arcs: int
    fitness: float
    precision: float
    discovery_time: float
    evaluation_time: float
    filename: str

    def score(self) -> float:
                                                                  
                                                               
        return self.fitness * 0.6 + self.precision * 0.4

    def to_dict(self):
                                                   
        return asdict(self)


def load_event_log(log_path):
\
\
\
       
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
            print("  Converting DataFrame to EventLog...")
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
        print(f"  Cached to: {pkl_path}")
    except Exception as e:
        print(f"  Warning: could not cache: {e}")

    return log


def sample_log(log, max_traces=2000, seed=42):
                                                                
    if len(log) <= max_traces:
        return log

    print(f"  Sampling {max_traces} traces for evaluation (from {len(log)} total)")
    rng = random.Random(seed)
    indices = list(range(len(log)))
    rng.shuffle(indices)
    indices = sorted(indices[:max_traces])

    sampled = EventLog()
    for i in indices:
        sampled.append(log[i])

    return sampled


def calculate_quality_metrics(log, net, im, fm):
                                                                   
    try:
        fitness_result = pm4py.fitness_token_based_replay(log, net, im, fm)
        precision = pm4py.precision_token_based_replay(log, net, im, fm)

        fitness = fitness_result["average_trace_fitness"]
        return fitness, precision
    except Exception as e:
        print(f"  ⚠ Error calculating metrics: {e}")
        return 0.0, 0.0


def count_model_elements(net, im, fm):
                                                           
    places = len(net.places)
    transitions = len(net.transitions)
    arcs = len(net.arcs)
    return places, transitions, arcs


def discover_inductive_miner_variant(
    log,
    noise_threshold: float,
    variant_name: str = "IMf",
    eval_log=None
) -> Tuple:
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
                                                                           
                                                                           
        net, im, fm = pm4py.discover_petri_net_inductive(
            log,
            noise_threshold=noise_threshold
        )
        discovery_time = time.time() - start_time
    except TypeError:
                                                                          
        print(f"    ⚠ noise_threshold not supported; falling back to default")
        net, im, fm = pm4py.discover_petri_net_inductive(log)
        discovery_time = time.time() - start_time
    except Exception as e:
        print(f"    ⚠ Discovery failed: {e}")
        return None

                
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = calculate_quality_metrics(eval_log, net, im, fm)
    eval_time = time.time() - eval_start

    return net, im, fm, discovery_time, eval_time, fitness, precision


def discover_heuristic_miner(
    log,
    dependency_threshold: float = 0.8,
    and_threshold: float = 0.8,
    eval_log=None
) -> Tuple:
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
       
    import pm4py

               
    start_time = time.time()
    try:
                                                    
        net, im, fm = pm4py.discover_petri_net_heuristics(
            log,
            dependency_threshold=dependency_threshold,
            and_threshold=and_threshold
        )
        discovery_time = time.time() - start_time

    except Exception as e:
        print(f"{str(e)}")
        return None

                
    try:
        eval_start = time.time()
        eval_log = eval_log if eval_log is not None else log
        fitness, precision = calculate_quality_metrics(eval_log, net, im, fm)
        eval_time = time.time() - eval_start
    except Exception as e:
        print(f" (eval failed: {e})")
        return None

    return net, im, fm, discovery_time, eval_time, fitness, precision


def discover_alpha_miner(log, eval_log=None) -> Tuple:
\
\
\
\
\
\
       
    from pm4py.algo.discovery.alpha import algorithm as alpha_miner

               
    start_time = time.time()
    try:
        net, im, fm = alpha_miner.apply(log)
        discovery_time = time.time() - start_time
    except Exception as e:
        print(f"    ⚠ Discovery failed: {e}")
        return None

                
    eval_start = time.time()
    eval_log = eval_log if eval_log is not None else log
    fitness, precision = calculate_quality_metrics(eval_log, net, im, fm)
    eval_time = time.time() - eval_start

    return net, im, fm, discovery_time, eval_time, fitness, precision


def explore_model_space(
    log,
    dataset_name: str,
    output_dir: Path,
    max_eval_traces: int = None
) -> List[DiscoveredModel]:
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
    print(f"EXPLORING MODEL SPACE FOR: {dataset_name}")
    print(f"{'='*80}")

                                                                          
    if max_eval_traces is not None:
        eval_log = sample_log(log, max_traces=max_eval_traces)
    else:
        eval_log = log
        print(f"  Using full log for evaluation ({len(log)} traces)")

    discovered_models = []

                                                                               
                                                         
                                                                               
    print(f"\n1️⃣  Inductive Miner (IMf) with varying noise thresholds")
    print(f"   Strategy: Explore noise from 0.0 (restrictive) to 0.8 (permissive)")

                                    
    noise_grid = [
        0.00, 0.05, 0.10, 0.15, 0.20,
        0.30, 0.40, 0.50, 0.60, 0.70, 0.80
    ]

    for noise in noise_grid:
        print(f"\n   Trying noise={noise:.3f}...", end=" ")

        result = discover_inductive_miner_variant(
            log, noise, variant_name="IMf", eval_log=eval_log
        )

        if result is None:
            print("❌ Failed")
            continue

        net, im, fm, disc_time, eval_time, fitness, precision = result
        places, transitions, arcs = count_model_elements(net, im, fm)

        print(f"✓ f={fitness:.3f} p={precision:.3f} n={noise:.3f} ({places}p, {transitions}t)")

                    
        filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_IMf_n{noise:.3f}_model.pkl"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump({"net": net, "im": im, "fm": fm}, f)

        model_info = DiscoveredModel(
            dataset_name=dataset_name,
            discovery_method=f"InductiveMiner_IMf",
            noise_threshold=noise,
            dependency_threshold=-1.0,              
            places=places,
            transitions=transitions,
            arcs=arcs,
            fitness=fitness,
            precision=precision,
            discovery_time=disc_time,
            evaluation_time=eval_time,
            filename=filename
        )

        discovered_models.append(model_info)

                                                                               
                                                        
                                                                               
    print(f"\n2️⃣  Heuristic Miner with varying dependency thresholds")
    print(f"   Strategy: Explore dependency threshold from 0.5 to 0.95")
    print(f"   Note: Using fresh XES read for HM compatibility")

    dependency_grid = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

                                                             
                                                                       
    import os
    xes_path = None
    for possible_path in [
        f"data/{dataset_name}.xes",
        f"data\\{dataset_name}.xes",
        f"{dataset_name}.xes",
    ]:
        if os.path.exists(possible_path):
            xes_path = possible_path
            break

    if xes_path:
        print(f"   Loading fresh XES for HM: {xes_path}")
        try:
            hm_log = pm4py.read_xes(xes_path)
            print(f"   ✓ Loaded {len(hm_log)} traces for HM")
        except Exception as e:
            print(f"   ⚠ Could not load XES: {e}")
            hm_log = log                           
    else:
        print(f"   ⚠ XES file not found, using cached log")
        hm_log = log

    for dep_thresh in dependency_grid:
        print(f"\n   Trying dependency={dep_thresh:.2f}...", end=" ")

        result = discover_heuristic_miner(
            hm_log,                     
            dependency_threshold=dep_thresh,
            and_threshold=dep_thresh,
            eval_log=eval_log                                        
        )

        if result is None:
            print("❌ Failed")
            continue

        net, im, fm, disc_time, eval_time, fitness, precision = result
        places, transitions, arcs = count_model_elements(net, im, fm)

        print(f"✓ f={fitness:.3f} p={precision:.3f} ({places}p, {transitions}t)")

                    
        filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_Heuristic_d{dep_thresh:.2f}_model.pkl"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump({"net": net, "im": im, "fm": fm}, f)

        model_info = DiscoveredModel(
            dataset_name=dataset_name,
            discovery_method="HeuristicMiner",
            noise_threshold=-1.0,              
            dependency_threshold=dep_thresh,
            places=places,
            transitions=transitions,
            arcs=arcs,
            fitness=fitness,
            precision=precision,
            discovery_time=disc_time,
            evaluation_time=eval_time,
            filename=filename
        )

        discovered_models.append(model_info)

                                                                               
                                                  
                                                                               
    print(f"\n3️⃣  Alpha Miner (typically high precision)")
    print(f"   Strategy: Single discovery, usually produces restrictive models")

    print(f"\n   Discovering...", end=" ")

    result = discover_alpha_miner(log, eval_log=eval_log)

    if result is not None:
        net, im, fm, disc_time, eval_time, fitness, precision = result
        places, transitions, arcs = count_model_elements(net, im, fm)

        print(f"✓ f={fitness:.3f} p={precision:.3f} ({places}p, {transitions}t)")

                    
        filename = f"{dataset_name}_f{fitness:.4f}_p{precision:.4f}_Alpha_model.pkl"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump({"net": net, "im": im, "fm": fm}, f)

        model_info = DiscoveredModel(
            dataset_name=dataset_name,
            discovery_method="AlphaMiner",
            noise_threshold=-1.0,
            dependency_threshold=-1.0,
            places=places,
            transitions=transitions,
            arcs=arcs,
            fitness=fitness,
            precision=precision,
            discovery_time=disc_time,
            evaluation_time=eval_time,
            filename=filename
        )

        discovered_models.append(model_info)
    else:
        print("❌ Failed")

    return discovered_models


def analyze_coverage(models: List[DiscoveredModel]) -> None:
                                                                     
    print(f"\n{'='*80}")
    print(f"MODEL SPACE COVERAGE ANALYSIS")
    print(f"{'='*80}")

    if not models:
        print("  No models discovered!")
        return

                     
    models_sorted = sorted(models, key=lambda m: m.fitness, reverse=True)

    print(f"\n📊 Discovered {len(models)} models")

                   
    fitness_values = [m.fitness for m in models]
    print(f"\n   Fitness range:")
    print(f"      Min: {min(fitness_values):.3f}")
    print(f"      Max: {max(fitness_values):.3f}")
    print(f"      Mean: {sum(fitness_values)/len(fitness_values):.3f}")

                     
    precision_values = [m.precision for m in models]
    print(f"\n   Precision range:")
    print(f"      Min: {min(precision_values):.3f}")
    print(f"      Max: {max(precision_values):.3f}")
    print(f"      Mean: {sum(precision_values)/len(precision_values):.3f}")

                  
    print(f"\n   Fitness distribution:")
    fitness_bins = [0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(len(fitness_bins) - 1):
        low, high = fitness_bins[i], fitness_bins[i+1]
        count = sum(1 for f in fitness_values if low <= f < high)
        print(f"      [{low:.1f}, {high:.1f}): {count} models")

                                   
    print(f"\n   Top 5 models by quality score:")
    top_5 = sorted(models, key=lambda m: m.score(), reverse=True)[:5]
    for i, model in enumerate(top_5, 1):
        print(f"      {i}. f={model.fitness:.3f} p={model.precision:.3f} "
              f"score={model.score():.3f} ({model.discovery_method})")

                                
    print(f"\n   Coverage gaps to consider:")

                                     
    high_fitness = [m for m in models if m.fitness >= 0.85]
    if high_fitness:
        high_f_precisions = [m.precision for m in high_fitness]
        print(f"      High fitness (≥0.85): {len(high_fitness)} models, "
              f"precision range [{min(high_f_precisions):.3f}, {max(high_f_precisions):.3f}]")
    else:
        print(f"      ⚠ No high fitness models (≥0.85)")

                      
    mid_fitness = [m for m in models if 0.70 <= m.fitness < 0.85]
    if mid_fitness:
        mid_f_precisions = [m.precision for m in mid_fitness]
        print(f"      Mid fitness [0.70, 0.85): {len(mid_fitness)} models, "
              f"precision range [{min(mid_f_precisions):.3f}, {max(mid_f_precisions):.3f}]")
    else:
        print(f"      ⚠ No mid fitness models [0.70, 0.85)")

                 
    low_fitness = [m for m in models if m.fitness < 0.70]
    if low_fitness:
        low_f_precisions = [m.precision for m in low_fitness]
        print(f"      Low fitness (<0.70): {len(low_fitness)} models, "
              f"precision range [{min(low_f_precisions):.3f}, {max(low_f_precisions):.3f}]")


def save_summary(models: List[DiscoveredModel], output_dir: Path, dataset_name: str):
                                                    
    if not models:
        print("  No models to save!")
        return

    df = pd.DataFrame([m.to_dict() for m in models])

                                        
    df['quality_score'] = df['fitness'] * 0.6 + df['precision'] * 0.4
    df = df.sort_values('quality_score', ascending=False)

    summary_path = output_dir / f"{dataset_name}_discovered_models_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\n✅ Summary saved to: {summary_path}")

                                                      
    json_path = output_dir / f"{dataset_name}_discovered_models_summary.json"
    with open(json_path, 'w') as f:
        json.dump([m.to_dict() for m in models], f, indent=2)

    print(f"✅ JSON summary saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover multiple process models with varying fitness/precision"
    )
    parser.add_argument(
        "log_path",
        type=str,
        help="Path to XES event log file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="alignment_checker/results",
        help="Directory to save models (default: alignment_checker/results)"
    )
    parser.add_argument(
        "--max-eval-traces",
        type=int,
        default=None,
        help="Maximum traces for model evaluation (default: None = use full log)"
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"❌ Error: Log file not found: {log_path}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = log_path.stem

    print(f"{'='*80}")
    print(f"MODEL DISCOVERY - FITNESS/PRECISION SPACE EXPLORATION")
    print(f"{'='*80}")
    print(f"Dataset: {log_path}")
    print(f"Output directory: {output_dir}")
    eval_mode = "full log" if args.max_eval_traces is None else f"{args.max_eval_traces} traces"
    print(f"Evaluation mode: {eval_mode}")
    print(f"{'='*80}")

              
    start_time = time.time()
    log = load_event_log(log_path)
    load_time = time.time() - start_time
    print(f"✅ Log loaded in {load_time:.2f}s")

                     
    discovery_start = time.time()
    discovered_models = explore_model_space(
        log,
        dataset_name,
        output_dir,
        max_eval_traces=args.max_eval_traces
    )
    discovery_time = time.time() - discovery_start

    print(f"\n✅ Discovery complete in {discovery_time:.2f}s")

                      
    analyze_coverage(discovered_models)

                  
    save_summary(discovered_models, output_dir, dataset_name)

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETE - Total time: {total_time:.2f}s")
    print(f"Discovered {len(discovered_models)} models saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()