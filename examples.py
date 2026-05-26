#!/usr/bin/env python3
\
\
\
\
\
\
\
\
   

import sys
import time
import numpy as np
from pathlib import Path

                          
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

                             
from models.petri_net_loader import (
    load_trace_from_xes,
    discover_petri_from_log,
    compute_fitness_precision
)
from models.sync_product_incidence import build_sync_product_incidence
from solvers.a_star_solver import run_a_star_alignment
from solvers.solver_selector import solve_alignment


def example_1_basic_workflow():
                                                        
    print("=" * 70)
    print("Example 1: Basic Workflow")
    print("=" * 70)
    
                            
    trace_labels = ['a', 'b', 'c', 'd']
    print(f"\nTrace: {trace_labels}")
    
                                          
                                         
                                      
                       
    
    print("\n→ To run this example, you need:")
    print("  1. An event log (.xes file)")
    print("  2. A Petri net model (.pnml file)")
    print("\nSee example_2_with_dataset() for a complete example.")


def example_2_with_dataset(log_path):
                                                           
    print("\n" + "=" * 70)
    print("Example 2: Complete Workflow with Dataset")
    print("=" * 70)
    
                            
    print(f"\n→ Loading event log from: {log_path}")
    from pm4py.objects.log.importer.xes import importer as xes_importer
    log = xes_importer.apply(log_path)
    print(f"✓ Loaded {len(log)} traces")
    
                                      
    print("\n→ Discovering Petri net model...")
    net, im, fm = discover_petri_from_log(log, variant='imf', noise_threshold=0.2)
    
                           
    fitness, precision = compute_fitness_precision(log, net, im, fm)
    print(f"✓ Model discovered:")
    print(f"  - Places: {len(net.places)}")
    print(f"  - Transitions: {len(net.transitions)}")
    print(f"  - Fitness: {fitness:.3f}")
    print(f"  - Precision: {precision:.3f}")
    
                                     
    trace = log[0]               
    trace_labels = [event["concept:name"] for event in trace]
    print(f"\n→ Aligning trace with {len(trace_labels)} events")
    
                                       
    print("\n→ Building synchronous product...")
    start = time.time()
    I, c, m_i, m_f, transition_labels, _ = build_sync_product_incidence(
        net, im, fm, trace_labels
    )
    build_time = time.time() - start
    print(f"✓ Synchronous product built in {build_time:.3f}s")
    print(f"  - Transitions: {I.shape[1]}")
    print(f"  - Places: {I.shape[0]}")
    
                              
    print("\n→ Running A* alignment...")
    start = time.time()
    astar_result = run_a_star_alignment(trace_labels, net, im, fm)
    astar_time = time.time() - start
    print(f"✓ A* completed in {astar_time:.3f}s")
    print(f"  - Cost: {astar_result['cost']:.4f}")
    print(f"  - Alignment length: {len(astar_result['alignment'])}")
    
                                
    print("\n→ Running MILP alignment...")
    n = len(trace_labels) * 2 + 10                                   
    start = time.time()
    try:
        X_milp, model = solve_alignment(
            I, c, m_i, m_f, n,
            method="gurobi",
            model="milp",
            time_limit=30
        )
        milp_time = time.time() - start
        milp_cost = np.sum(X_milp * c.reshape(-1, 1))
        print(f"✓ MILP completed in {milp_time:.3f}s")
        print(f"  - Cost: {milp_cost:.4f}")
    except Exception as e:
        print(f"✗ MILP failed: {e}")
    
                                            
    print("\n→ Running LP (URC2) alignment...")
    start = time.time()
    try:
        result = solve_alignment(
            I, c, m_i, m_f, n,
            method="gurobi",
            model="netflow_lp",
            time_limit=30
        )
        lp_time = time.time() - start
        
        if isinstance(result, tuple) and len(result) >= 2:
            X_lp = result[0]
            info = result[1]
            lp_cost = np.sum(X_lp * c.reshape(-1, 1))
            
            print(f"✓ LP (URC2) completed in {lp_time:.3f}s")
            print(f"  - Cost: {lp_cost:.4f}")
            
            if isinstance(info, dict):
                if 'rg_build_time' in info:
                    print(f"  - RG build time: {info['rg_build_time']:.3f}s")
                if 'lp_solve_time' in info:
                    print(f"  - LP solve time: {info['lp_solve_time']:.3f}s")
                if 'n_states' in info:
                    print(f"  - Reachability graph states: {info['n_states']}")
        else:
            print(f"✓ LP completed in {lp_time:.3f}s")
    except Exception as e:
        print(f"✗ LP failed: {e}")
    
                             
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"\n{'Method':<15} {'Time (s)':<12} {'Cost':<12} {'Status'}")
    print("-" * 70)
    print(f"{'A*':<15} {astar_time:<12.3f} {astar_result['cost']:<12.4f} ✓")
    print(f"{'MILP':<15} {milp_time:<12.3f} {milp_cost:<12.4f} ✓")
    print(f"{'LP (URC2)':<15} {lp_time:<12.3f} {lp_cost:<12.4f} ✓")
    
    print("\nKey observations:")
    if abs(astar_result['cost'] - milp_cost) < 0.001 and abs(astar_result['cost'] - lp_cost) < 0.001:
        print("✓ All methods found the same optimal cost")
    
    if lp_time < astar_time:
        speedup = astar_time / lp_time
        print(f"✓ LP (URC2) is {speedup:.2f}x faster than A* on this trace")
    else:
        print("✓ A* is faster on this trace (LP advantage typically shows on longer traces)")


def example_3_batch_processing():
                                                         
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)
    
    print("\n→ For batch processing of multiple traces, use:")
    print("  1. run_experiments.py for automated batch processing")
    print("  2. experiments.enhanced_runner for programmatic batch processing")
    print("\nSee the documentation in experiments/enhanced_runner.py")


def main():
                                        
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Examples of using the conformance checking framework"
    )
    parser.add_argument(
        "--log",
        help="Path to XES event log for Example 2"
    )
    
    args = parser.parse_args()
    
                  
    print("\nConformance Checking Framework - Usage Examples")
    print("=" * 70)
    
                               
    example_1_basic_workflow()
    
                                                     
    if args.log:
        from pathlib import Path
        if Path(args.log).exists():
            example_2_with_dataset(args.log)
        else:
            print(f"\n✗ Error: Log file not found: {args.log}")
    else:
        print("\n→ To run Example 2, provide a log file:")
        print("  python examples.py --log data/your_log.xes")
    
                                 
    example_3_batch_processing()
    
    print("\n" + "=" * 70)
    print("For more examples, see:")
    print("  - quickstart.py - One-command workflow")
    print("  - run_experiments.py - Full experiment runner")
    print("  - experiments/enhanced_runner.py - Advanced features")
    print("=" * 70)


if __name__ == "__main__":
    main()
