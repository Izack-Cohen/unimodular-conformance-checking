#!/usr/bin/env python3
\
\
\
   
import sys
import os
import time
import json

                              
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.enhanced_runner import run_dataset_experiment

if __name__ == "__main__":
    print("=" * 70)
    print("BPI 2012 COMPLETE DATASET - A* vs LP Comparison")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Dataset: BPI Challenge 2012 (ALL 13,087 traces)")
    print("  - Methods: A* and LP only (MILP skipped)")
    print("  - Solver: Gurobi")
    print("  - Time limit: 30 seconds per trace")
    print("  - Model: Fitness >= 0.9, Precision ~ 0.7")
    print()
    print("⚠️  WARNING: This will take ~60 minutes!")
    print("  - Estimated time: 60 minutes (with vectorized optimizations)")
    print("  - Caching enabled: Subsequent runs much faster")
    print()
    print("Performance expectations (from 100-trace test):")
    print("  - A*: 0.22s per trace average")
    print("  - LP: 0.27s per trace average")
    print("  - Total time: ~60 minutes")
    print("=" * 70)

                          
    response = input("\nRun on FULL dataset (13,087 traces)? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        sys.exit(0)

    print("\n" + "=" * 70)
    print("Starting full dataset experiment...")
    print("=" * 70)
    start_time = time.time()

                              
    os.makedirs("results", exist_ok=True)

                        
    try:
        result = run_dataset_experiment(
            dataset_path="datasets/BPI_Challenge_2012.xes",
            time_limit=30.0,
            sample_size=None,                     
            min_fitness=0.9,
            target_precision=0.7,
            milp_solver="none",             
            lp_solver="gurobi",
            use_trace_cache=True
                                                                     
        )

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)

        print("\n" + "=" * 70)
        print("FULL DATASET EXPERIMENT COMPLETED!")
        print("=" * 70)
        print(f"\nTotal runtime: {hours}h {minutes}m {seconds}s")

                           
        print("\nKey Results:")
        print(f"  Total traces: {result['n_traces']}")
        print(f"  Model fitness: {result['fitness']:.3f}")
        print(f"  Model precision: {result['precision']:.3f}")
        print(f"\n  A* Performance:")
        print(f"    - Optimal: {result['methods']['A*']['pct_optimal']:.1f}%")
        print(f"    - Avg time: {result['methods']['A*']['avg_solve_time']:.3f}s")
        print(f"\n  LP Performance:")
        print(f"    - Optimal: {result['methods']['LP']['pct_optimal']:.1f}%")
        print(f"    - Avg time: {result['methods']['LP']['avg_solve_time']:.3f}s")
        print(f"    - RG build: {result['methods']['LP']['avg_rg_build_time']:.3f}s")
        print(f"    - LP solve: {result['methods']['LP']['avg_lp_solve_time']:.3f}s")

        print("\n  Cache Performance:")
        print(f"    - Hit rate: {result['cache_stats']['hit_rate']:.1f}%")
        print(f"    - Hits: {result['cache_stats']['cache_hits']}")
        print(f"    - Misses: {result['cache_stats']['cache_misses']}")

                                                                
        output_file = "results/bpi2012_full_optimized.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print("\nResults saved to:")
        print(f"  - {output_file}")

                                       
        summary_file = "results/bpi2012_full_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"BPI 2012 Full Dataset Results\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(f"Runtime: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Traces: {result['n_traces']}\n")
            f.write(f"A* avg: {result['methods']['A*']['avg_solve_time']:.3f}s\n")
            f.write(f"LP avg: {result['methods']['LP']['avg_solve_time']:.3f}s\n")
            f.write(f"LP optimal: {result['methods']['LP']['pct_optimal']:.1f}%\n")

        print(f"  - {summary_file}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        elapsed = time.time() - start_time
        print(f"Ran for {elapsed / 60:.1f} minutes before interruption.")
        print("Partial results saved in cache - can resume later!")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()

        elapsed = time.time() - start_time
        print(f"\nFailed after {elapsed / 60:.1f} minutes")
        sys.exit(1)