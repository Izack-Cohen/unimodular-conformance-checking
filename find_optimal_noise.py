#!/usr/bin/env python3
\
\
\
\
\
\
   

import sys
from pathlib import Path
import pm4py
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

         
FITNESS_TARGET = 0.90
PRECISION_TARGET = 0.60

                                                 
NOISE_VALUES = [
    0.055, 0.060, 0.062, 0.064, 0.066, 0.068, 0.070, 0.072, 
    0.074, 0.076, 0.078, 0.080, 0.082, 0.084, 0.085
]


def evaluate_noise_threshold(log, noise):
                                                                     
    try:
                        
        net, im, fm = inductive_miner.apply(
            log, 
            parameters={"noise_threshold": noise}
        )
        
                          
        fitness_result = fitness_evaluator.apply(
            log, net, im, fm, 
            variant=fitness_evaluator.Variants.TOKEN_BASED
        )
        fitness = fitness_result['averageFitness']
        
                            
        precision = precision_evaluator.apply(
            log, net, im, fm,
            variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE
        )
        
                     
        n_places = len(net.places)
        n_transitions = len(net.transitions)
        
        return {
            'noise': noise,
            'fitness': fitness,
            'precision': precision,
            'places': n_places,
            'transitions': n_transitions,
            'meets_fitness': fitness >= FITNESS_TARGET,
            'meets_precision': precision >= PRECISION_TARGET,
            'meets_both': (fitness >= FITNESS_TARGET and precision >= PRECISION_TARGET)
        }
    except Exception as e:
        print(f"  ⚠ Error at noise={noise}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_optimal_noise.py <path_to_xes_file>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    
    if not Path(log_path).exists():
        print(f"Error: File not found: {log_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🔍 OPTIMAL NOISE THRESHOLD FINDER")
    print("=" * 80)
    print(f"\nDataset: {log_path}")
    print(f"Targets: fitness ≥ {FITNESS_TARGET:.2f}, precision ≥ {PRECISION_TARGET:.2f}")
    print(f"\nTesting {len(NOISE_VALUES)} noise thresholds in critical range...")
    print("=" * 80)
    
              
    print("\n📂 Loading event log...")
    log = pm4py.read_xes(log_path)
    print(f"   ✓ Loaded {len(log)} traces")
    
                               
    results = []
    best_candidate = None
    best_score = -float('inf')
    
    print("\n🧪 Testing noise thresholds:\n")
    print(f"{'Noise':<8} {'Fitness':<9} {'Precision':<11} {'Places':<8} {'Trans':<7} {'Status':<10}")
    print("-" * 80)
    
    for noise in NOISE_VALUES:
        result = evaluate_noise_threshold(log, noise)
        
        if result is None:
            continue
        
        results.append(result)
        
                      
        status = ""
        if result['meets_both']:
            status = "✓ BOTH"
        elif result['meets_fitness']:
            status = "✓ Fitness"
        elif result['meets_precision']:
            status = "✓ Precision"
        else:
            status = "✗"
        
        print(f"{result['noise']:<8.4f} "
              f"{result['fitness']:<9.4f} "
              f"{result['precision']:<11.4f} "
              f"{result['places']:<8} "
              f"{result['transitions']:<7} "
              f"{status:<10}")
        
                                                                
        if result['meets_both']:
            score = result['fitness'] + result['precision']              
            if score > best_score:
                best_score = score
                best_candidate = result
        elif best_candidate is None:
                                                    
            fitness_diff = max(0, FITNESS_TARGET - result['fitness'])
            precision_diff = max(0, PRECISION_TARGET - result['precision'])
            score = -(fitness_diff**2 + precision_diff**2)
            if score > best_score:
                best_score = score
                best_candidate = result
    
             
    print("\n" + "=" * 80)
    print("📊 RESULTS SUMMARY")
    print("=" * 80)
    
    models_meeting_both = [r for r in results if r['meets_both']]
    models_meeting_fitness = [r for r in results if r['meets_fitness']]
    models_meeting_precision = [r for r in results if r['meets_precision']]
    
    print(f"\nModels meeting both targets: {len(models_meeting_both)}/{len(results)}")
    print(f"Models meeting fitness ≥{FITNESS_TARGET}: {len(models_meeting_fitness)}/{len(results)}")
    print(f"Models meeting precision ≥{PRECISION_TARGET}: {len(models_meeting_precision)}/{len(results)}")
    
    if best_candidate:
        print("\n" + "=" * 80)
        print("🎯 BEST MODEL FOUND")
        print("=" * 80)
        print(f"\nNoise threshold: {best_candidate['noise']:.4f}")
        print(f"Fitness:         {best_candidate['fitness']:.4f} ", end="")
        print("✓" if best_candidate['meets_fitness'] else f"(target: {FITNESS_TARGET:.2f})")
        print(f"Precision:       {best_candidate['precision']:.4f} ", end="")
        print("✓" if best_candidate['meets_precision'] else f"(target: {PRECISION_TARGET:.2f})")
        print(f"Places:          {best_candidate['places']}")
        print(f"Transitions:     {best_candidate['transitions']}")
        
        if best_candidate['meets_both']:
            print("\n✅ SUCCESS! This model meets both targets!")
            print("\nTo use this model in your experiments:")
            print(f"   1. Edit run_experiments.py")
            print(f"   2. Set: noise_threshold = {best_candidate['noise']:.4f}")
            print(f"   3. Re-run experiments")
        else:
            print("\n⚠️  No model met both targets perfectly.")
            print("   This is the closest model found.")
            print("\nConsider:")
            print("   1. Testing more noise values around this threshold")
            print("   2. Trying a different discovery algorithm")
            print("   3. Relaxing the targets slightly")
    else:
        print("\n❌ No valid models found. Check for errors above.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
