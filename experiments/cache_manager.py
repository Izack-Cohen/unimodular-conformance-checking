#!/usr/bin/env python3
\
\
\
\
\
\
\
\
   
from __future__ import annotations
import os
import pickle
import hashlib
from pathlib import Path
from typing import Tuple, Optional, Any, Dict


def get_cache_dir() -> Path:
                                                         
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "models").mkdir(exist_ok=True)
    (cache_dir / "logs").mkdir(exist_ok=True)
    (cache_dir / "traces").mkdir(exist_ok=True)
    (cache_dir / "model_rg").mkdir(exist_ok=True)
    return cache_dir


def _get_cache_key(dataset_path: str, min_fitness: float = None, target_precision: float = None) -> str:
                                             
    dataset_name = Path(dataset_path).stem

    if min_fitness is None and target_precision is None:
        return dataset_name
    else:
        params = f"_f{min_fitness:.2f}_p{target_precision:.2f}"
        return dataset_name + params


def _hash_trace(trace_labels: list) -> str:
                                                                     
    trace_str = ",".join(trace_labels)
    return hashlib.sha256(trace_str.encode()).hexdigest()[:16]


def load_log_from_cache(dataset_path: str) -> Optional[Any]:
                                                 
    cache_dir = get_cache_dir()
    cache_key = _get_cache_key(dataset_path)
    cache_file = cache_dir / "logs" / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                log = pickle.load(f)
            print(f"  ✓ Loaded log from cache: {cache_file.name}")
            return log
        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}")
            return None
    return None


def save_log_to_cache(dataset_path: str, log: Any) -> None:
                                  
    cache_dir = get_cache_dir()
    cache_key = _get_cache_key(dataset_path)
    cache_file = cache_dir / "logs" / f"{cache_key}.pkl"

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Saved log to cache: {cache_file.name}")
    except Exception as e:
        print(f"  ⚠ Cache save failed: {e}")


def load_model_from_cache(
        dataset_path: str,
        min_fitness: float,
        target_precision: float
) -> Optional[Tuple[Any, Any, Any, float, float]]:
                                                        
    cache_dir = get_cache_dir()
    cache_key = _get_cache_key(dataset_path, min_fitness, target_precision)
    cache_file = cache_dir / "models" / f"{cache_key}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                model_data = pickle.load(f)

            net = model_data['net']
            im = model_data['im']
            fm = model_data['fm']
            fitness = model_data['fitness']
            precision = model_data['precision']

            print(f"  ✓ Loaded model from cache: {cache_file.name}")
            print(f"    Fitness: {fitness:.3f}, Precision: {precision:.3f}")
            print(f"    Places: {len(net.places)}, Transitions: {len(net.transitions)}")

            return net, im, fm, fitness, precision
        except Exception as e:
            print(f"  ⚠ Cache load failed: {e}")
            return None
    return None


def save_model_to_cache(
        dataset_path: str,
        min_fitness: float,
        target_precision: float,
        net: Any,
        im: Any,
        fm: Any,
        fitness: float,
        precision: float
) -> None:
                                         
    cache_dir = get_cache_dir()
    cache_key = _get_cache_key(dataset_path, min_fitness, target_precision)
    cache_file = cache_dir / "models" / f"{cache_key}.pkl"

    model_data = {
        'net': net,
        'im': im,
        'fm': fm,
        'fitness': fitness,
        'precision': precision,
        'dataset': Path(dataset_path).name,
        'min_fitness': min_fitness,
        'target_precision': target_precision,
    }

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Saved model to cache: {cache_file.name}")
    except Exception as e:
        print(f"  ⚠ Cache save failed: {e}")


def load_model_rg_cache(
        dataset_path: str,
        min_fitness: float,
        target_precision: float
) -> Optional[Any]:
\
\
\
\
\
       
    cache_dir = get_cache_dir()
    cache_key = _get_cache_key(dataset_path, min_fitness, target_precision)
    cache_file = cache_dir / "model_rg" / f"{cache_key}_rg.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                model_rg = pickle.load(f)
            print(f"  ✓ Loaded model RG cache: {cache_file.name}")
            print(f"    Cached markings: {len(model_rg.markings)}, transitions: {len(model_rg.model_transitions)}")
            return model_rg
        except Exception as e:
            print(f"  ⚠ Model RG cache load failed: {e}")
            return None
    return None


def save_model_rg_cache(
        dataset_path: str,
        min_fitness: float,
        target_precision: float,
        model_rg: Any
) -> None:
                                              
    cache_dir = get_cache_dir()
    cache_key = _get_cache_key(dataset_path, min_fitness, target_precision)
    cache_file = cache_dir / "model_rg" / f"{cache_key}_rg.pkl"

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(model_rg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Saved model RG cache: {cache_file.name}")
    except Exception as e:
        print(f"  ⚠ Model RG cache save failed: {e}")


def load_trace_results_from_cache(
        dataset_path: str,
        trace_labels: list,
        min_fitness: float,
        target_precision: float
) -> Optional[Dict[str, Any]]:
                                                             
    cache_dir = get_cache_dir()
    dataset_key = _get_cache_key(dataset_path, min_fitness, target_precision)
    trace_hash = _hash_trace(trace_labels)
    cache_file = cache_dir / "traces" / f"{dataset_key}_{trace_hash}.pkl"

    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                results = pickle.load(f)
            return results
        except Exception as e:
            return None
    return None


def save_trace_results_to_cache(
        dataset_path: str,
        trace_labels: list,
        min_fitness: float,
        target_precision: float,
        results: Dict[str, Any]
) -> None:
                                                      
    cache_dir = get_cache_dir()
    dataset_key = _get_cache_key(dataset_path, min_fitness, target_precision)
    trace_hash = _hash_trace(trace_labels)
    cache_file = cache_dir / "traces" / f"{dataset_key}_{trace_hash}.pkl"

    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        pass


def clear_cache(cache_type: str = "all") -> None:
                            
    cache_dir = get_cache_dir()

    if cache_type in ("all", "logs"):
        log_cache = cache_dir / "logs"
        for f in log_cache.glob("*.pkl"):
            f.unlink()
            print(f"Deleted: {f.name}")

    if cache_type in ("all", "models"):
        model_cache = cache_dir / "models"
        for f in model_cache.glob("*.pkl"):
            f.unlink()
            print(f"Deleted: {f.name}")

    if cache_type in ("all", "model_rg"):
        model_rg_cache = cache_dir / "model_rg"
        for f in model_rg_cache.glob("*.pkl"):
            f.unlink()
            print(f"Deleted: {f.name}")

    if cache_type in ("all", "traces"):
        trace_cache = cache_dir / "traces"
        for f in trace_cache.glob("*.pkl"):
            f.unlink()
            print(f"Deleted: {f.name}")


def list_cache() -> None:
                                
    cache_dir = get_cache_dir()

    print("Cached Logs:")
    log_cache = cache_dir / "logs"
    log_files = list(log_cache.glob("*.pkl"))
    if log_files:
        for f in log_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    else:
        print("  (none)")

    print("\nCached Models:")
    model_cache = cache_dir / "models"
    model_files = list(model_cache.glob("*.pkl"))
    if model_files:
        for f in model_files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("  (none)")

    print("\nCached Model RGs:")
    model_rg_cache = cache_dir / "model_rg"
    model_rg_files = list(model_rg_cache.glob("*.pkl"))
    if model_rg_files:
        for f in model_rg_files:
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    else:
        print("  (none)")

    print("\nCached Trace Results:")
    trace_cache = cache_dir / "traces"
    trace_files = list(trace_cache.glob("*.pkl"))
    if trace_files:
        total_size = sum(f.stat().st_size for f in trace_files)
        print(f"  - {len(trace_files)} cached traces ({total_size / (1024 * 1024):.1f} MB total)")
    else:
        print("  (none)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "list":
            list_cache()
        elif command == "clear":
            cache_type = sys.argv[2] if len(sys.argv) > 2 else "all"
            clear_cache(cache_type)
            print(f"Cleared {cache_type} cache")
        else:
            print("Usage: python cache_manager.py [list|clear [all|logs|models|model_rg|traces]]")
    else:
        list_cache()