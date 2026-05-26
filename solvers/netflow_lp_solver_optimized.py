#!/usr/bin/env python3
                                        
                                                                
 
                                                               
 
                    
                                                      
                                           
                              
                                          
                                        
                                     
                                              

from __future__ import annotations

import os
import sys

                                                                      
os.environ['GRB_LICENSE_FILE'] = os.environ.get('GRB_LICENSE_FILE', '')
os.environ['GUROBI_LOGFILE'] = ''


class _SuppressGurobiOutput:
                                                                            
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


                                          
try:
    with _SuppressGurobiOutput():
        import gurobipy as _gp
        _env = _gp.Env(empty=True)
        _env.setParam("OutputFlag", 0)
        _env.setParam("LogToConsole", 0)
        _env.start()
        _GUROBI_ENV = _env
    HAS_GUROBI = True
except:
    HAS_GUROBI = False
    _GUROBI_ENV = None

from typing import Dict, List, Tuple, Any, Optional
import time

import numpy as np

                                      
try:
    from numba import njit, prange, types
    from numba.typed import Dict as NumbaDict
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper
    prange = range

              
Marking = Tuple[int, ...]
Edge = Tuple[Marking, int, Marking]


class NetflowBuildError(Exception):
    pass


                                                                               
                                                          
                                                                               

if HAS_NUMBA:
    @njit(cache=True)
    def _compute_all_successors_numba(
        marking: np.ndarray,
        I: np.ndarray,
        W_minus: np.ndarray,
        n_transitions: int
    ) -> Tuple[np.ndarray, np.ndarray]:
\
\
\
\
\
\
           
        P = marking.shape[0]
        
                                               
        n_enabled = 0
        for t in range(n_transitions):
            enabled = True
            for p in range(P):
                if marking[p] < W_minus[p, t]:
                    enabled = False
                    break
            if enabled:
                n_enabled += 1
        
                                
        enabled_trans = np.empty(n_enabled, dtype=np.int32)
        successors = np.empty((n_enabled, P), dtype=np.int32)
        
                                         
        idx = 0
        for t in range(n_transitions):
            enabled = True
            for p in range(P):
                if marking[p] < W_minus[p, t]:
                    enabled = False
                    break
            if enabled:
                enabled_trans[idx] = t
                for p in range(P):
                    successors[idx, p] = marking[p] + I[p, t]
                idx += 1
        
        return enabled_trans, successors


    @njit(cache=True)
    def _batch_compute_successors_numba(
        markings: np.ndarray,
        I: np.ndarray,
        W_minus: np.ndarray,
        n_transitions: int,
        max_successors_per_marking: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
           
        n_markings = markings.shape[0]
        P = markings.shape[1]
        
                                                          
        max_total = n_markings * max_successors_per_marking
        
        all_from = np.empty(max_total, dtype=np.int32)
        all_trans = np.empty(max_total, dtype=np.int32)
        all_succ = np.empty((max_total, P), dtype=np.int32)
        counts = np.empty(n_markings, dtype=np.int32)
        
        total_edges = 0
        
        for m_idx in range(n_markings):
            count = 0
            for t in range(n_transitions):
                                                
                enabled = True
                for p in range(P):
                    if markings[m_idx, p] < W_minus[p, t]:
                        enabled = False
                        break
                
                if enabled:
                                         
                    if total_edges >= max_total:
                                                                                
                        break
                    
                    all_from[total_edges] = m_idx
                    all_trans[total_edges] = t
                    
                                       
                    for p in range(P):
                        all_succ[total_edges, p] = markings[m_idx, p] + I[p, t]
                    
                    total_edges += 1
                    count += 1
            
            counts[m_idx] = count
        
        return all_from[:total_edges], all_trans[:total_edges], all_succ[:total_edges], counts


    @njit(cache=True)
    def _check_negative_markings_numba(successors: np.ndarray) -> np.ndarray:
                                                                    
        n = successors.shape[0]
        valid = np.ones(n, dtype=np.bool_)
        
        for i in range(n):
            for p in range(successors.shape[1]):
                if successors[i, p] < 0:
                    valid[i] = False
                    break
        
        return valid

else:
                                                        
    
    def _compute_all_successors_numba(
        marking: np.ndarray,
        I: np.ndarray,
        W_minus: np.ndarray,
        n_transitions: int
    ) -> Tuple[np.ndarray, np.ndarray]:
\
\
           
                                                                                
                                                                             
        enabled_mask = np.all(marking[:, np.newaxis] >= W_minus, axis=0)
        enabled_trans = np.where(enabled_mask)[0].astype(np.int32)
        
        if len(enabled_trans) == 0:
            return np.array([], dtype=np.int32), np.empty((0, len(marking)), dtype=np.int32)
        
                                                           
                                                                       
        successors = marking[:, np.newaxis] + I[:, enabled_trans]                  
        successors = successors.T.astype(np.int32)                  
        
        return enabled_trans, successors


    def _batch_compute_successors_numba(
        markings: np.ndarray,
        I: np.ndarray,
        W_minus: np.ndarray,
        n_transitions: int,
        max_successors_per_marking: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
\
\
           
        n_markings = markings.shape[0]
        P = markings.shape[1]
        T = n_transitions
        
                                                                        
                                                    
                                                                  
                                                
                                                                          
        enabled_matrix = np.all(
            markings[:, :, np.newaxis] >= W_minus[np.newaxis, :, :],
            axis=1
        )                   
        
                                                                    
        m_indices, t_indices = np.where(enabled_matrix)
        
        if len(m_indices) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
                np.empty((0, P), dtype=np.int32),
                np.zeros(n_markings, dtype=np.int32)
            )
        
                                                                     
        successors = markings[m_indices] + I[:, t_indices].T                
        
                                      
        counts = np.bincount(m_indices, minlength=n_markings).astype(np.int32)
        
        return (
            m_indices.astype(np.int32),
            t_indices.astype(np.int32),
            successors.astype(np.int32),
            counts
        )


    def _check_negative_markings_numba(successors: np.ndarray) -> np.ndarray:
                                        
        return np.all(successors >= 0, axis=1)


                                                                               
                                                
                                                                               

class FastMarkingCache:
\
\
\
\
       
    __slots__ = ('_hash_to_idx', '_markings', '_capacity', '_count', '_P')
    
    def __init__(self, P: int, initial_capacity: int = 100000):
\
\
\
\
\
\
           
        self._P = P
        self._capacity = initial_capacity
        self._count = 0
        self._hash_to_idx: Dict[bytes, int] = {}
                                                                    
        self._markings = np.empty((initial_capacity, P), dtype=np.int32)
    
    def add(self, marking: np.ndarray) -> Tuple[int, bool]:
\
\
\
\
\
           
        key = marking.tobytes()
        idx = self._hash_to_idx.get(key)
        
        if idx is not None:
            return idx, False
        
                                 
        if self._count >= self._capacity:
            self._grow()
        
        new_idx = self._count
        self._markings[new_idx] = marking
        self._hash_to_idx[key] = new_idx
        self._count += 1
        
        return new_idx, True
    
    def add_batch(self, markings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
\
\
\
\
\
\
           
        n = markings.shape[0]
        indices = np.empty(n, dtype=np.int32)
        is_new = np.zeros(n, dtype=np.bool_)
        
        for i in range(n):
            idx, new = self.add(markings[i])
            indices[i] = idx
            is_new[i] = new
        
        return indices, is_new
    
    def _grow(self):
                                        
        new_capacity = self._capacity * 2
        new_markings = np.empty((new_capacity, self._P), dtype=np.int32)
        new_markings[:self._count] = self._markings[:self._count]
        self._markings = new_markings
        self._capacity = new_capacity
    
    def get_marking(self, idx: int) -> np.ndarray:
                                                    
        return self._markings[idx]
    
    def get_all_markings(self) -> np.ndarray:
                                           
        return self._markings[:self._count]
    
    def __len__(self) -> int:
        return self._count
    
    def contains_bytes(self, key: bytes) -> bool:
        return key in self._hash_to_idx
    
    def get_idx_by_bytes(self, key: bytes) -> Optional[int]:
        return self._hash_to_idx.get(key)


                                                                               
                                            
                                                                               

class FastEdgeStorage:
\
\
\
\
       
    __slots__ = ('_from', '_trans', '_to', '_capacity', '_count')
    
    def __init__(self, initial_capacity: int = 500000):
        self._capacity = initial_capacity
        self._count = 0
        self._from = np.empty(initial_capacity, dtype=np.int32)
        self._trans = np.empty(initial_capacity, dtype=np.int32)
        self._to = np.empty(initial_capacity, dtype=np.int32)
    
    def add(self, from_idx: int, trans: int, to_idx: int):
                              
        if self._count >= self._capacity:
            self._grow()
        
        self._from[self._count] = from_idx
        self._trans[self._count] = trans
        self._to[self._count] = to_idx
        self._count += 1
    
    def add_batch(self, from_indices: np.ndarray, transitions: np.ndarray, to_indices: np.ndarray):
                                             
        n = len(from_indices)
        while self._count + n > self._capacity:
            self._grow()
        
        self._from[self._count:self._count + n] = from_indices
        self._trans[self._count:self._count + n] = transitions
        self._to[self._count:self._count + n] = to_indices
        self._count += n
    
    def _grow(self):
                              
        new_capacity = self._capacity * 2
        new_from = np.empty(new_capacity, dtype=np.int32)
        new_trans = np.empty(new_capacity, dtype=np.int32)
        new_to = np.empty(new_capacity, dtype=np.int32)
        
        new_from[:self._count] = self._from[:self._count]
        new_trans[:self._count] = self._trans[:self._count]
        new_to[:self._count] = self._to[:self._count]
        
        self._from = new_from
        self._trans = new_trans
        self._to = new_to
        self._capacity = new_capacity
    
    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                                      
        return (
            self._from[:self._count],
            self._trans[:self._count],
            self._to[:self._count]
        )
    
    def __len__(self) -> int:
        return self._count


                                                                               
                                            
                                                                               

def _build_sync_rg_ultra_optimized(
    I: np.ndarray,
    m_i: np.ndarray,
    m_f: np.ndarray,
    max_depth: int,
    rg_timeout: Optional[float] = None,
    cost_vector: Optional[np.ndarray] = None,
) -> Tuple[FastMarkingCache, FastEdgeStorage, float, Dict[str, Any]]:
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
       
    if max_depth <= 0:
        raise NetflowBuildError("max_depth must be positive.")
    
    t0 = time.time()
    
    P, T = I.shape
    W_minus = np.maximum(-I, 0).astype(np.int32)
    I_int = I.astype(np.int32)
    
                                                                               
                                              
                                                                               
                                                             
      
                                                  
                                                                         
                                                                 
                                                              
                                                         
     
                                                                        
                                                                
                                                                               
    
    if cost_vector is not None:
                                                        
                                                    
        is_tau_transition = (cost_vector > 0) & (cost_vector < 0.01)
        n_tau = np.sum(is_tau_transition)
    else:
                                                   
        is_tau_transition = np.zeros(T, dtype=np.bool_)
        n_tau = 0
    
                    
    stats = {
        "markings_explored": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "status": "complete",
        "layers_processed": 0,
        "tau_edges_pruned": 0,
        "self_loops_pruned": 0,
        "tau_transitions_detected": int(n_tau),
        "total_transitions": T,
    }
    
                                                          
    estimated_nodes = min(max_depth * T * 10, 5000000)
    estimated_edges = estimated_nodes * 3
    
                                
    cache = FastMarkingCache(P, initial_capacity=estimated_nodes)
    edges = FastEdgeStorage(initial_capacity=estimated_edges)
    
           
    stats = {
        "markings_explored": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "status": "complete",
        "layers_processed": 0,
        "tau_edges_pruned": 0,
        "self_loops_pruned": 0,
        "tau_transitions_detected": int(n_tau),
        "total_transitions": T,
    }
    
                         
    m_i_int = m_i.astype(np.int32, copy=False)
    start_idx, _ = cache.add(m_i_int)
    
                     
    current_layer = np.array([start_idx], dtype=np.int32)
    current_depth = 0
    
                                                        
    avg_successors = min(T, 20)
    
    while len(current_layer) > 0 and current_depth < max_depth:
                                           
        if rg_timeout is not None:
            elapsed = time.time() - t0
            if elapsed > rg_timeout:
                stats["status"] = "timeout"
                stats["layers_processed"] = current_depth
                build_time = elapsed
                return cache, edges, build_time, stats
        
        layer_size = len(current_layer)
        stats["markings_explored"] += layer_size
        stats["layers_processed"] = current_depth
        
                                                                      
        CHUNK_SIZE = 50000
        
        if layer_size > CHUNK_SIZE:
                                           
            all_from_list = []
            all_trans_list = []
            all_succ_list = []
            
            for chunk_start in range(0, layer_size, CHUNK_SIZE):
                if rg_timeout is not None:
                    elapsed = time.time() - t0
                    if elapsed > rg_timeout:
                        stats["status"] = "timeout"
                        stats["layers_processed"] = current_depth
                        build_time = elapsed
                        return cache, edges, build_time, stats
                
                chunk_end = min(chunk_start + CHUNK_SIZE, layer_size)
                chunk_indices = current_layer[chunk_start:chunk_end]
                
                chunk_markings = np.empty((len(chunk_indices), P), dtype=np.int32)
                for i, idx in enumerate(chunk_indices):
                    chunk_markings[i] = cache.get_marking(idx)
                
                c_from, c_trans, c_succ, _ = _batch_compute_successors_numba(
                    chunk_markings, I_int, W_minus, T, avg_successors * 2
                )
                
                if len(c_from) > 0:
                    c_from = c_from + chunk_start
                    all_from_list.append(c_from)
                    all_trans_list.append(c_trans)
                    all_succ_list.append(c_succ)
            
            if not all_from_list:
                break
            
            all_from = np.concatenate(all_from_list)
            all_trans = np.concatenate(all_trans_list)
            all_succ = np.vstack(all_succ_list)
        else:
            layer_markings = np.empty((layer_size, P), dtype=np.int32)
            for i, idx in enumerate(current_layer):
                layer_markings[i] = cache.get_marking(idx)
            
            all_from, all_trans, all_succ, _ = _batch_compute_successors_numba(
                layer_markings, I_int, W_minus, T, avg_successors * 2
            )
        
        if len(all_from) == 0:
            break
        
                                                
        valid_mask = _check_negative_markings_numba(all_succ)
        
        if not np.any(valid_mask):
            break
        
        all_from = all_from[valid_mask]
        all_trans = all_trans[valid_mask]
        all_succ = all_succ[valid_mask]
        
                                                                           
                               
                                                                           
                                                              
                                                                       
                                                                           
        global_from = current_layer[all_from]
        
                                            
        source_markings = np.empty_like(all_succ)
        for i, idx in enumerate(global_from):
            source_markings[i] = cache.get_marking(idx)
        
                                        
        not_self_loop = ~np.all(all_succ == source_markings, axis=1)
        n_self_loops = np.sum(~not_self_loop)
        stats["self_loops_pruned"] += n_self_loops
        
        if n_self_loops > 0:
            all_from = all_from[not_self_loop]
            all_trans = all_trans[not_self_loop]
            all_succ = all_succ[not_self_loop]
            global_from = global_from[not_self_loop]
        
        if len(all_from) == 0:
            break
        
                                                                           
                                                                          
                                                                           
        succ_indices, is_new = cache.add_batch(all_succ)
        
                                                                           
                                                                 
                                                                           
                                                                              
                                                              
                                                                 
                                            
         
                                                                    
                                                          
                                                                           
        if n_tau > 0:
                                                        
            is_tau_edge = is_tau_transition[all_trans]
            
                            
                                                
                                             
            keep_edge = ~is_tau_edge | is_new
            
            n_tau_pruned = np.sum(~keep_edge)
            stats["tau_edges_pruned"] += n_tau_pruned
            
            if n_tau_pruned > 0:
                               
                keep_indices = np.where(keep_edge)[0]
                all_from = all_from[keep_indices]
                all_trans = all_trans[keep_indices]
                global_from = global_from[keep_indices]
                succ_indices = succ_indices[keep_indices]
                is_new = is_new[keep_indices]
        
                     
        stats["cache_misses"] += np.sum(is_new)
        stats["cache_hits"] += np.sum(~is_new)
        
                   
        if len(global_from) > 0:
            edges.add_batch(global_from, all_trans, succ_indices)
        
                                        
        current_layer = succ_indices[is_new]
        current_depth += 1
        
        if layer_size > 0:
            avg_successors = max(1, len(all_from) // layer_size)
    
                                                   
    mf_int = m_f.astype(np.int32, copy=False)
    mf_key = mf_int.tobytes()
    
    build_time = time.time() - t0
    stats["num_nodes"] = len(cache)
    stats["num_edges"] = len(edges)
    stats["nodes_per_second"] = len(cache) / build_time if build_time > 0 else 0
    
    if not cache.contains_bytes(mf_key):
                                                                          
                                                                                        
        stats["status"] = "no_path"
        stats["final_marking_reached"] = False
    else:
        stats["final_marking_reached"] = True
    
    return cache, edges, build_time, stats


                                                                               
                         
                                                                               

def _solve_netflow_lp_gurobi_sparse(
    cache: FastMarkingCache,
    edges: FastEdgeStorage,
    m_i: np.ndarray,
    m_f: np.ndarray,
    c: np.ndarray,
    time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
\
\
\
\
\
\
\
       
    import gurobipy as gp
    from gurobipy import GRB
    
    t0 = time.time()
    
    from_arr, trans_arr, to_arr = edges.get_arrays()
    n_edges = len(from_arr)
    n_nodes = len(cache)
    
                             
    mi_key = m_i.astype(np.int32, copy=False).tobytes()
    mf_key = m_f.astype(np.int32, copy=False).tobytes()
    
    src_idx = cache.get_idx_by_bytes(mi_key)
    sink_idx = cache.get_idx_by_bytes(mf_key)
    
    if src_idx is None or sink_idx is None:
        raise RuntimeError("Source or sink marking not in reachability graph")
    
                  
    global _GUROBI_ENV
    if _GUROBI_ENV is not None:
        model = gp.Model("netflow_lp", env=_GUROBI_ENV)
    else:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        model = gp.Model("netflow_lp", env=env)
    
    model.setParam("OutputFlag", 0)
    
                        
    if n_edges < 1000:
        model.setParam("Threads", 1)
    elif n_edges < 10000:
        model.setParam("Threads", 2)
    else:
        model.setParam("Threads", 4)
    
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    
                       
    model.setParam("Method", 1)                
    model.setParam("OptimalityTol", 1e-6)
    model.setParam("FeasibilityTol", 1e-6)
    model.setParam("Presolve", 2)
    model.setParam("NetworkAlg", 1)
    
                                   
    obj_coeffs = c[trans_arr].astype(float)
    x = model.addMVar(n_edges, lb=0.0, obj=obj_coeffs, vtype=GRB.CONTINUOUS, name="x")
    
    model.setAttr("ModelSense", GRB.MINIMIZE)
    
                                                               
                                                           
                                                               
    
                                            
    from scipy import sparse
    
                                        
                                                                    
    row_out = from_arr
    row_in = to_arr
    col_out = np.arange(n_edges, dtype=np.int32)
    col_in = np.arange(n_edges, dtype=np.int32)
    
    data_out = np.ones(n_edges, dtype=float)
    data_in = -np.ones(n_edges, dtype=float)
    
    rows = np.concatenate([row_out, row_in])
    cols = np.concatenate([col_out, col_in])
    data = np.concatenate([data_out, data_in])
    
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_edges))
    
                
    rhs = np.zeros(n_nodes, dtype=float)
    rhs[src_idx] = 1.0
    rhs[sink_idx] = -1.0
    
                             
    model.addMConstr(A, x, '=', rhs, name="flow")
    
    model.optimize()
    solve_time = time.time() - t0
    
    if model.status == GRB.INFEASIBLE:
        raise RuntimeError(
            f"LP infeasible (sparse): no path from initial to final marking. "
            f"Nodes: {n_nodes}, Edges: {n_edges}. Check max_depth or model structure."
        )
    
    if model.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi LP did not converge (sparse). Status: {model.status}")
    
    x_val = x.X
    obj = model.objVal
    
    return x_val, obj, solve_time


def _solve_netflow_lp_gurobi_fallback(
    cache: FastMarkingCache,
    edges: FastEdgeStorage,
    m_i: np.ndarray,
    m_f: np.ndarray,
    c: np.ndarray,
    time_limit: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
\
\
       
    import gurobipy as gp
    from gurobipy import GRB
    
    t0 = time.time()
    
    from_arr, trans_arr, to_arr = edges.get_arrays()
    n_edges = len(from_arr)
    n_nodes = len(cache)
    
                             
    mi_key = m_i.astype(np.int32, copy=False).tobytes()
    mf_key = m_f.astype(np.int32, copy=False).tobytes()
    
    src_idx = cache.get_idx_by_bytes(mi_key)
    sink_idx = cache.get_idx_by_bytes(mf_key)
    
    if src_idx is None or sink_idx is None:
        raise RuntimeError("Source or sink marking not in reachability graph")
    
                  
    global _GUROBI_ENV
    if _GUROBI_ENV is not None:
        model = gp.Model("netflow_lp", env=_GUROBI_ENV)
    else:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        model = gp.Model("netflow_lp", env=env)
    
    model.setParam("OutputFlag", 0)
    model.setParam("Method", 1)
    model.setParam("OptimalityTol", 1e-6)
    model.setParam("FeasibilityTol", 1e-6)
    model.setParam("Presolve", 2)
    
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    
                   
    x = model.addVars(n_edges, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    
               
    obj = gp.LinExpr()
    for e in range(n_edges):
        obj += float(c[trans_arr[e]]) * x[e]
    model.setObjective(obj, GRB.MINIMIZE)
    
                       
    flow = {v: gp.LinExpr() for v in range(n_nodes)}
    
    for e in range(n_edges):
        flow[from_arr[e]] += x[e]
        flow[to_arr[e]] -= x[e]
    
    for v in range(n_nodes):
        if v == src_idx:
            model.addConstr(flow[v] == 1.0)
        elif v == sink_idx:
            model.addConstr(flow[v] == -1.0)
        else:
            model.addConstr(flow[v] == 0.0)
    
    model.optimize()
    solve_time = time.time() - t0
    
    if model.status == GRB.INFEASIBLE:
        raise RuntimeError(
            f"LP infeasible (fallback): no path from initial to final marking. "
            f"Nodes: {n_nodes}, Edges: {n_edges}. Check max_depth or model structure."
        )
    
    if model.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi LP did not converge (fallback). Status: {model.status}")
    
    x_val = np.array([x[e].X for e in range(n_edges)])
    obj_val = model.objVal
    
    return x_val, obj_val, solve_time


                                                                          
try:
    from scipy import sparse as _sparse
    _USE_SPARSE = True
except ImportError:
    _USE_SPARSE = False


                                                                               
          
                                                                               

def solve_alignment_netflow_lp_optimized(
    I: np.ndarray,
    c: np.ndarray,
    m_i: np.ndarray,
    m_f: np.ndarray,
    max_depth: int,
    time_limit: Optional[float] = None,
    rg_timeout: Optional[float] = None,
    enable_optimizations: bool = True,
    use_custom_hash: bool = None,                              
    model_cache: Any = None,                              
    trace_labels: Optional[List[str]] = None,                              
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
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
       
    I = np.asarray(I, dtype=np.int32)
    c = np.asarray(c, dtype=float)
    m_i = np.asarray(m_i, dtype=np.int32)
    m_f = np.asarray(m_f, dtype=np.int32)
    
                              
                                                                    
                                                                  
    cache, edges, build_time, rg_stats = _build_sync_rg_ultra_optimized(
        I, m_i, m_f, max_depth, 
        rg_timeout=rg_timeout,
        cost_vector=None,                                  
    )
    
                                                   
    if rg_stats.get("status") in ("timeout", "no_path"):
        info = {
            "build_time": build_time,
            "solve_time": 0.0,
            "total_time": build_time,
            "num_nodes": len(cache),
            "num_edges": len(edges),
            "status": "rg_" + rg_stats.get("status", "timeout"),
            "rg_stats": rg_stats,
            "nodes_per_second": rg_stats.get("nodes_per_second", 0),
        }
        return np.array([]), info
    
    if len(edges) == 0:
        raise RuntimeError(
            f"No enabled transitions reachable. Nodes: {len(cache)}, "
            f"max_depth: {max_depth}, I.shape: {I.shape}"
        )
    
              
    try:
        if _USE_SPARSE:
            x_val, obj, solve_time = _solve_netflow_lp_gurobi_sparse(
                cache, edges, m_i, m_f, c, time_limit=time_limit
            )
        else:
            x_val, obj, solve_time = _solve_netflow_lp_gurobi_fallback(
                cache, edges, m_i, m_f, c, time_limit=time_limit
            )
    except Exception as e:
                                      
        if _USE_SPARSE:
            x_val, obj, solve_time = _solve_netflow_lp_gurobi_fallback(
                cache, edges, m_i, m_f, c, time_limit=time_limit
            )
        else:
            raise
    
                                   
    T = I.shape[1]
    X = np.zeros((T, 1), dtype=float)
    
    _, trans_arr, _ = edges.get_arrays()
    for e_idx in range(len(x_val)):
        X[trans_arr[e_idx], 0] += x_val[e_idx]
    
    info = {
        "objective": obj,
        "num_nodes": len(cache),
        "num_edges": len(edges),
        "build_time": build_time,
        "solve_time": solve_time,
        "total_time": build_time + solve_time,
        "rg_stats": rg_stats,
        "optimizations_enabled": True,
        "n_states": len(cache),
        "n_edges": len(edges),
        "rg_build_time": build_time,
        "nodes_per_second": rg_stats.get("nodes_per_second", 0),
    }
    
    return X, info


def solve_alignment_netflow_lp(
    I: np.ndarray,
    c: np.ndarray,
    m_i: np.ndarray,
    m_f: np.ndarray,
    max_depth: int,
    time_limit: Optional[float] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
                                       
    return solve_alignment_netflow_lp_optimized(
        I, c, m_i, m_f,
        max_depth=max_depth,
        time_limit=time_limit,
        enable_optimizations=True,
        **kwargs,
    )


                                                                               
                                                        
                                                                               

def _mtup(m: np.ndarray) -> Marking:
                                                                       
    return tuple(m.tolist())


                                                                               
                                            
                                                                               

if HAS_NUMBA:
    try:
        _dummy_marking = np.array([1, 2, 3], dtype=np.int32)
        _dummy_I = np.array([[1, -1], [-1, 1], [0, 0]], dtype=np.int32)
        _dummy_W = np.array([[0, 1], [1, 0], [0, 0]], dtype=np.int32)
        
                                           
        _compute_all_successors_numba(_dummy_marking, _dummy_I, _dummy_W, 2)
        
                                  
        _dummy_batch = np.array([[1, 2, 3], [2, 1, 3]], dtype=np.int32)
        _batch_compute_successors_numba(_dummy_batch, _dummy_I, _dummy_W, 2, 4)
        
                               
        _check_negative_markings_numba(_dummy_batch)
    except Exception:
        pass


                                                                               
                     
                                                                               

def benchmark_rg_construction(
    I: np.ndarray,
    m_i: np.ndarray,
    m_f: np.ndarray,
    max_depth: int,
    rg_timeout: float = 60.0,
    cost_vector: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
\
\
\
\
       
    I = np.asarray(I, dtype=np.int32)
    m_i = np.asarray(m_i, dtype=np.int32)
    m_f = np.asarray(m_f, dtype=np.int32)
    
    cache, edges, build_time, stats = _build_sync_rg_ultra_optimized(
        I, m_i, m_f, max_depth, rg_timeout=rg_timeout, cost_vector=cost_vector
    )
    
    nodes = len(cache)
    edges_count = len(edges)
    
    return {
        "nodes": nodes,
        "edges": edges_count,
        "build_time_seconds": build_time,
        "nodes_per_second": nodes / build_time if build_time > 0 else 0,
        "edges_per_second": edges_count / build_time if build_time > 0 else 0,
        "status": stats.get("status", "complete"),
        "layers_processed": stats.get("layers_processed", 0),
        "cache_hit_rate": stats.get("cache_hits", 0) / max(1, stats.get("cache_hits", 0) + stats.get("cache_misses", 0)),
        "tau_edges_pruned": stats.get("tau_edges_pruned", 0),
        "self_loops_pruned": stats.get("self_loops_pruned", 0),
    }
