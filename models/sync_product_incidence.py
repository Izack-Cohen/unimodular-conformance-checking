                                  
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking

TAU_LABELS = {"tau", "τ", None}


def _marking_vector(places: List[PetriNet.Place], marking: Marking) -> np.ndarray:
                                                       
    idx = {p: i for i, p in enumerate(places)}
    m = np.zeros(len(places), dtype=np.int32)
    for p, tokens in marking.items():
        if p in idx:
            m[idx[p]] = int(tokens)
    return m


def build_sync_product_incidence(
        net: PetriNet,
        im: Marking,
        fm: Marking,
        trace_labels: List[str],
        *,
        cost_sync: float = 0.0,
        cost_log: float = 1.0,
        cost_model: float = 1.0,
        cost_tau: float = 0.0001,
        model_cache=None,                                        
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
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
       
                                                               
    if model_cache is not None:
                                                      
        I_model = model_cache.I_model
        Pm = model_cache.Pm
        Tm = model_cache.Tm
        P_list = model_cache.P_list

                                                                          
        n = len(trace_labels)
        trace_places = n + 1

                                                               
        def trace_step_col(k: int) -> np.ndarray:
                                                                       
            col = np.zeros(trace_places, dtype=np.int32)
            col[k] += 1                         
            col[k - 1] -= 1                               
            return col

                                                      
        tr_cols = [trace_step_col(k) for k in range(1, n + 1)]

                                                       
        cols: List[np.ndarray] = []
        costs: List[float] = []
        meta: List[Dict] = []

                                                         
        for j in range(Tm):
                                               
            is_tau = j in model_cache.tau_indices

                                                              
            col_model_only = np.concatenate([
                I_model[:, j],                   
                np.zeros(trace_places, dtype=np.int32)                             
            ])
            cols.append(col_model_only)
            costs.append(cost_tau if is_tau else cost_model)

                                    
            t = model_cache.T_list[j]
            lab = getattr(t, "label", None)
            meta.append({
                "kind": "model_tau" if is_tau else "model",
                "model_t_index": j,
                "k": None,
                "label": lab
            })

                                                                          
            if not is_tau and lab is not None:
                for k_idx, trace_label in enumerate(trace_labels, start=1):
                    if trace_label == lab:
                                                                       
                        col_sync = np.concatenate([
                            I_model[:, j],                   
                            tr_cols[k_idx - 1]                    
                        ])
                        cols.append(col_sync)
                        costs.append(cost_sync)
                        meta.append({
                            "kind": "sync",
                            "model_t_index": j,
                            "k": k_idx,
                            "label": lab
                        })

                                                                
        zero_model = np.zeros(Pm, dtype=np.int32)
        for k_idx in range(1, n + 1):
                                                            
            col_log = np.concatenate([
                zero_model,                             
                tr_cols[k_idx - 1]                    
            ])
            cols.append(col_log)
            costs.append(cost_log)
            meta.append({
                "kind": "log",
                "model_t_index": None,
                "k": k_idx,
                "label": trace_labels[k_idx - 1]
            })

                                       
        m0_model = model_cache.marking_to_vector(im)
        mf_model = model_cache.marking_to_vector(fm)

    else:
                                                                                    
        from models.petri_utils import construct_incidence_matrix

                                   
        I_model, T_list, P_list = construct_incidence_matrix(net)
        Pm, Tm = I_model.shape                                                                 
        n = len(trace_labels)                   

                                                                    
        trace_places = n + 1

                                                               
        def trace_step_col(k: int) -> np.ndarray:
                                                                       
            col = np.zeros(trace_places, dtype=np.int32)
            col[k] += 1                         
            col[k - 1] -= 1                               
            return col

                                                       
        cols: List[np.ndarray] = []
        costs: List[float] = []
        meta: List[Dict] = []

                                                      
        tr_cols = [trace_step_col(k) for k in range(1, n + 1)]

                                                         
        for j, t in enumerate(T_list):
            lab = getattr(t, "label", None)
            is_tau = (lab in TAU_LABELS)

                                                              
            col_model_only = np.concatenate([
                I_model[:, j],                   
                np.zeros(trace_places, dtype=np.int32)                             
            ])
            cols.append(col_model_only)
            costs.append(cost_tau if is_tau else cost_model)
            meta.append({
                "kind": "model_tau" if is_tau else "model",
                "model_t_index": j,
                "k": None,
                "label": lab
            })

                                                                          
            if (not is_tau) and lab is not None:
                for k_idx, trace_label in enumerate(trace_labels, start=1):
                    if trace_label == lab:
                                                                       
                        col_sync = np.concatenate([
                            I_model[:, j],                   
                            tr_cols[k_idx - 1]                    
                        ])
                        cols.append(col_sync)
                        costs.append(cost_sync)
                        meta.append({
                            "kind": "sync",
                            "model_t_index": j,
                            "k": k_idx,
                            "label": lab
                        })

                                                                
        zero_model = np.zeros(Pm, dtype=np.int32)
        for k_idx in range(1, n + 1):
                                                            
            col_log = np.concatenate([
                zero_model,                             
                tr_cols[k_idx - 1]                    
            ])
            cols.append(col_log)
            costs.append(cost_log)
            meta.append({
                "kind": "log",
                "model_t_index": None,
                "k": k_idx,
                "label": trace_labels[k_idx - 1]
            })

                                       
        m0_model = _marking_vector(P_list, im)

                                     
        mf_model = _marking_vector(P_list, fm)

                                                                   
    I = np.stack(cols, axis=1)
    c = np.asarray(costs, dtype=np.float64)

                                                                   
    m0_trace = np.zeros(trace_places, dtype=np.int32)
    m0_trace[0] = 1                                              
    m0 = np.concatenate([m0_model, m0_trace])

                                                                 
    mf_trace = np.zeros(trace_places, dtype=np.int32)
    mf_trace[-1] = 1                                           
    mf = np.concatenate([mf_model, mf_trace])

    return I, c, m0, mf, meta