                          
from __future__ import annotations
from typing import List, Tuple, Dict, Any


def format_alignment(alignment: List[Tuple[str, str]], show_indices: bool = True) -> str:
\
\
       
    if not alignment:
        return "Empty alignment"
    lines = []
    for i, (log_move, model_move) in enumerate(alignment, 1):
        prefix = f"  Step {i:2d}: " if show_indices else "  "
        if log_move == ">>" and model_move != ">>":
            lines.append(f"{prefix}Model-only: {model_move}")
        elif log_move != ">>" and model_move == ">>":
            lines.append(f"{prefix}Log-only: {log_move}")
        else:
            lines.append(f"{prefix}Sync: {log_move} ↔ {model_move}")
    return "\n".join(lines)


def run_a_star_alignment(trace_labels, net, im, fm) -> Dict[str, Any]:
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
       
    from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
    from pm4py.objects.log.obj import EventLog, Trace, Event

                                                
    trace = Trace([Event({"concept:name": a}) for a in trace_labels])
    log = EventLog([trace])

                                      
    res = align_algo.apply(
        log, net, im, fm,
        variant=align_algo.Variants.VERSION_STATE_EQUATION_A_STAR
    )

                                                                   
    result = res[0] if isinstance(res, list) else res
    if not isinstance(result, dict) or "alignment" not in result:
        raise RuntimeError(f"Unexpected A* result: {type(res)}")

                                
    raw_cost = result.get("cost")
    if raw_cost is not None:
                                                                  
                                                       
        normalized_cost = float(raw_cost) / 10000.0
    else:
        normalized_cost = None

    return {
        "alignment": result["alignment"],
        "cost": normalized_cost,                   
        "raw_cost": raw_cost,                               
        "fitness": result.get("fitness")
    }