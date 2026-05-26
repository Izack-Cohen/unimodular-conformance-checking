
                         
from __future__ import annotations
from typing import Dict, Any
import pm4py
import statistics as stats
from pm4py.objects.conversion.log import converter as log_converter

def load_xes(path: str):
    log = pm4py.read_xes(path)
    try:
                                                                   
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)
    except Exception:
        pass
    return log

def log_stats(log) -> Dict[str, Any]:
    ntraces = len(log)
    lengths = []
    for t in log:
        try:
            lengths.append(len(t))
        except Exception:
                          
            lengths.append(0)
    avg_len = float(stats.mean(lengths)) if lengths else 0.0
    return {"num_traces": ntraces, "avg_trace_length": avg_len}
