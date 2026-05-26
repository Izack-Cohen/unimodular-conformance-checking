                                     

from __future__ import annotations
import numpy as np

from models.alignment_model_builder import build_alignment_model_gurobi


def _get_var(x, j, k):
\
\
\
       
    try:
        row = x[j]
    except Exception:
        return None
    try:
        return row[k]
    except Exception:
        return None


def solve_milp_alignment(I, c, m_i, m_f, n, time_limit=None, mip_gap=None, threads=None, verbose=True, **kwargs):
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
       
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(
            "Gurobi is not installed or licensed. Please install `gurobipy` and ensure you have a valid license."
        ) from e

                                               
    try:
        x, model = build_alignment_model_gurobi(I, c, m_i, m_f, n, model_type="milp")
    except Exception as e:
        raise RuntimeError(f"Failed to build MILP model: {e}")

                             
    try:
                 
        try:
            model.setParam(GRB.Param.LogToConsole, 1 if verbose else 0)
        except Exception:
                                                      
            pass

                    
        if time_limit is not None:
            model.setParam(GRB.Param.TimeLimit, float(time_limit))

                                                   
        if mip_gap is not None:
            model.setParam(GRB.Param.MIPGap, float(mip_gap))
        else:
                                                                             
            model.setParam(GRB.Param.MIPGap, 0.0)                     
            model.setParam(GRB.Param.MIPGapAbs, 0.0001)                         

                 
        if threads is not None:
            model.setParam(GRB.Param.Threads, int(threads))

                                                                          
        model.setParam(GRB.Param.Presolve, 2)                       
        model.setParam(GRB.Param.MIPFocus, 1)                                                    

    except Exception as e:
                                                                                 
        raise RuntimeError(f"Failed to set Gurobi parameters: {e}")

              
    model.optimize()

    status = model.Status
                                                                                       
    has_solution = getattr(model, "SolCount", 0) and model.SolCount > 0

    if status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT):
        if not has_solution:
                                
            if status == GRB.TIME_LIMIT:
                raise RuntimeError("MILP reached time limit without finding any feasible solution.")
            if status == GRB.SUBOPTIMAL:
                raise RuntimeError("MILP finished SUBOPTIMAL without a stored solution.")
            raise RuntimeError(f"Gurobi finished with status={status} and no feasible solution.")
    elif status == GRB.INFEASIBLE:
        raise RuntimeError("MILP problem is infeasible.")
    elif status == GRB.UNBOUNDED:
        raise RuntimeError("MILP problem is unbounded.")
    else:
                                                                                   
        if not has_solution:
            raise RuntimeError(f"Gurobi MILP optimization failed with status: {status}")

                                     
    T = int(len(c))
    result = np.zeros((T, n), dtype=float)

                                       
    for j in range(T):
        for k in range(n):
            var = _get_var(x, j, k)
            if var is not None:
                try:
                    result[j, k] = var.X
                except Exception:
                                                                                      
                    pass

    return result, model