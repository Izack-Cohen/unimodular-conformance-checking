                            

from __future__ import annotations

try:
    from solvers.netflow_lp_solver_optimized import solve_alignment_netflow_lp_optimized

    HAS_OPTIMIZED_NETFLOW = True
except ImportError:
    HAS_OPTIMIZED_NETFLOW = False


def solve_alignment(I, c, m_i, m_f, n, method="gurobi", model="milp", **kwargs):
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
       
    if method not in ["gurobi", "pulp"]:
        raise ValueError(f"Unknown method: {method}. Must be 'gurobi' or 'pulp'")
    if model not in ["milp", "lp", "netflow_lp"]:
        raise ValueError(f"Unknown model: {model}. Must be 'milp', 'lp', or 'netflow_lp'")

    time_limit = kwargs.pop("time_limit", None)

    try:
        if model == "netflow_lp":
                                                                      
            if HAS_OPTIMIZED_NETFLOW:
                try:
                    return solve_alignment_netflow_lp_optimized(
                        I,
                        c,
                        m_i,
                        m_f,
                        max_depth=n,
                        time_limit=time_limit,
                        enable_optimizations=True,
                        **kwargs,
                    )
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Optimized netflow solver failed: {e}. Falling back to simple version."
                    )

                                                  
            from solvers.netflow_lp_solver_optimized import solve_alignment_netflow_lp

            return solve_alignment_netflow_lp(
                I,
                c,
                m_i,
                m_f,
                max_depth=n,
                time_limit=time_limit,
                **kwargs,
            )

        if method == "gurobi":
            if model == "milp":
                from solvers.gurobi_milp_solver import solve_milp_alignment

                return solve_milp_alignment(
                    I, c, m_i, m_f, n, time_limit=time_limit, **kwargs
                )
            elif model == "lp":
                from solvers.gurobi_lp_solver import solve_lp_alignment

                return solve_lp_alignment(
                    I, c, m_i, m_f, n, time_limit=time_limit, **kwargs
                )

        elif method == "pulp":
            if model == "milp":
                from solvers.pulp_milp_solver import solve_milp_alignment

                return solve_milp_alignment(I, c, m_i, m_f, n, **kwargs)
            elif model == "lp":
                from solvers.pulp_lp_solver import solve_lp_alignment

                return solve_lp_alignment(I, c, m_i, m_f, n, **kwargs)

    except ImportError as e:
        raise ImportError(f"Failed to import {method} solver: {str(e)}")
    except TypeError as e:
                                                                           
        if "unexpected keyword argument" in str(e):
            if method == "gurobi" and model == "milp":
                from solvers.gurobi_milp_solver import solve_milp_alignment

                return solve_milp_alignment(I, c, m_i, m_f, n)
            if method == "gurobi" and model == "lp":
                from solvers.gurobi_lp_solver import solve_lp_alignment

                return solve_lp_alignment(I, c, m_i, m_f, n)
        raise
    except Exception as e:
        raise RuntimeError(f"Solver error ({method} {model}): {str(e)}")
