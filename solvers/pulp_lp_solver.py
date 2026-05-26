                                 

import numpy as np
from models.alignment_model_builder import build_alignment_model_pulp


def solve_lp_alignment(I, c, m_i, m_f, n):
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
                                             
        x, prob = build_alignment_model_pulp(I, c, m_i, m_f, n, model_type="lp")

                                         
        status = prob.solve()

                               
        from pulp import LpStatusOptimal, LpStatusInfeasible, LpStatusUnbounded

        if status == LpStatusOptimal:
                              
            result = np.array([[x[j][k].varValue if x[j][k].varValue is not None else 0.0
                                for k in range(n)] for j in range(len(x))])
            return result, prob
        elif status == LpStatusInfeasible:
            raise RuntimeError("LP problem is infeasible")
        elif status == LpStatusUnbounded:
            raise RuntimeError("LP problem is unbounded")
        else:
            raise RuntimeError(f"LP optimization failed with status: {status}")

    except Exception as e:
        raise RuntimeError(f"PuLP LP solver error: {str(e)}")