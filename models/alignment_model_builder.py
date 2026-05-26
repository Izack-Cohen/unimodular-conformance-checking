                                         

def build_alignment_model_pulp(I, c, m_i, m_f, n, model_type="milp"):
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
       
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous

    num_places, num_transitions = I.shape

                     
    if len(c) != num_transitions:
        raise ValueError(f"Cost vector length ({len(c)}) must match number of transitions ({num_transitions})")
    if len(m_i) != num_places or len(m_f) != num_places:
        raise ValueError(f"Marking vectors must have length {num_places} (number of places)")
    if n <= 0:
        raise ValueError("Alignment length n must be positive")

                                                           
    var_cat = LpBinary if model_type == "milp" else LpContinuous

                                                                
    x = [[LpVariable(f"x_{j}_{k}", lowBound=0, upBound=1, cat=var_cat) for k in range(n)]
         for j in range(num_transitions)]

                                                                           
    z = [LpVariable(f"z_{k}", lowBound=0, upBound=1, cat=var_cat) for k in range(n)]

                    
    prob = LpProblem(f"{'MILP' if model_type == 'milp' else 'LP'}_Alignment", LpMinimize)

                                    
    prob += lpSum(c[j] * x[j][k] for j in range(num_transitions) for k in range(n))

                                                                    
    for i in range(num_places):
        prob += (m_i[i] + lpSum(I[i, j] * lpSum(x[j][k] for k in range(n))
                                for j in range(num_transitions)) == m_f[i],
                 f"final_marking_{i}")

                                                     
    for k in range(n):
        for i in range(num_places):
            prob += (m_i[i] + lpSum(I[i, j] * lpSum(x[j][tau] for tau in range(k + 1))
                                    for j in range(num_transitions)) >= 0,
                     f"feasible_{i}_{k}")

                                                                 
    for k in range(n):
        prob += (lpSum(x[j][k] for j in range(num_transitions)) + z[k] == 1,
                 f"move_or_stop_{k}")

                                                             
    for k in range(n - 1):
        prob += (z[k + 1] >= z[k], f"monotonic_{k}")

    return x, prob


def build_alignment_model_gurobi(I, c, m_i, m_f, n, model_type="milp"):
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
    except ImportError:
        raise ImportError(
            "Gurobi is not installed or licensed. Please install gurobipy and ensure you have a valid license.")

    num_places, num_transitions = I.shape

                     
    if len(c) != num_transitions:
        raise ValueError(f"Cost vector length ({len(c)}) must match number of transitions ({num_transitions})")
    if len(m_i) != num_places or len(m_f) != num_places:
        raise ValueError(f"Marking vectors must have length {num_places} (number of places)")
    if n <= 0:
        raise ValueError("Alignment length n must be positive")

                  
    model = gp.Model("Alignment_Gurobi")
    model.setParam("OutputFlag", 0)                   

                    
    vtype = GRB.BINARY if model_type == "milp" else GRB.CONTINUOUS

                                                                
    x = [[model.addVar(lb=0, ub=1, vtype=vtype, name=f"x_{j}_{k}") for k in range(n)]
         for j in range(num_transitions)]

                                                                           
    z = [model.addVar(lb=0, ub=1, vtype=vtype, name=f"z_{k}") for k in range(n)]

                                    
    model.setObjective(
        gp.quicksum(c[j] * x[j][k] for j in range(num_transitions) for k in range(n)),
        GRB.MINIMIZE
    )

                                                                    
    for i in range(num_places):
        model.addConstr(
            m_i[i] + gp.quicksum(I[i, j] * gp.quicksum(x[j][k] for k in range(n))
                                 for j in range(num_transitions)) == m_f[i],
            name=f"final_marking_{i}"
        )

                                                     
    for k in range(n):
        for i in range(num_places):
            model.addConstr(
                m_i[i] + gp.quicksum(I[i, j] * gp.quicksum(x[j][tau] for tau in range(k + 1))
                                     for j in range(num_transitions)) >= 0,
                name=f"feasible_{i}_{k}"
            )

                                                                 
    for k in range(n):
        model.addConstr(
            gp.quicksum(x[j][k] for j in range(num_transitions)) + z[k] == 1,
            name=f"move_or_stop_{k}"
        )

                                                             
    for k in range(n - 1):
        model.addConstr(z[k + 1] >= z[k], name=f"monotonic_{k}")

    return x, model