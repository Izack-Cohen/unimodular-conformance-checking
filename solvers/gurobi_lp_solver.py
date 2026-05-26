\
\
\
\
\
\
\
   

import time
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from numba import jit, njit, prange
import gurobipy as gp
from gurobipy import GRB

logger = logging.getLogger(__name__)


class OptimizedGurobiLPSolver:
                                                                         

    def __init__(self, time_limit: float = 120.0):
                                                        
        self.time_limit = time_limit
        self.env = gp.Env(empty=True)
        self.env.setParam('OutputFlag', 0)
        self.env.setParam('LogToConsole', 0)

                                            
        self.env.setParam('Method', 2)                                          
        self.env.setParam('Crossover', 0)                                      
        self.env.setParam('BarConvTol', 1e-6)                                        
        self.env.setParam('FeasibilityTol', 1e-6)
        self.env.setParam('OptimalityTol', 1e-6)
        self.env.setParam('Threads', 4)                        
        self.env.setParam('Presolve', 2)                       
        self.env.setParam('ScaleFlag', 3)                      

        self.env.start()

    def solve(self, trace: List[str], sync_prod: Any, costs: Dict) -> Dict:
                                                                                  
        start_total = time.time()

                                                     
        start_rg = time.time()
        rg = self._build_reachability_graph_optimized(trace, sync_prod, costs)
        rg_time = time.time() - start_rg

                                   
        start_solve = time.time()
        model = gp.Model("alignment_lp", env=self.env)
        model.Params.TimeLimit = max(1.0, self.time_limit - rg_time)

                                      
        flow_vars = {}
        for (u, v) in rg['edges']:
            flow_vars[(u, v)] = model.addVar(
                lb=0.0,
                ub=1.0,
                obj=rg['edge_costs'].get((u, v), 0),
                vtype=GRB.CONTINUOUS,
                name=f"flow_{u}_{v}"
            )

        model.update()

                                                       
        for node_id in rg['nodes']:
            if node_id == rg['source']:
                                          
                model.addConstr(
                    gp.quicksum(flow_vars.get((node_id, v), 0)
                                for v in rg['nodes']
                                if (node_id, v) in flow_vars) == 1,
                    name=f"source_{node_id}"
                )
            elif node_id == rg['sink']:
                                       
                model.addConstr(
                    gp.quicksum(flow_vars.get((u, node_id), 0)
                                for u in rg['nodes']
                                if (u, node_id) in flow_vars) == 1,
                    name=f"sink_{node_id}"
                )
            else:
                                                     
                inflow = gp.quicksum(flow_vars.get((u, node_id), 0)
                                     for u in rg['nodes']
                                     if (u, node_id) in flow_vars)
                outflow = gp.quicksum(flow_vars.get((node_id, v), 0)
                                      for v in rg['nodes']
                                      if (node_id, v) in flow_vars)
                model.addConstr(inflow == outflow, name=f"flow_{node_id}")

                  
        model.optimize()
        solve_time = time.time() - start_solve

                         
        if model.Status == GRB.OPTIMAL:
                                             
            alignment = self._extract_alignment_from_flow(flow_vars, rg, trace, sync_prod)

            return {
                'status': 'optimal',
                'cost': model.ObjVal,
                'alignment': alignment,
                'solve_time': solve_time,
                'total_time': time.time() - start_total,
                'rg_build_time': rg_time,
                'num_nodes': len(rg['nodes']),
                'num_edges': len(rg['edges'])
            }
        elif model.Status == GRB.TIME_LIMIT:
            return {
                'status': 'timeout',
                'cost': float('inf'),
                'alignment': None,
                'solve_time': solve_time,
                'total_time': time.time() - start_total
            }
        else:
            return {
                'status': 'failed',
                'cost': float('inf'),
                'alignment': None,
                'solve_time': solve_time,
                'total_time': time.time() - start_total
            }

    def _build_reachability_graph_optimized(self, trace: List[str], sync_prod: Any, costs: Dict) -> Dict:
                                                          
                                                       
        trace_positions = {event: [] for event in set(trace)}
        for i, event in enumerate(trace):
            trace_positions[event].append(i)

                                          
        n_trace = len(trace)
        n_places = len(sync_prod.sync_net.places)

                                     
        source = (0, sync_prod.initial_marking)
        sink = (n_trace, sync_prod.final_marking)

        nodes = set([source])
        edges = set()
        edge_costs = {}

                                
        visited = set([source])
        queue = [source]
        queue_idx = 0

                                                          
        marking_cache = {}

        while queue_idx < len(queue):
            current_node = queue[queue_idx]
            queue_idx += 1

            trace_pos, marking = current_node

                                             
            if trace_pos == n_trace and marking == sync_prod.final_marking:
                nodes.add(sink)
                edges.add((current_node, sink))
                edge_costs[(current_node, sink)] = 0
                continue

                                                    
            marking_tuple = tuple(sorted(marking.items()))
            if marking_tuple not in marking_cache:
                marking_cache[marking_tuple] = self._get_enabled_transitions_fast(
                    marking, sync_prod
                )
            enabled = marking_cache[marking_tuple]

                                             
            for trans in enabled:
                trans_label = str(trans.label) if trans.label else "tau"

                                                                           
                if trans_label == "tau" or trans_label.startswith(">>"):
                                
                    new_trace_pos = trace_pos
                    cost = costs.get('model_move', 1.0)
                elif trans_label.startswith("<<") or trans_label == "skip":
                              
                    if trace_pos < n_trace:
                        new_trace_pos = trace_pos + 1
                        cost = costs.get('log_move', 1.0)
                    else:
                        continue
                else:
                                                                            
                    if trace_pos < n_trace and trans_label == trace[trace_pos]:
                        new_trace_pos = trace_pos + 1
                        cost = costs.get('sync_move', 0.0)
                    else:
                        continue

                                                    
                new_marking = self._fire_transition_fast(marking, trans, sync_prod)
                new_node = (new_trace_pos, new_marking)

                          
                edges.add((current_node, new_node))
                edge_costs[(current_node, new_node)] = cost

                                             
                if new_node not in visited:
                    visited.add(new_node)
                    nodes.add(new_node)
                    queue.append(new_node)

        return {
            'nodes': nodes,
            'edges': edges,
            'edge_costs': edge_costs,
            'source': source,
            'sink': sink
        }

    def _get_enabled_transitions_fast(self, marking: Any, sync_prod: Any) -> List:
                                                              
        enabled = []

                                                              
        for trans in sync_prod.sync_net.transitions:
                                                   
            is_enabled = True
            for arc in trans.in_arcs:
                place = arc.source
                if marking.get(place, 0) < arc.weight:
                    is_enabled = False
                    break

            if is_enabled:
                enabled.append(trans)

        return enabled

    def _fire_transition_fast(self, marking: Any, transition: Any, sync_prod: Any) -> Any:
                                                                   
                                        
        new_marking = dict(marking)

                                         
        for arc in transition.in_arcs:
            place = arc.source
            new_marking[place] = new_marking.get(place, 0) - arc.weight
            if new_marking[place] == 0:
                del new_marking[place]

                                     
        for arc in transition.out_arcs:
            place = arc.target
            new_marking[place] = new_marking.get(place, 0) + arc.weight

        return new_marking

    def _extract_alignment_from_flow(self, flow_vars: Dict, rg: Dict, trace: List[str], sync_prod: Any) -> List:
                                                   
        alignment = []

                                                             
        current = rg['source']
        visited_edges = set()

        while current != rg['sink']:
                                                   
            next_node = None
            for (u, v) in rg['edges']:
                if u == current and (u, v) not in visited_edges:
                    if (u, v) in flow_vars and flow_vars[(u, v)].X > 0.5:
                        next_node = v
                        visited_edges.add((u, v))

                                                       
                        trace_pos_u = u[0]
                        trace_pos_v = v[0]

                        if trace_pos_v > trace_pos_u:
                                                   
                            if trace_pos_u < len(trace):
                                alignment.append(('sync', trace[trace_pos_u]))
                        else:
                                        
                            alignment.append(('model', None))
                        break

            if next_node is None:
                break
            current = next_node

        return alignment

    def __del__(self):
                                         
        if hasattr(self, 'env'):
            self.env.dispose()


                                                             
@njit
def check_marking_enabled_numba(marking_array: np.ndarray, trans_input_req: np.ndarray) -> bool:
                                                                              
    for i in prange(len(marking_array)):
        if marking_array[i] < trans_input_req[i]:
            return False
    return True


@njit
def fire_transition_numba(marking_array: np.ndarray,
                          trans_input: np.ndarray,
                          trans_output: np.ndarray) -> np.ndarray:
                                                               
    new_marking = marking_array.copy()
    new_marking -= trans_input
    new_marking += trans_output
    return new_marking