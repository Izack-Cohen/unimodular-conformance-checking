          
import numpy as np
from models.sync_product_incidence import build_sync_product_incidence

def _hardcoded_fig5():
    I = np.array([
        [-1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 1, 0,  1, 0, 0,  1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0,-1,  0, 0, 0,  0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 1, 0,  1, 0, 0,  1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0,-1,  0, 0, 0,  0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 1,  0, 0, 0,  0, 0, 0, 1,-1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,  0,-1, 0,  0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,  0, 1, 0,  0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0, 0, 0],
        [ 0, 0,  0, 0,-1,  0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0,  0, 0, 1,  0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0],
        [ 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 1, 1,-1, 0, 0, 0, 0, 0],
        [ 0, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [-1, 0,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
        [ 1,-1,  0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0],
        [ 0, 1, -1, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0],
        [ 0, 0,  1,-1, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1],
        [ 0, 0,  0, 1,-1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=int)
                                                                     
    c = np.array([0,0,0,0,0, 1,1,1,1, 0,1,1,1, 0,1,1,1,1,1,1], dtype=float)
    m_i = np.zeros(I.shape[0], dtype=int); m_i[0] = 1; m_i[12] = 1
    m_f = np.zeros(I.shape[0], dtype=int); m_f[11] = 1; m_f[16] = 1
    return I, c, m_i, m_f

def load_example_data(example: str = "fig3"):
    if example.lower() == "fig3":
                                                        
        from pm4py.objects.petri_net.obj import PetriNet, Marking
        from pm4py.objects.petri_net.utils import petri_utils

        net = PetriNet("Fig3Process")
        p = {i: PetriNet.Place(f"p{i}") for i in range(1, 7)}
        for pl in p.values():
            net.places.add(pl)
        t = {
            1: PetriNet.Transition("t1", "a"),
            2: PetriNet.Transition("t2", "b"),
            3: PetriNet.Transition("t3", "c"),
            4: PetriNet.Transition("t4", "d"),
            5: PetriNet.Transition("t5", "e"),
        }
        for tr in t.values():
            net.transitions.add(tr)
        arcs = [
            (p[1], t[1]), (t[1], p[2]), (t[1], p[3]),
            (p[2], t[2]), (p[2], t[4]),
            (p[3], t[3]), (p[3], t[4]),
            (t[2], p[4]), (t[3], p[5]),
            (t[4], p[4]), (t[4], p[5]),
            (p[4], t[5]), (p[5], t[5]),
            (t[5], p[6]),
        ]
        for s, d in arcs:
            petri_utils.add_arc_from_to(s, d, net)
        im = Marking(); im[p[1]] = 1
        fm = Marking(); fm[p[6]] = 1
        trace_labels = ["a", "b", "e"]

        I, c, m_i, m_f, _meta = build_sync_product_incidence(
            net, im, fm, trace_labels,
            cost_sync=0.0, cost_log=1.0, cost_model=1.0, cost_tau=0.0
        )
        return I.astype(int), c.astype(float), m_i.astype(int), m_f.astype(int)

    elif example.lower() == "fig5":
        return _hardcoded_fig5()

    else:
        raise ValueError("example must be 'fig3' or 'fig5'")
