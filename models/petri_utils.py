                       
                                                   

from typing import List, Tuple
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet


def construct_incidence_matrix(net: PetriNet) -> Tuple[np.ndarray, List[PetriNet.Transition], List[PetriNet.Place]]:
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
       
                                                 
    P_list = sorted(list(net.places), key=lambda p: p.name)
    T_list = sorted(list(net.transitions), key=lambda t: t.name)
    
                           
    place_to_idx = {p: i for i, p in enumerate(P_list)}
    trans_to_idx = {t: j for j, t in enumerate(T_list)}
    
                                 
    num_places = len(P_list)
    num_trans = len(T_list)
    I = np.zeros((num_places, num_trans), dtype=int)
    
                                  
    for arc in net.arcs:
        if isinstance(arc.source, PetriNet.Place) and isinstance(arc.target, PetriNet.Transition):
                                                        
            place_idx = place_to_idx[arc.source]
            trans_idx = trans_to_idx[arc.target]
            I[place_idx, trans_idx] -= int(arc.weight) if hasattr(arc, 'weight') else 1
            
        elif isinstance(arc.source, PetriNet.Transition) and isinstance(arc.target, PetriNet.Place):
                                                       
            trans_idx = trans_to_idx[arc.source]
            place_idx = place_to_idx[arc.target]
            I[place_idx, trans_idx] += int(arc.weight) if hasattr(arc, 'weight') else 1
    
    return I, T_list, P_list
