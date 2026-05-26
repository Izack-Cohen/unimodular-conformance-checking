                       
\
\
\
   

from __future__ import annotations
from typing import Dict, List, Set, Tuple
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking

TAU_LABELS = {"tau", "τ", None}


class ModelCache:
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
       

    __slots__ = (
        'I_model', 'T_list', 'P_list', 'tau_indices',
        'non_tau_transitions', 'Pm', 'Tm', '_place_idx_cache'
    )

    def __init__(self, net: PetriNet):
\
\
\
\
\
           
                                                      
        self.I_model, self.T_list, self.P_list = self._construct_incidence_matrix(net)
        self.Pm, self.Tm = self.I_model.shape

                                            
        self.tau_indices: Set[int] = set()
        self.non_tau_transitions: Dict[str, List[int]] = {}

        for j, t in enumerate(self.T_list):
            lab = getattr(t, "label", None)
            if lab in TAU_LABELS:
                self.tau_indices.add(j)
            elif lab is not None:
                if lab not in self.non_tau_transitions:
                    self.non_tau_transitions[lab] = []
                self.non_tau_transitions[lab].append(j)

                                                           
        self._place_idx_cache = {p: i for i, p in enumerate(self.P_list)}

    def _construct_incidence_matrix(
            self, net: PetriNet
    ) -> Tuple[np.ndarray, List[PetriNet.Transition], List[PetriNet.Place]]:
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
        I = np.zeros((num_places, num_trans), dtype=np.int32)

                                      
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

    def marking_to_vector(self, marking: Marking) -> np.ndarray:
\
\
\
\
\
\
\
\
           
        m = np.zeros(self.Pm, dtype=np.int32)
        for p, tokens in marking.items():
            if p in self._place_idx_cache:
                m[self._place_idx_cache[p]] = int(tokens)
        return m