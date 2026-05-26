#!/usr/bin/env python3
                                  
\
\
\
\
\
\
\
\
   

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking


class ModelTransitionCache:
\
\
\
\
\
\
\
       

    __slots__ = (
        'net', 'im', 'fm',
        'silent_transitions', 'visible_transitions',
        'transition_to_idx', 'idx_to_transition',
        'input_places', 'output_places',
        'enabling_conditions', 'transition_read_places',
        'num_silent', 'num_visible',
        'P_list'                                               
    )

    def __init__(self, net: PetriNet, im: Marking, fm: Marking):
\
\
\
\
\
\
\
           
        self.net = net
        self.im = im
        self.fm = fm

                                       
        self._build_model_structure()

    def _build_model_structure(self):
\
\
\
           
                                                                   
        self.P_list = sorted(list(self.net.places), key=lambda p: p.name)

                                   
        self.silent_transitions = []
        self.visible_transitions = []

        for t in self.net.transitions:
            if t.label is None or t.label.lower() in ('tau', 'τ'):
                self.silent_transitions.append(t)
            else:
                self.visible_transitions.append(t)

        self.num_silent = len(self.silent_transitions)
        self.num_visible = len(self.visible_transitions)

                                         
        self._build_transition_maps()

                                        
        self._build_enabling_cache()

    def _build_transition_maps(self):
\
\
           
        all_transitions = list(self.net.transitions)
        self.transition_to_idx = {t: i for i, t in enumerate(all_transitions)}
        self.idx_to_transition = {i: t for t, i in self.transition_to_idx.items()}

                                                        
        self.input_places = {}                                   
        self.output_places = {}                                   

        for t in self.net.transitions:
                                                   
            inputs = []
            for arc in t.in_arcs:
                weight = getattr(arc, 'weight', 1)
                inputs.append((arc.source, weight))
            self.input_places[t] = inputs

                                                    
            outputs = []
            for arc in t.out_arcs:
                weight = getattr(arc, 'weight', 1)
                outputs.append((arc.target, weight))
            self.output_places[t] = outputs

    def _build_enabling_cache(self):
\
\
\
           
        self.enabling_conditions = {}
        self.transition_read_places = {}

        for t in self.net.transitions:
                                                       
            conditions = []
            read_places = []
            for place, weight in self.input_places[t]:
                conditions.append((place, weight))
                read_places.append(place)
            self.enabling_conditions[t] = conditions
            self.transition_read_places[t] = read_places

    def is_transition_enabled(self, marking_dict: Dict, transition) -> bool:
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
           
        for place, required in self.enabling_conditions[transition]:
            if marking_dict.get(place, 0) < required:
                return False
        return True

    def get_enabled_silent_transitions(self, marking_dict: Dict) -> List:
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
           
        enabled = []
        for t in self.silent_transitions:
            if self.is_transition_enabled(marking_dict, t):
                enabled.append(t)
        return enabled

    def get_enabled_model_moves(self, marking_dict: Dict,
                                trace_activities: Set[str]) -> List:
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
           
        enabled = []

                                                
        for t in self.silent_transitions:
            if self.is_transition_enabled(marking_dict, t):
                enabled.append(t)

                                                             
        for t in self.visible_transitions:
            if t.label not in trace_activities:
                if self.is_transition_enabled(marking_dict, t):
                    enabled.append(t)

        return enabled

    def fire_transition(self, marking_dict: Dict, transition) -> Dict:
\
\
\
\
\
\
\
\
\
           
        new_marking = marking_dict.copy()

                                         
        for place, weight in self.input_places[transition]:
            new_marking[place] = new_marking.get(place, 0) - weight
            if new_marking[place] <= 0:
                del new_marking[place]

                                     
        for place, weight in self.output_places[transition]:
            new_marking[place] = new_marking.get(place, 0) + weight

        return new_marking

    def batch_check_enabled(self, marking_dict: Dict,
                            transitions: List) -> List[bool]:
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
           
        enabled_list = []
        for t in transitions:
            enabled_list.append(self.is_transition_enabled(marking_dict, t))
        return enabled_list


class ReachabilityPruner:
\
\
\
\
\
       

    __slots__ = ('fm_dict', 'required_places', 'net')

    def __init__(self, net: PetriNet, fm: Marking):
\
\
\
\
\
\
           
        self.net = net
        self.fm_dict = {p: tokens for p, tokens in fm.items()}
        self.required_places = set(fm.keys())

    def can_reach_final(self, marking_dict: Dict, model_cache: ModelTransitionCache) -> bool:
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
           
                                                                   
        for req_place in self.required_places:
            current_tokens = marking_dict.get(req_place, 0)
            required_tokens = self.fm_dict[req_place]

            if current_tokens < required_tokens:
                                                                             
                can_produce = False
                for t in self.net.transitions:
                    for out_place, weight in model_cache.output_places[t]:
                        if out_place == req_place:
                            can_produce = True
                            break
                    if can_produce:
                        break

                                                                                  
                if not can_produce:
                    return False

        return True


def convert_marking_to_dict(marking_vector: np.ndarray, places: List) -> Dict:
\
\
\
\
\
\
\
\
\
       
    marking_dict = {}
    for i, tokens in enumerate(marking_vector):
        if tokens > 0:
            marking_dict[places[i]] = int(tokens)
    return marking_dict


def convert_dict_to_marking(marking_dict: Dict, places: List) -> np.ndarray:
\
\
\
\
\
\
\
\
\
       
    marking_vector = np.zeros(len(places), dtype=np.int32)
    for i, place in enumerate(places):
        marking_vector[i] = marking_dict.get(place, 0)
    return marking_vector