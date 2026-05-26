                               

def build_event_log_from_trace(trace_labels):
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
        from pm4py.objects.log.obj import EventLog, Trace, Event
    except ImportError:
        raise ImportError("PM4Py is not installed. Please install pm4py.")

    if not trace_labels:
        raise ValueError("Trace labels cannot be empty")

    trace = Trace()
    for i, label in enumerate(trace_labels):
        if not isinstance(label, str):
            raise ValueError(f"Activity label at position {i} must be a string, got: {type(label)}")
        trace.append(Event({"concept:name": label}))

    return EventLog([trace])


def get_alignment_synchronous_product(net, im, fm, trace_labels):
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
        from pm4py.algo.conformance.alignments.petri_net import algorithm as align_algo
    except ImportError:
        raise ImportError("PM4Py is not installed. Please install pm4py.")

                     
    if net is None or im is None or fm is None:
        raise ValueError("Petri net, initial marking, and final marking cannot be None")
    if not trace_labels:
        raise ValueError("Trace labels cannot be empty")

    try:
                                    
        log = build_event_log_from_trace(trace_labels)

                           
        aligned_traces = align_algo.apply_log(log, net, im, fm)

        if not aligned_traces:
            raise RuntimeError("Alignment computation returned empty result")

        return aligned_traces[0]                                  

    except Exception as e:
        raise RuntimeError(f"Synchronous product alignment failed: {str(e)}")