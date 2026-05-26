                            

import os


def load_petri_net(pnml_path):
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
        from pm4py.objects.petri.importer import importer as pnml_importer
    except ImportError:
        raise ImportError("PM4Py is not installed. Please install pm4py.")

    if not os.path.exists(pnml_path):
        raise FileNotFoundError(f"PNML file not found: {pnml_path}")

    try:
        net, initial_marking, final_marking = pnml_importer.apply(pnml_path)

        if net is None:
            raise RuntimeError("Failed to load Petri net from PNML file")
        if initial_marking is None or final_marking is None:
            raise RuntimeError("Failed to load markings from PNML file")

        return net, initial_marking, final_marking

    except Exception as e:
        raise RuntimeError(f"Error loading Petri net from {pnml_path}: {str(e)}")


def load_trace_from_xes(xes_path, trace_index=0):
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
        from pm4py.objects.log.importer.xes import factory as xes_importer
    except ImportError:
        raise ImportError("PM4Py is not installed. Please install pm4py.")

    if not os.path.exists(xes_path):
        raise FileNotFoundError(f"XES file not found: {xes_path}")

    try:
        log = xes_importer.apply(xes_path)

        if not log:
            raise RuntimeError("XES file contains no traces")

        if trace_index < 0 or trace_index >= len(log):
            raise IndexError(f"Trace index {trace_index} out of bounds. Log has {len(log)} traces.")

        trace = log[trace_index]

        if not trace:
            raise RuntimeError(f"Trace {trace_index} is empty")

                                 
        activity_labels = []
        for event in trace:
            if "concept:name" not in event:
                raise RuntimeError(f"Event missing 'concept:name' attribute: {event}")
            activity_labels.append(event["concept:name"])

        return activity_labels

    except Exception as e:
        raise RuntimeError(f"Error loading trace from {xes_path}: {str(e)}")