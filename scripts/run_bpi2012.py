
                        
import os, yaml, sys
from pathlib import Path
from experiments.runner import run_dataset

def main():
    here = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((here/"configs/experiment_bpi2012.yaml").read_text())
    xes = cfg["xes_path"]
    if not os.path.isabs(xes):
                                                               
        xes = str((Path.cwd() / xes).resolve())
    run_dataset(
        xes_path=xes,
        sample_variants=cfg["discovery"]["top_variants"],
        max_sample_traces=cfg["discovery"]["max_sample_traces"],
        fitness_target=cfg["discovery"]["fitness_target"],
        precision_target=cfg["discovery"]["precision_target"],
        per_trace_time_limit=cfg["timing"]["per_trace_time_limit"],
        out_csv=cfg["output_csv"]
    )

if __name__ == "__main__":
    main()
