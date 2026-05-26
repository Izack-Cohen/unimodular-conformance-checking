# URC²: Unimodular Reformulation for Conformance Checking

This repository contains the implementation and experiment scripts accompanying the paper:

> Izack Cohen. *Developing a Totally Unimodular Linear Program for Optimal Conformance Checking: When and Why It Complements A\**. Expert Systems with Applications, forthcoming.

URC² formulates alignment-based conformance checking as a minimum-cost network-flow linear program over a reachability graph of the synchronous product. The resulting node-arc incidence matrix is totally unimodular, so the LP admits integral optimal extreme-point solutions without binary decision variables.

Please cite the paper if you are using this code.

## Repository contents

```text
.
├── run_model_variant_experiments.py    # Main experiment runner
├── discover_model_variants.py          # Process-model discovery script
├── discover_all_models.sh              # Batch model-discovery script
├── analyze_results.py                  # Result analysis and visualizations
├── analyze_lp_wins.py                  # LP win-rate analysis
├── analyze_existing_results.py         # Re-analysis of existing result files
├── generate_summary_table.py           # Summary-table generation
├── dataset_statistics.py               # Dataset statistics
├── quickstart.py                       # Quick validation and demonstration script
├── examples.py                         # Usage examples
├── run_experiments.py                  # Alternative experiment runner
├── run_full_bpi2012.py                 # BPI 2012 experiment script
├── run_model_variant_experiments.py    # Model-variant experiment script
├── models/                             # Petri-net and synchronous-product utilities
├── solvers/                            # A*, LP, MILP, and solver-selection modules
├── experiments/                        # Experiment infrastructure
├── scripts/                            # Dataset-specific scripts
├── data/                               # Place input event logs here
├── results/                            # Experiment outputs
└── alignment_checker/results/          # Generated model/result files
```

## Installation

Create and activate a Python environment, then install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

The experiments that use Gurobi require a valid Gurobi installation and license. Academic licenses are available from Gurobi.

## Quick start

Run a small validation example:

```bash
python quickstart.py
```

Discover process-model variants from an event log:

```bash
python discover_model_variants.py data/BPI_Challenge_2012.xes --output-dir results
```

Compare A* and URC² on discovered models:

```bash
python run_model_variant_experiments.py data/BPI_Challenge_2012.xes results/ --max-traces 1000
```

Run the comparison for a single model file:

```bash
python run_model_variant_experiments.py data/BPI_Challenge_2012.xes models/model.pkl --single-model --max-traces 100
```

Analyze experiment results:

```bash
python analyze_results.py results/experiment_results.json
python analyze_lp_wins.py results/experiment_results.json
python generate_summary_table.py results/experiment_results.json
```

## Data

Raw event logs are not included in this repository. Place `.xes` or `.xes.gz` files in the `data/` directory before running the experiment scripts. The experiments in the paper used public process-mining benchmark logs and synthetic benchmark instances.

## Main modules

- `models/synchronous_product.py` and `models/sync_product_incidence.py`: construction of synchronous products and incidence structures.
- `solvers/a_star_solver.py`: wrapper around the PM4Py A* alignment implementation.
- `solvers/netflow_lp_solver_optimized.py`: optimized URC² network-flow LP implementation.
- `solvers/gurobi_lp_solver.py`: Gurobi-based LP formulation.
- `solvers/gurobi_milp_solver.py`: MILP formulation used for comparison.
- `solvers/solver_selector.py`: rule-based algorithm selection.
- `experiments/enhanced_runner.py`: experiment orchestration and result collection.

## Algorithm-selection rule

The empirical selection rule used in the paper recommends the LP approach when:

```text
L > 20 and (1 - F) * L > 1.5
```

where `L` is the trace length and `F` is the model fitness. Otherwise, A* is selected.

## Reproducibility notes

- Generated results, logs, local solver files, virtual environments, and large event logs are excluded by `.gitignore`.
- Gurobi-related output files such as `.lp`, `.mps`, `.sol`, and `gurobi.log` are excluded by default.
- The code expects input logs and generated model files to be supplied locally.

## Citation

If you use this repository or build on the URC² formulation in academic work, please cite the accompanying paper:

```bibtex
@article{cohen2026urc2,
  author  = {Cohen, Izack},
  title   = {Developing a Totally Unimodular Linear Program for Optimal Conformance Checking: When and Why It Complements A*},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Forthcoming}
}
```

A `CITATION.cff` file is also provided so that GitHub can display citation information for the repository.

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Contact

For implementation questions, please open an issue in this repository.
