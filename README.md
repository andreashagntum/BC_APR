# Branch-Price-Cut-and-Switch (BPC&S) for Team Formation and Routing for Airport Baggage Handling Tasks

This repository contains the implementation of the Branch-Price-Cut-and-Switch (BPC&S)
algorithm proposed in:

> Andreas Hagn Rainer Kolisch, Giacomo Dall'Olio, Stefan Weltge (2026): *A Branch-Price-and-Cut Algorithm for Stochastic Workforce Scheduling
> with Heterogeneous Worker Formations.*
> arXiv preprint, DOI: [10.48550/arXiv.2405.20912](https://doi.org/10.48550/arXiv.2405.20912)

The algorithm solves a team formation and routing problem arising in airport bagging handling systems. 
Teams consisting of workers with different qualifications must be formed, and tasks must be assigned to them.
Tasks have hard time windows that can only be violated by at most a given probabilities, and travel
times are uncertain and time-dependent.

If you use this code, please cite:
```bibtex
@article{hagn2026bpcs,
  title   = {A Branch-Price-and-Cut Algorithm for Stochastic Workforce Scheduling
             with Heterogeneous Worker Formations},
  author  = {Hagn, Andreas; Kolisch, Rainer; Dall'Olio, Giacomo; Weltge, Stefan},
  journal = {arXiv preprint},
  year    = {2026},
  doi     = {10.48550/arXiv.2405.20912}
}
```

---

## Core Algorithm Overview

The BPC&S algorithm extends classical branch-and-price with three additional components:

- **Gomory cuts**: Chvátal-Gomory cuts are separated at the root node to tighten the LP relaxation.
- **AMP/DMP switch**: Tours are priced under an aggregated master problem (AMP) by default. When an integer solution fails the disaggregated feasibility check, the affected branch is re-solved using the full disaggregated master problem (DMP).
- **Pricing**: The pricing subproblem is an ESPPRC solved via a forward labeling algorithm that explicitly propagates start-time probability distributions and enforces an alpha chance constraint at each task.

Three branching rules are implemented, applied in priority order:
1. Branching on task quantile-case finish times
2. Branching on the number of active tours per time instant
3. Branching on individual tour variables (fallback)

---

## Benchmark approach by Yuan et al. (2015)
The repository also contains a reference implementation of the approach by [Yuan et al., 2015](https://doi.org/10.1080/00207543.2015.1082041).
It can be toggled on or off using a central boolean parameter. If toggled on, all algorithm components
that are newly proposed by Hagn et al. (2026) and which were not used by Yuan et al. (2015) are disabled.

## Repository Structure
```
├── config/
│   ├── config.py              # Global numerical tolerance parameters
│   ├── requirements.txt       # Python dependencies
│   └── conda_env.yaml         # Conda environment file
├── experiments/
│   ├── run.py                 # Main entry point; sample call included in __main__
│   ├── compute_workforce.py   # Workforce size computation from resource strength
│   └── utils.py               # Miscellaneous experiment utilities
└── src/
    ├── core/
    │   ├── branchandprice.py  # B&P tree, node management, solution tracking
    │   ├── columngeneration.py# Column generation loop and Gomory cut separation
    │   └── utils.py           # Shared core utilities
    ├── pricing/
    │   ├── graph.py           # Pricing network construction and management
    │   ├── dynamic_programming.py # ESPPRC labeling algorithm
    │   ├── pricing.py         # Pricing subproblem coordinator
    │   └── utils.py           # Pricing utilities
    ├── master/
    │   ├── master.py          # Restricted master LP/MIP (solved with Gurobi)
    │   └── feasibility_check.py # Feasibility check for disaggregated feasibility
    ├── branching/
    │   ├── branching_rules.py # Branching candidate selection functions
    │   └── node_constructor.py# Child node generation for each branching rule
    ├── cuts/
    │   └── gomory_cuts.py     # Chvátal-Gomory cut separation
    ├── instance_loader/
    │   └── instance_loader.py # JSON instance parser
    └── utils/
        └── gh_tour.py         # Tour data structure
```

---


## Requirements

- Python 3.12
- [Gurobi](https://www.gurobi.com/) (version 12.0.2, requires a valid licence)
- `networkx`, `numpy`

Install dependencies via pip:
```bash
pip install -r config/requirements.txt
```

Or create the full conda environment:
```bash
conda env create -f config/conda_env.yaml
conda activate BC_APR
```

---

## Input Format

Instances are provided as `.json` files containing task data (time windows, weights,
locations), worker formations, skill levels, and bin-dependent stochastic travel time
distributions. See `src/instance_loader/instance_loader.py` for the full specification.

## Usage

The main entry point is `experiments/run.py`. A minimal example:
```python
from src.instance_loader.instance_loader import load_InstWithTimeDiscr
from experiments.run import solve_one_instance

inst = load_InstWithTimeDiscr("path/to/instance.json", worker_quantile=0.9)

sol = solve_one_instance(
    inst=inst,
    rs=0.5,                        # workforce strength relative to trivial solution
    no_gomory_cuts=12,             # max. Gomory cuts at root node (0 to disable)
    branch_on_task_finish_times=True,
    use_dmp=True,                  # use AMP/DMP switch (False: use no-good cuts instead)
    time_limit_bap=180,            # BPC&S time limit in seconds
    time_limit_heur=120,           # fallback heuristic time limit in seconds
    cores_per_thread=4,
    verbose=True,
    solve_only_with_best_tasks=True,
    best_task_cnt=4
)
```
Instance files must be in JSON format as expected by `instance_loader.py`.
For reference, the instances used by Hagn et al.(2026) can be accessed in [this public git repository](https://github.com/andreashagntum/StochasticTeamFormationRoutingAirport).

---
