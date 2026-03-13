"""This script contains functions to run the proposed Branch-Price-Cut-and-Switch (or -Check) algorithm.
A sample call structure is provided in the main clause of this script.
"""
from src.core.branchandprice import get_solution
from experiments.compute_workforce import compute_workforce_rs
from experiments.utils import blockPrint, enablePrint
from src.instance_loader.instance_loader import load_InstWithTimeDiscr
import os

def solve_one_instance(inst, rs, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap, time_limit_heur,
                cores_per_thread, verbose, yuan_approach = False, normalize_workforce = False, inst_wf = None,
                json_out_file = None, warmstart = False, solve_only_with_best_tasks = True, best_task_cnt = 4):
    """Solve instance for a fixed, parametrized resource strength (rs).

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    rs: float in [0,1]
        Workforce strength relative to the no. of workers required to trivially solve the instance.
    no_gomory_cuts: int
        Maximum no. of gomory cuts to be added at the root node.
    branch_on_task_finish_times: bool
        Indicates if branching on task finish time should be used.
    use_dmp: bool
        Indicates if switching between AMP and DMP should be used (if False, uses no-good cuts as in Dall'Olio & Kolisch (2023))
    time_limit_bap: float
        Time limit for branch&price&cut&switch (in seconds).
    time_limit_heur: float
        Time limit for heuristic (in seconds).
    cores_per_thread: int
        Number of cores to use by gurobi when solving the master problem.
    verbose: bool
        Indicates if print statements should be forwarded to the console.
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
        branch_on_task_finish_times = False, and use_dmp = True.
    normalize_workforce: bool
        True iff. workforce size calculation should be done based on a different instance file inst_wf (needed to ensure
        same workforce sizes for stochastic and deterministic instances).
    inst_wf: instance_loader.Instance
        Instance data for instance that should be used for workforce size calculation (typically: instance data from
        stochastic instance).
    warmstart: bool
        Indicates if a solution from a prior run (not necessarily a feasible one) should be loaded and reused.
    json_out_file: str
        Filepath and filename of warmstart solution.
    solve_only_with_best_tasks: bool
        If set to True, the pricing problem is first solved only allowing extensions along arcs that connect tasks with
        very large task rewards. If set to False, the pricing problem is always solved allowing all extensions.
    best_task_cnt: int
            No. of incident tasks for each node that should be considered if only_best_tasks is set to True
    Returns
    -------
    sol: branchandprice.GH_solution
        Solution data including tour data and team formations.
    """
    # 1. if no print output desired: disable print
    if not verbose:
        blockPrint()

    # 2. compute workforce size
    inst = compute_workforce_rs(inst, rs, normalize_workforce, inst_wf)

    # 3. call solver
    sol = get_solution(inst, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap, time_limit_heur,
                       cores_per_thread, yuan_approach, json_out_file, warmstart, solve_only_with_best_tasks,
                       best_task_cnt)

    # 4. print solution summary
    print(sol.to_string())

    # 5. enable print again
    if not verbose:
        enablePrint()
    return sol




if __name__ == "__main__":
    ##### SAMPLE CALL SIGNATURE #####
    # specify and load instance
    inst_location = ("H:/TUM-PC/Dokumente/Test Instances Stochastic Time Bins/real_data/sunny_weekday-True_holiday-True_peak-True/tt_scale1.0/"
                     "tw_factor2/2min/90min-20fph/90min-20fph-sf/twviol5_alpha0.9/90min-20fph-sf_157.json")
    inst_location = ("H:/TUM-PC/Dokumente/Test Instances Stochastic Time Bins/real_data/sunny_weekday-True_holiday-True_peak-True/tt_scale1.0/"
                     "tw_factor2/2min/60min-20fph/60min-20fph-sif/twviol5_alpha0.9/60min-20fph-sif_157.json") # with no cuts: returns a few disaggregated-infeas. sols
    worker_quantile = 0.9 # parameter \gamma from the paper, used in the workforce constraints
    inst = load_InstWithTimeDiscr(inst_location, worker_quantile)  # load instance .json file

    # specify additional required parameters
    rs = 0.5 # workforce size factor (aka worker strength)
    yuan_approach = False # True iff. approach by Yuan et al. (2025) should be used
    no_gomory_cuts = 12 # maximum no. of gomory cuts to add at the root node (disabled when set to 0)
    branch_on_task_finish_times = True # branching on task finish times enabled
    use_dmp = True # use DMP to handle disaggregated-infeasible solutions (if True) or add no-good cuts (if False)
    solve_only_with_best_tasks = True # initially solve each pricing network only allow extensions along the most promising arcs
    best_task_cnt = 4 # no. of incident nodes in pricing networks that are considered to be the most promising ones
    time_limit_bap = 180 # BPC&S time limit
    time_limit_heur = 120 # heuristic time limit (only called after time_limit_bap is reached)

    cores_per_thread = os.cpu_count() # use all cores by default, Hagn et al. (2026) used 4
    verbose = True # True iff. print statements should be visible in the console

    solve_one_instance(inst, rs, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap, time_limit_heur,
                       cores_per_thread, verbose, yuan_approach = yuan_approach, solve_only_with_best_tasks = solve_only_with_best_tasks,
                       best_task_cnt = best_task_cnt)
