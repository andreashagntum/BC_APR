"""This script contains functions to run the proposed Branch-Price-Cut-and-Switch (or -Check) algorithm.
A sample call structure is provided in the main clause of this script.
"""
from src.core.branchandprice import get_solution
from experiments.compute_workforce import compute_workforce_rs
from experiments.utils import blockPrint, enablePrint
from src.instance_loader.instance_loader import load_InstWithTimeDiscr
import os
from src.rolling_horizon.utils import get_segments, adjust_inst_to_segment
import copy
from pathlib import Path

def solve_one_instance(inst, travel_times_per_bin, rs, no_gomory_cuts, branch_on_task_finish_times, use_dmp, segment_length,
                       lookahead, time_limit_bap, time_limit_heur, cores_per_thread, verbose, yuan_approach = False,
                       normalize_workforce = False, inst_wf = None, json_out_file = None, warmstart = False,
                       solve_only_with_best_tasks = True, best_task_cnt = 4):
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
    segment_length: int
        Length of each segment (in time steps) during rolling horizon separation
    lookahead: int
        Length of interval whose tasks will be considered during planning of each segment. Tasks that start after the segment
        end will be planned, but their solution will not be implemented and might be overwritten by the decision in the
        subsequent segment.
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
    all_tours: list[GH_tour]
        List of conducted tours
    """
    if yuan_approach and segment_length < inst.end_horizon * 2:
        raise Exception("Approach by Yuan et al. (2015) not supported for rolling horizon solving.")

    # 1. if no print output desired: disable print
    if not verbose:
        blockPrint()

    # 2. compute workforce size
    inst = compute_workforce_rs(inst, travel_times_per_bin, rs, normalize_workforce, inst_wf)

    # 3. compute segments
    th_segments = get_segments(inst, segment_length)

    # 4. iteratively solve segments and store solutions
    sol_per_seg = {} # maps segment indices to solutions
    workers_used_prev_sols = {k: {t: 0 for t in inst.workers[k]} for k in inst.workers} # keeps track of workers used by previous solutions
    all_tours = [] # all tours found by any of the segment solutions
    for seg_idx in range(len(th_segments)):

        # 4.1 deepcopy instance object and restrict planning horizon
        inst_seg = copy.deepcopy(inst)

        # 4.2 Adjust instance characteristics (time windows, horizon, task lists, etc.)
        # Note: This function also returns all tours from the last segment that have finished
        tours_returning_to_depot, all_tours = adjust_inst_to_segment(inst_seg, th_segments[seg_idx],
                                                                     travel_times_per_bin, sol_per_seg, lookahead,
                                                                     all_tours)
        # 4.3 Restrict workforce to considered planning horizon
        # Note: we no longer adjust the workforce according to the solutions from the previous segments because
        # workers occupied by tours from the previous segments that are still active during the new segment's time horizon
        # will be force-occupied downstream anyway: for each existing tour, an artificial label in the pricing network
        # will be created with a matching leave time, formation, and task sequence (up until the last task that was
        # finished before the time horizon). This tour then occupies the exact number of workers that it also occupied
        # in the solution of the last segment.
        inst_seg.workers = copy.deepcopy(inst.workers)
        inst_seg.workers = {k: {t: v for (t,v) in inst_seg.workers[k].items() if (t>=inst_seg.begin_horizon and t <= inst_seg.end_horizon)}
                            for k in inst_seg.workers}
        # An exception for this are tours that are returning to the depot and have not returned yet at the segment start:
        # We need to adjust the available workforce according to their occupancy
        for tour in tours_returning_to_depot:
            for k in tour.skill_comp_cnt:
                for t in range(tour.leave_time, tour.quantile_return_time):
                    if t in inst_seg.workers[k]:
                        inst_seg.workers[k][t] -= tour.skill_comp_cnt[k]
        inst_seg.workers_w_d = {k: {t: v for (t,v) in inst_seg.workers_w_d[k].items() if (t>=inst_seg.begin_horizon and t <= inst_seg.end_horizon)}
                            for k in inst_seg.workers_w_d}

        # 4.4 Solve instance and return solution
        print_str =  (f"Solving segment {th_segments[seg_idx]}. Lookahead: {lookahead}time steps. "
                      f"True considered planning horizon: [{inst_seg.begin_horizon}, {inst_seg.end_horizon}]")
        if verbose:
            print(print_str)
        sol = get_solution(inst_seg, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap, time_limit_heur,
                           cores_per_thread, yuan_approach, json_out_file, warmstart, solve_only_with_best_tasks,
                           best_task_cnt)
        sol_per_seg[seg_idx] = sol

        # 4.4 Update used worker counts
        if sol:
            if sol.infeasible_time > 0:
                print("Segment infeasible. Cannot continue.")
                return None, None, None, None
            for k in sol.workers_used:
                for t in sol.workers_used[k]:
                    workers_used_prev_sols[k][t] += sol.workers_used[k][t]
            # 4.5 Store tours
            if sol.optimal is not None:
                all_tours += sol.optimal.tours
            if sol.heuristic is not None:
                all_tours += sol.heuristic.tours


    # 5. Compute actual tours and task costs
    executed_tasks = [] # list of tasks that were executed, used for sanity check
    cost_per_task = {} # contains cost per task of the *actual* tour that was conducted and contained the task
    net_cost_per_task = {} # same as cost_per_task, but for net cost
    for tour in all_tours:
        executed_tasks += tour.tasks
        for task in tour.tasks:
            cost_per_task[task] = tour.task_cost_dict[task]
            net_cost_per_task[task] = tour.net_task_cost_dict[task]
    if len(set(executed_tasks)) != len(executed_tasks):
        raise Exception("Some tasks executed more than once.")
    if len(executed_tasks) != len(inst.tasks):
        raise Exception("Some tasks were not executed at all.")


    # 6. Enable print again
    if not verbose:
        enablePrint()

    # 7. Print solution summary
    if verbose:
        print(f"Solutions per segment:")
        for seg_idx in sol_per_seg:
            print("-" * 75, f"Segment {seg_idx}", "-" * 75)
            if sol_per_seg[seg_idx]:
                print(sol_per_seg[seg_idx].to_string())
            else:
                print("Segment infeasible or empty")
            print("\n\n\n")

    print(f"Total solution cost: {sum(cost_per_task.values())} (net {sum(net_cost_per_task.values())})")

    # 5. enable print again
    if not verbose:
        enablePrint()
    return sol_per_seg, all_tours




if __name__ == "__main__":
    ###################### SAMPLE CALL SIGNATURE ######################
    worker_quantile = 0.9 # parameter \gamma from the paper, used in the workforce constraints

    # specify and load instance
    # inst_type = "classic" # instances from Hagn et al. (2026)
    inst_type = "new" # new instances for large planning horizons, only solvable via rolling horizon
    inst_path = f"{Path(os.getcwd()).parent}/sample_instances/"
    if inst_type == "classic":
        inst_location = f"{inst_path}/90min-20fph-sf_157.json"
        # inst_location =  f"{inst_path}/60min-20fph-sif_157.json" # with no cuts: returns a few disaggregated-infeas. sols
        t_len_int = 2  # time length per instant (in minutes), only used for old instance format, must be coherent with instance content
        inst, travel_times_per_bin = load_InstWithTimeDiscr(inst_location, worker_quantile, t_len = t_len_int)
    elif inst_type == "new":
        inst_location = f"{inst_path}/1140min-5fph-sif_155.json" # large instance (should be solved with rolling horizon of length 60-90)
        inst, travel_times_per_bin = load_InstWithTimeDiscr(inst_location, worker_quantile, tt_data_path = inst_path)
    else:
        raise Exception(f"Unsupported instance type: {inst_type}")

    # specify additional required parameters
    rs = 0.9 if inst_type == "new" else 0.5 # workforce size factor (aka worker strength)
    yuan_approach = False # True iff. approach by Yuan et al. (2025) should be used
    no_gomory_cuts = 0 # maximum no. of gomory cuts to add at the root node (disabled when set to 0)
    branch_on_task_finish_times = True # branching on task finish times enabled
    use_dmp = True # use DMP to handle disaggregated-infeasible solutions (if True) or add no-good cuts (if False)
    solve_only_with_best_tasks = True # initially solve each pricing network only allow extensions along the most promising arcs
    best_task_cnt = 4 # no. of incident nodes in pricing networks that are considered to be the most promising ones
    time_limit_bap = 180 # BPC&S time limit
    time_limit_heur = 120 # heuristic time limit (only called after time_limit_bap is reached)
    segment_length = 60 if inst_type == "new" else inst.end_horizon * 2 # time horizon segmentation length in time steps (set to inst.end_horizon if no segmentation is desired)
    lookahead = 90 if inst_type == "new" else inst.end_horizon * 2 # time horizon segmentation length in time steps (set to inst.end_horizon if no segmentation is desired)

    cores_per_thread = os.cpu_count() # use all cores by default, Hagn et al. (2026) used 4
    verbose = True # True iff. print statements should be visible in the console

    sol_per_seg, all_tours = solve_one_instance(inst, travel_times_per_bin, rs, no_gomory_cuts, branch_on_task_finish_times,
                                                use_dmp, segment_length, lookahead, time_limit_bap, time_limit_heur, cores_per_thread,
                                                verbose, yuan_approach = yuan_approach,
                                                solve_only_with_best_tasks = solve_only_with_best_tasks,
                                                best_task_cnt = best_task_cnt)
