"""This script contains functions to run the proposed Branch-Price-Cut-and-Switch (or -Check) algorithm.
A sample call structure is provided in the main clause of this script.
"""
from src.core.branchandprice import get_solution
from experiments.compute_workforce import compute_workforce_rs
from experiments.utils import blockPrint, enablePrint
from src.instance_loader.instance_loader import load_InstWithTimeDiscr
import os
from tqdm import tqdm
from src.rolling_horizon.utils import get_segments, min_val, max_val, expected_val, plot_workforce_usage
import src.rolling_horizon.monte_carlo as mc
import src.rolling_horizon.a_priori as ap
import copy
from pathlib import Path
import numpy as np

def solve_one_instance(inst, travel_times_per_bin, rs, no_gomory_cuts, branch_on_task_finish_times, use_dmp, segment_length,
                       lookahead, monte_carlo, time_limit_bap, time_limit_heur, cores_per_thread, rng, verbose, plot_worker_usage,
                       yuan_approach = False,
                       normalize_workforce = False, inst_wf = None, json_out_file = None, warmstart = False,
                       solve_only_with_best_tasks = True, best_task_cnt = 4):
    """Solve instance for a fixed, parametrized resource strength (rs).

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    rs: float
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
    monte_carlo: bool
        If enabled, at the end of each segment, travel times, service times, and time windows are sampled in a monte-carlo
        simulation. Then, worker availabilities for the subsequent period are based on the sampled durations.
    time_limit_bap: float
        Time limit for branch&price&cut&switch (in seconds).
    time_limit_heur: float
        Time limit for heuristic (in seconds).
    cores_per_thread: int
        Number of cores to use by gurobi when solving the master problem.
    rng: np.random.RandomState.Generator
        RNG object for sampling (only used if sample is set to True)
    verbose: bool
        Indicates if print statements should be forwarded to the console.
    plot_worker_usage: bool
        Indicates if a final plot of workforce usage over time should be created and shown.
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

    if segment_length > lookahead:
        raise Exception("Segment length can not be > lookahead.")

    # 1. if no print output desired: disable print
    if not verbose:
        blockPrint()

    # 2. compute workforce size
    inst = compute_workforce_rs(inst, travel_times_per_bin, rs, normalize_workforce, inst_wf)

    # 3. compute segments
    th_segments = get_segments(inst, segment_length)

    # 4. iteratively solve segments and store solutions
    if not monte_carlo:
        all_tours, sol_opt_segs, sol_heur_segs, heur_gaps, sol_per_seg = solve(inst, th_segments, lookahead, travel_times_per_bin,
                                                                  no_gomory_cuts, branch_on_task_finish_times, use_dmp,
                                                                  time_limit_bap, time_limit_heur, cores_per_thread, yuan_approach,
                                                                  json_out_file, warmstart, solve_only_with_best_tasks,
                                                                  best_task_cnt, verbose, plot_worker_usage)
    else:
        all_tours, sol_opt_segs, sol_heur_segs, heur_gaps, sol_per_seg = solve_monte_carlo(inst, th_segments, lookahead, rng,
                                                                              travel_times_per_bin, no_gomory_cuts,
                                                                              branch_on_task_finish_times, use_dmp, time_limit_bap,
                                                                              time_limit_heur, cores_per_thread,
                                                                              yuan_approach, json_out_file, warmstart,
                                                                              solve_only_with_best_tasks, best_task_cnt, verbose,
                                                                                           plot_worker_usage)

    # 5. Enable print again
    if not verbose:
        enablePrint()

    # 6. Compute actual tours and task costs
    if not all_tours:
        print("Instance terminated with no solution. At least one segment is infeasible")
        return
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
        raise Exception(f"Some tasks were not executed at all: {sorted(set(inst.tasks).difference(set(executed_tasks)),
                                                                       key = lambda x: inst.earliest_start[x])}")



    # 7. Print solution summary
    print("All tours:")
    for tour in all_tours:
        print(tour.to_string(), "\n")


    print(f"Total solution cost: {sum(cost_per_task.values())} (net {sum(net_cost_per_task.values())}).\n"
          f"No. of optimal segments: {len(sol_opt_segs)}. heuristic segments: {len(sol_heur_segs)} ({sol_heur_segs},"
          f"gaps (percent): [{[round(val * 100, 2) for val in heur_gaps]}")

    return sol_per_seg, all_tours



def solve(inst, th_segments, lookahead, travel_times_per_bin, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap,
          time_limit_heur, cores_per_thread, yuan_approach, json_out_file, warmstart, solve_only_with_best_tasks,
          best_task_cnt, verbose, plot_worker_usage):
    """Solve the instances using 'classic' rolling horizon.
    Time horizon is separated into disjunct segments.
    At the beginning of each segment, an optimal solution for the next segment's lookahead (start at the segment_start and lasts
    for 'lookahead' time steps) is derived. Then, proceeds to the beginning of the next segment.
    Tours derived during the
    last segment are then evaluated. If they have finished (according to quantile return times), they are frozen. If they
    have not yet finished but are currently returning to the depot, they must finish their depot return and then become
    available for disposition again. If they have not yet finished and are currently executing a task, they are passed
    to the solver such that it is possible to adjust their itinerary after the current task is finished.
    Workforce is then updated to account for currently occupied workforce and the segment is solved, until no segments remain.

    Parameters
    ----------
    See documentation of solve_one_instance().


    Returns
    -------
    all_tours: list[GH_tour]
       List of all tours that are part of the final solution.
    sol_opt_segs: dict
       List of segment indices in which the MIP was solved to optimality
    sol_heur_segs: dict
       List of segment indices in which thh MIP was not solved to optimality
    heur_gaps: list
       List of optimality gaps for all segments that were not solved to optimality
    sol_per_seg: dict
       Maps segment indices to the solution derived during team. These solutions normally differ from the final solution
       described by all_tours, since some segments also contain tours that were changed in a later segment.
    """

    # 1. iteratively solve segments and store solutions
    sol_per_seg = {} # maps segment indices to solutions
    sol_opt_segs = []
    sol_heur_segs = []
    heur_gaps = []
    workers_used_prev_sols = {k: {t: 0 for t in inst.workers[k]} for k in inst.workers} # keeps track of workers used by previous solutions
    all_tours = [] # all tours found by any of the segment solutions
    for seg_idx in tqdm(range(len(th_segments)),desc=f"seg_len={segment_length}, lookahead={lookahead}",
                        position=0, leave=True, dynamic_ncols=True):
        # 2. Deepcopy instance object
        inst_seg = copy.deepcopy(inst)

        # 3. Adjust instance characteristics (time windows, horizon, task lists, etc.)
        # Note: This function also returns all tours from the last segment that have finished
        tours_returning_to_depot, all_tours = ap.adjust_inst_to_segment(inst_seg, th_segments[seg_idx],
                                                                     travel_times_per_bin, sol_per_seg, lookahead,
                                                                     all_tours)

        # 4. Restrict workforce to considered planning horizon
        ap.restrict_workforce(inst, inst_seg, th_segments, seg_idx, all_tours, tours_returning_to_depot)


        # 5. Solve instance and return solution
        print_str =  (f"Solving segment {th_segments[seg_idx]}. Lookahead: {lookahead}time steps. "
                      f"True considered planning horizon: [{inst_seg.begin_horizon}, {inst_seg.end_horizon}]")
        if verbose:
            print(print_str)
        sol = get_solution(inst_seg, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap, time_limit_heur,
                           cores_per_thread, yuan_approach, json_out_file, warmstart, solve_only_with_best_tasks,
                           best_task_cnt)
        sol_per_seg[seg_idx] = sol


        # 6. Update used worker counts
        if sol:
            if sol.infeasible_time > 0:
                print("Segment infeasible. Cannot continue.")
                return None, None, None, None, None
            for k in sol.workers_used:
                for t in sol.workers_used[k]:
                    workers_used_prev_sols[k][t] += sol.workers_used[k][t]
            # 6.1 Store tours
            if sol.optimal is not None:
                all_tours += sol.optimal.tours
                sol_opt_segs.append(seg_idx)

            if sol.heuristic is not None:
                all_tours += sol.heuristic.tours
                sol_heur_segs.append(seg_idx)
                heur_gaps.append(sol.heuristic.optimality_gap)

    # 7. If desired: plot worker usage over time
    if plot_worker_usage:
        plot_workforce_usage(inst, all_tours)

    # 8. Return once all segments are finished
    return all_tours, sol_opt_segs, sol_heur_segs, heur_gaps, sol_per_seg


def solve_monte_carlo(inst, th_segments, lookahead, rng, travel_times_per_bin, no_gomory_cuts, branch_on_task_finish_times,
                      use_dmp, time_limit_bap, time_limit_heur, cores_per_thread, yuan_approach, json_out_file, warmstart,
                      solve_only_with_best_tasks,  best_task_cnt, verbose, plot_worker_usage):
    """Solve the instances using 'monte-carlo' rolling horizon.
    Time horizon is separated into disjunct segments.
    At the beginning of each segment, an optimal solution for the next segment's lookahead (start at the segment_start and lasts
    for 'lookahead' time steps) is derived.
    Then, for all derived tours, travel times, service times, and time windows are monte-carlo sampled.
    At the beginning of the next segment, these sampled values are used to separate tours into three groups, which are
    treated differently downstream:
    - (a) tours that have finished: only used for (net) cost computation
    - (b) tours that are not finished, but are currently returning to the depot: kept in a separate list tours_returning_to_depot,
          assumed return time is max(seg_start + 1, quantile_return_time assuming finish task of previous task is known)
    - (c) tours that are not finished and currently executing a task/traveling to a task: their finish time distribution
          is computed exactly up until to the last task/edge that they have visited before segment start. The remainder
          of the tour, just like before, can be adjusted by the solver.


    Then, proceeds to the beginning of the next segment. Tours derived during the
    last segment are then evaluated. If they have finished before the new segment's start, they are frozen. If they
    have not yet finished but are currently returning to the depot, they must finish their depot return and then become
    available for disposition again. If they have not yet finished and are currently executing a task, they are passed
    to the solver such that it is possible to adjust their itinerary after the current task is finished.
    At the start of each segment, a snapshot of the system is made and estimates on travel times, service times, and time
    windows are updated according to what has been observed so far.
    Workforce is then updated to account for currently occupied workforce and the segment is solved, until no segments remain.

    Parameters
    ----------
    See documentation of solve_one_instance().


    Returns
    -------
    all_tours: list[GH_tour]
       List of all tours that are part of the final solution.
    sol_opt_segs: dict
       List of segment indices in which the MIP was solved to optimality
    sol_heur_segs: dict
       List of segment indices in which thh MIP was not solved to optimality
    heur_gaps: list
       List of optimality gaps for all segments that were not solved to optimality
    sol_per_seg: dict
       Maps segment indices to the solution derived during team. These solutions normally differ from the final solution
       described by all_tours, since some segments also contain tours that were changed in a later segment.
    """

    # 1. Iteratively solve segments and store solutions
    sol_per_seg = {} # maps segment indices to solutions
    sol_opt_segs = []
    sol_heur_segs = []
    heur_gaps = []
    workers_used_prev_sols = {k: {t: 0 for t in inst.workers[k]} for k in inst.workers} # keeps track of workers used by previous solutions
    all_tours = [] # all tours found by any of the segment solutions
    knowledge_state = mc.KnowledgeState(inst)
    for seg_idx in tqdm(range(len(th_segments)),desc=f"seg_len={segment_length}, lookahead={lookahead}",
                        position=0, leave=True, dynamic_ncols=True):
        # 2. Deepcopy instance object
        inst_seg = copy.deepcopy(inst)

        # 3. Reset knowledge_state: remove active tasks lists etc.
        knowledge_state.reset()

        # 4. Sample travel times, service times, and time windows for all tours
        all_tours = mc.simulate(inst, inst_seg, th_segments[seg_idx], travel_times_per_bin, rng, all_tours,
                                knowledge_state)

        # 5. Join knowledge state and instance
        travel_times_per_bin = knowledge_state.join(inst_seg, travel_times_per_bin)

        # 6. Adjust instance characteristics (time windows, horizon, task lists, etc.)
        # Note: This function also returns all tours from the last segment that have finished (->all-tours)
        all_tours = mc.adjust_inst_to_segment(inst_seg, th_segments[seg_idx], travel_times_per_bin, lookahead,
                                                                     all_tours, knowledge_state)

        # 7. Restrict workforce to considered planning horizon
        mc.restrict_workforce(inst, inst_seg, knowledge_state, travel_times_per_bin,
                           th_segments, seg_idx, all_tours)

        # 8. Solve instance and return solution
        print_str =  (f"Solving segment {th_segments[seg_idx]}. Lookahead: {lookahead}time steps. "
                      f"True considered planning horizon: [{inst_seg.begin_horizon}, {inst_seg.end_horizon}]")
        if verbose:
            print(print_str)
        sol = get_solution(inst_seg, no_gomory_cuts, branch_on_task_finish_times, use_dmp, time_limit_bap, time_limit_heur,
                           cores_per_thread, yuan_approach, json_out_file, warmstart, solve_only_with_best_tasks,
                           best_task_cnt)
        sol_per_seg[seg_idx] = sol

        # 6. Update used worker counts
        if sol:
            if sol.infeasible_time > 0:
                print("Segment infeasible. Cannot continue.")
                return None, None, None, None, None
            for k in sol.workers_used:
                for t in sol.workers_used[k]:
                    workers_used_prev_sols[k][t] += sol.workers_used[k][t]
            # 6.1 Store tours
            if sol.optimal is not None:
                all_tours += sol.optimal.tours
                sol_opt_segs.append(seg_idx)
                # 4.5.1. Add boolean flag to indicate that uncertain parameter were not yet sampled for current tour
                for tour in sol.optimal.tours:
                    tour.simulation_done = False

            if sol.heuristic is not None:
                all_tours += sol.heuristic.tours
                sol_heur_segs.append(seg_idx)
                heur_gaps.append(sol.heuristic.optimality_gap)
                # 4.5.1. Add boolean flag to indicate that uncertain parameter were not yet sampled for current tour
                for tour in sol.heuristic.tours:
                    tour.simulation_done = False

    # 7. Final simulation to ensure that task costs are computed correctly
    all_tours = mc.simulate(inst, inst, (inst.end_horizon * 2, np.inf), travel_times_per_bin, rng, all_tours,
                                      knowledge_state)

    # 8. If desired: plot workforce usage
    if plot_worker_usage:
        plot_workforce_usage(inst, all_tours)

    # 9. Return once all segments are finished
    return all_tours, sol_opt_segs, sol_heur_segs, heur_gaps, sol_per_seg



if __name__ == "__main__":
    ###################### SAMPLE CALL SIGNATURE ######################
    worker_quantile = 0.9 # parameter \gamma from the paper, used in the workforce constraints

    ###################################### Block 1: Specify parameters ################################################
    # Instance class: either 'classic' or 'new'
    # inst_type = "classic" # instances from Hagn et al. (2026)
    inst_type = "new" # new instances for large planning horizons, only solvable via rolling horizon
    # Custom max. twviol and alpha (aka service level/chance constraint) value: MUST be set for classic instances
    # CAN be set of new instances; always overrides the value contain in the instance JSON
    custom_max_twviol = 5
    custom_service_level = 0.9
    # Solution approach: either (a) monte-carlo or (b) a-priori [for details, look into the docstrings of a_priori.py and monte_carlo.py
    monte_carlo = False
    sample_cnt = 100 # no. of service time/time window samples that should be created
    rng = np.random.default_rng(2947) # only used if monte_carlo == True
    # Estimator function that is used to derive deterministic estimates for service times and time windows
    # You can use the pre-implemented min_val, max_val, and expected_val functions, or define a function yourself
    estimator_fn = min_val if inst_type == "new" else min_val
    # Specify time length per instant (in minutes), must be coherent with instance content
    # Note: only needed for old instance format, new instance format contains this info already
    t_len_int = 2  if inst_type == "classic" else None
    # Specify path where instances are located
    inst_path = f"{Path(os.getcwd()).parent}/sample_instances/"
    ############################## Block 2: Load instances (no changes needed here) ####################################
    if inst_type == "classic":
        inst_location = f"{inst_path}/90min-20fph-sf_157.json"
        # inst_location =  f"{inst_path}/60min-20fph-sif_157.json" # with no cuts: returns a few disaggregated-infeas. sols
        inst, travel_times_per_bin = load_InstWithTimeDiscr(inst_location, worker_quantile, estimator_fn, sample_cnt=sample_cnt,
                                                            t_len = t_len_int, inst_type = inst_type, max_twviol = custom_max_twviol,
                                                            service_level = custom_service_level)
    elif inst_type == "new":
        inst_location = f"{inst_path}/1140min-5fph-sif_155.json" # large instance (should be solved with rolling horizon of length 60-90)
        inst, travel_times_per_bin = load_InstWithTimeDiscr(inst_location, worker_quantile, estimator_fn, sample_cnt=sample_cnt,
                                                            rng = rng, tt_data_path = inst_path, inst_type = inst_type,
                                                            max_twviol = custom_max_twviol, service_level = custom_service_level)
    else:
        raise Exception(f"Unsupported instance type: {inst_type}")

    ########################## Block 3: Specify additional parameters ##################################################
    # Solver configuration
    yuan_approach = False # True iff. approach by Yuan et al. (2025) should be used
    no_gomory_cuts = 0 # maximum no. of gomory cuts to add at the root node (disabled when set to 0)
    branch_on_task_finish_times = True # branching on task finish times enabled
    use_dmp = True # use DMP to handle disaggregated-infeasible solutions (if True) or add no-good cuts (if False)
    solve_only_with_best_tasks = True # initially solve each pricing network only allow extensions along the most promising arcs
    best_task_cnt = 4 # no. of incident nodes in pricing networks that are considered to be the most promising ones
    time_limit_bap = 180 # BPC&S time limit
    time_limit_heur = 120 # heuristic time limit (only called after time_limit_bap is reached)

    # Worker strength
    rs = 3 if inst_type == "new" else 0.5 # set to values >>1 for new instances: values <~0.9-1 often lead to infeasibility

    # Segment length and lookahead
    # Note: you can also set these values for classic instances, but inst.end_horizon * 2 ensures no segmentation
    # and can be used to validate results against results from the non-rolling horizon implementation
    segment_length = 60 if inst_type == "new" else inst.end_horizon * 2
    lookahead = 90 if inst_type == "new" else inst.end_horizon * 2

    # Other parameters
    plot_worker_usage = True # if you want to have a plot of the worker usage over time at the end
    cores_per_thread = os.cpu_count() # use all cores by default, Hagn et al. (2026) used 4
    verbose = False # True iff. print statements should be visible in the console

    print(f"Solving for segment length {segment_length}, lookahead {lookahead}, rs={rs}")
    sol_per_seg, all_tours = solve_one_instance(inst, travel_times_per_bin, rs, no_gomory_cuts, branch_on_task_finish_times,
                                                use_dmp, segment_length, lookahead, monte_carlo,
                                                time_limit_bap, time_limit_heur, cores_per_thread, rng,
                                                verbose, plot_worker_usage, yuan_approach = yuan_approach,
                                                solve_only_with_best_tasks = solve_only_with_best_tasks,
                                                best_task_cnt = best_task_cnt,)

    # optional: apply solution from a priori approach to multiple MC trajectories
    apply_to_monte_carlo = True
    if not monte_carlo and apply_to_monte_carlo:
        print("Applying solution to monte-carlo trajectories.")
        ap.apply_solution_to_monte_carlo(all_tours, inst, travel_times_per_bin, rng, sample_cnt)