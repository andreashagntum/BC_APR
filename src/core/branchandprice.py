"""
Branch-Price-and-Cut algorithm for stochastic workforce scheduling with heterogeneous worker formations.

This module implements a customized Branch-Price-and-Cut-and-Switch (BPC&S) framework for
solving a stochastic task scheduling problem in which heterogeneous worker teams (formations)
must be routed and assigned to tasks with time windows and uncertain travel/service times.

Core algorithm overview:
  - Column generation (CG): iteratively solves a restricted master LP and a pricing subproblem
    (ESPPRC on a pricing network per formation) to generate columns with negative reduced cost.
  - Branch-and-bound tree: nodes are explored best-first; fractional LP solutions are resolved
    by branching on (in priority order) task worst-case finish times, number of active tours
    per time instant, tour arcs (only in approach by Yuan et al.(2015)), or individual tour variables.
  - Cutting: optional Gomory cuts are added at the root node to tighten the LP relaxation.
  - Aggregated / Disaggregated Master (AMP/DMP switch): tours are initially solved under an
    aggregated skill-composition master; infeasible integer solutions trigger a switch to the
    full disaggregated master for the affected branch. Alternatively, no-good cuts are used.
  - Feasibility check: integer solutions are validated against a detailed stochastic feasibility
    model before being accepted as incumbents.
  - Fallback heuristic: if the time limit is reached before proving optimality, a MIP heuristic
    is solved over all columns discovered so far to produce a best-effort solution with an
    optimality gap estimate.

Supporting classes defined here:
  GH_node       – B&P tree node, storing branching constraints and LP/MIP data.
  GH_solution   – Solution container tracking incumbent, bounds, runtimes, and statistics.
"""

import os
import copy
import time
import json
import math
from src.core import columngeneration as cg
from src.utils.gh_tour import GH_tour
from src.pricing.graph import PricingNetwork
from src.core.columngeneration import CG_timeout_exception
from src.core.utils import *
from src.branching.node_constructor import branch
from src.master.master import solve_heuristic_master
from src.master.feasibility_check import is_sol_actually_feasible_extended, is_sol_actually_feasible
from config.config import *

def get_solution(inst, no_gomory_cuts, branch_on_task_finish_times, use_dmp, tl_b_and_p, tl_heur, cores_per_thread,
                 yuan_approach, json_out_file, warmstart, solve_only_with_best_tasks, best_task_cnt):
    """Solve problem instance using Branch&Price.


    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    no_gomory_cuts: int
        Maximum no. of gomory cuts to be added at the root node.
    branch_on_task_finish_times: bool
        Indicates if branching on task finish time should be used.
    use_dmp: bool
        Indicates if switching between AMP and DMP should be used (if False, uses no-good cuts as in Dall'Olio & Kolisch (2024))
    tl_b_and_p: float
        Time limit for branch&price&cut&switch (in seconds).
    tl_heur: float
        Time limit for heuristic (in seconds).
    cores_per_thread: int
        Number of cores to use by gurobi when solving the master problem
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
        branch_on_task_finish_times = False, and use_dmp = True.
    warmstart: bool
        Indicates if a solution from a prior run (not necessarily a feasible one) should be loaded and reused.
    json_out_file: str
        Filepath and filename of warmstart solution.

    Returns
    -------
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics
    """

    # 1. preprocessing & preparations
    # 1.1 initialize solution metadata and Solution object
    cum_forbidden_tours = 0  # cumulated forbidden tours (derived via branching rule)
    solution = GH_solution(inst)  # Solution object, in which information regarding the solution is stored (obj. value etc.)
    all_w_int = all_weights_int(inst)  # check if all weights are integer

    # 1.2 best objective value, solution status etc.
    curr_best_int_value = math.inf
    curr_best_int_solution = None
    curr_best_tours = []  # tours making up the optimal solution
    disaggr_infeas_solutions = []  # solutions that failed the feasibility check

    # 1.3 generate fake tour used to initialize the column generation
    fake_tour = get_fake_tour(inst)
    all_tours = [fake_tour]

    # 1.4 get single-task tours and add them to column pool
    all_tours += get_start_at_earliest_tours(inst)
    all_tours_hashes = []
    # 1.5 reasonability check: get all single-task tours and check if there are any duplicates
    for tour in all_tours:
        tour_hash = tour.get_hash()
        if tour_hash in all_tours_hashes:
            raise Exception("Same tour generated twice in the initial solution")
        all_tours_hashes.append(tour.get_hash())

    # 1.6 if warmstart is desired: read existing tours from memory and introduce them to column set
    # NOTE: this functionality is not used by Hagn et al. (2026), but it can be helpful when e.g. computing the EVPI
    # of a stochastic solution, because solutions typically are (almost) feasible among many (or even all) travel
    # time scenarios.
    # NOTE: when transferring tours to the current column set, ONLY time windows are checked.
    # alpha chance-constraints are NOT checked. Thus, this functionality should ONLY be used for deterministic instances
    if warmstart:
        # 1.5.1 check if instance is deterministic
        for time_bin in inst.travel_times_per_bin:
            max_tt_cnt = max([len(inst.travel_times_per_bin[time_bin][k]) for k in inst.travel_times_per_bin[time_bin]])
            if max_tt_cnt > 1:
                raise Exception("Warmstart only supported for deterministic instances.")
        if os.path.isfile(json_out_file):
            with open(json_out_file, "r") as f:
                warmstart_sol = json.load(f)
                # 1.5.2 construct tours based on json content
                tours = [tour for tour in warmstart_sol.values()]
                feasible_tours = []
                for tour in tours:
                    is_feasible = True
                    curr_time = tour["leave"]
                    curr_node = "depot"
                    for task in tour["tasks"]:
                        curr_time_bin = inst.bin_per_instant[curr_time]
                        if task == "sink":
                            curr_time = max(
                                curr_time + max(inst.travel_times_per_bin[curr_time_bin][(curr_node, "depot")]),
                                inst.earliest_start[task])
                        else:
                            curr_time = max(curr_time + max(inst.travel_times_per_bin[curr_time_bin][(curr_node, task)]),
                                            inst.earliest_start[task])

                        curr_time = curr_time + inst.modes_with_domination[task][tour["formation"]]
                        # 1.5.3 if tour violates time window: do not add it as it is infeasible
                        if curr_time > inst.latest_finish[task]:
                            is_feasible = False
                            break

                    if is_feasible:
                        feasible_tours.append(tour)

                # 1.5.4 add tours to column pool
                for tour_obj in feasible_tours:
                    formation_w_d = {int(idx): tour_obj["formation_w_d"][idx] for idx in tour_obj["formation_w_d"]}
                    tour = GH_tour(formation_w_d, tour_obj["formation"])
                    tour.is_initial_tour = False
                    tour = get_default_skill_comps([tour], inst)[0]
                    # set values
                    tour.tasks = tour_obj["tasks"]
                    tour.cost = tour_obj["cost"]
                    tour.worst_case_start_time = tour_obj["worst_case_start_time"]
                    tour.quantile_finish_time = tour_obj["quantile_finish_time"]
                    tour.quantile_return_time = tour_obj["quantile_return"]
                    tour.tw_viol_prob = tour_obj["tw_viol_prob"]
                    tour.leave_time = tour_obj["leave"]
                    # add tour to column pool
                    all_tours.append(tour)

    # 1.7 initialize and memorize set of initial tours
    init_tours = []  # all tours used to initialize the column generation: fake tour + single-task tours
    init_tours.extend(all_tours)

    # 1.8 define root node
    root = GH_node(inst)
    root.is_root = True
    root.lb = solution.get_min_value() # compute trivial LB for root node
    root.tours = init_tours
    # 1.8.1 Yuan et al. (2015) only use a DMP-equivalent formulation => set root.solve_as_dmp to True
    if yuan_approach:
        root.solve_as_dmp = True

    # 1.9 define B&P tree object with
    tree = []
    tree.append(root)
    nodes_count = 0
    node_hashes = []

    # 1.10 maximum precision for travel time values in input data, used to properly round objective values
    # Note: this feature has a negligible impact on performance in Hagn et al. (2026), but depending on the instance
    # characteristics, it might be helpful for other instance sets
    maximum_input_precision = 0
    for bin in inst.travel_times_per_bin:
        for tup in inst.travel_times_per_bin[bin]:
            for travel_time in inst.travel_times_per_bin[bin][tup]:
                floating_digits = 0
                dist_prob = inst.travel_times_per_bin[bin][tup][travel_time]
                while True:
                    dist_prob_rounded = round(dist_prob, floating_digits)
                    if abs(dist_prob_rounded - dist_prob) < 0.1 ** 10:
                        if maximum_input_precision < floating_digits:
                            maximum_input_precision = floating_digits
                        break
                    floating_digits += 1

    # 1.11 compute lower bound for the objective value
    earliest_finish_sums = solution.get_min_value()

    # 1.12 once-setup pricing networks
    pricing_networks = {}  # store pricing networks to avoid re-creation at each node/CG step
    for formation_id in inst.formations:
        pricing_networks[formation_id] = PricingNetwork(inst, formation_id)


    # 2. start exploring the branching tree
    start_time = time.time()

    while tree:
        # 2.1 if elapsed time > max. runtime for B&P: compute a heuristic solution using the current column pool
        elapsed_time = time.time() - start_time
        if elapsed_time >= tl_b_and_p:
            solution = solve_heuristic_master(inst, solution, all_tours, root, pricing_networks, cum_forbidden_tours,
                                              nodes_count, tl_heur, start_time, elapsed_time, cores_per_thread)
            return solution



        # 2.2 explore next node and solve corresponding problem
        node = tree.pop(0)
        print("\nExploring node number " + str(nodes_count))
        if node.solve_as_dmp:
            print("Solving the skill composition-extended master problem")
        print(node.to_string())

        # 2.2.1 if parent lb is worse than best integer solution found: current node will not be optimal => skip this node
        if node != root and node.parent.lb >= curr_best_int_value:
            print("Node killed by parent with better integer LB")
            node.update_lb(math.inf)
            continue

        # 2.2.2 solve model at current node of branching tree
        (node_sol, node_val, node_tours, count_labels, count_dom_labels, total_time, time_master, time_pricing,
         time_dominance, time_start_distr_calc, master_setup_time, nr_iterations_cg, initial_label_cnt,
         only_best_task_cnt, runtime_exceeded) = cg.get_CG_LB_no_heuristic(pricing_networks, node, disaggr_infeas_solutions,
                                                         tl_b_and_p, elapsed_time,
                                                         earliest_finish_sums, no_gomory_cuts,
                                                         cores_per_thread, yuan_approach, solve_only_with_best_tasks,
                                                         best_task_cnt)

        # 2.2.3 sanity check if first tour in node.tours is fake tour
        workers_available = {}
        for k in inst.workers_w_d:
            workers_available[k] = inst.workers_w_d[k]
        for forced_tour in node.forced_tours:
            for k in forced_tour.formation_w_d:
                workers_available[k] -= forced_tour.formation_w_d[k]
        for k in workers_available:
            if node.tours[0].formation_w_d[k] < workers_available[k]:
                raise Exception("First tour is not the fake tour.")

        # 2.2.4 store results in solution and node objects
        solution, node, nodes_count, cum_forbidden_tours = store_solution_and_stats(solution, node, root,
                                                disaggr_infeas_solutions, node_sol, node_val, nodes_count, total_time,
                                                 master_setup_time, time_master, time_pricing, nr_iterations_cg,
                                                 initial_label_cnt, count_labels, count_dom_labels, time_dominance,
                                                 time_start_distr_calc, only_best_task_cnt, cum_forbidden_tours,
                                                                                    runtime_exceeded)


        # 2.3 if time limit has been reached: store total no. of (dominated) labels and continue
        # Note: if this happens, the master problem will be solved heuristically right afterward
        if runtime_exceeded:
            tree.insert(0, node)
            # add tours to column pool
            for i in range(1, len(node_tours)):  # start at index 1 to skip fake tour
                tour = node_tours[i]
                if tour.get_hash() not in all_tours_hashes:
                    all_tours.append(tour)
                    all_tours_hashes.append(tour.get_hash())
            # skip rest of the loop: immediately solve heuristic master problem
            continue


        # 2.4 if all weights are integer: can round optimal value to the next possible objective value
        # (given by the maximum precision of the values for travel time probabilities)
        if all_w_int:
            if not node_val == math.inf:
                node_val = round(node_val, maximum_input_precision)


        # 2.5 check if current node is infeasible
        if not node_sol or node_sol[0] > eps_global: # fake tour (index 0) used => node must be infeasible
            print("Node number " + str(nodes_count) + " is infeasible")
            node.update_lb(math.inf)
            continue

        # 2.6 else: update lower bound at node, add new tours found during column generation to all_tours
        else:
            node.own_lb = node_val  # store LB of current node (node.lb is updated whenever a better LB is found in a child node)            node.update_lb(node_val)
            node.update_lb(node_val)
            for i in range(1, len(node_tours)):  # start at index 1 to skip fake tour
                tour = node_tours[i]
                if tour.get_hash() not in all_tours_hashes:
                    all_tours.append(tour)
                    all_tours_hashes.append(tour.get_hash())

        # 2.7 if solution is integer: check feasibility and update upper bounds if necessary
        if is_sol_integer(node_sol):
            # 2.7.1 check if solution is better than best integer solution previously found
            if node_val < curr_best_int_value:
                (solution, tree, node, curr_best_int_value, curr_best_int_solution,
                 curr_best_tours) = check_integer_sol(node, node_tours, inst, solution, node_sol, node_val, tree,
                                                      disaggr_infeas_solutions, curr_best_int_value, curr_best_int_solution,
                                                      curr_best_tours, use_dmp)


        # 2.8 else: branch on current solution
        else:
            # 2.8.1 if solution worse than best integer feasible solution: can not be optimal => prune branch
            if node_val >= curr_best_int_value:
                print("Node pruned")
            # 2.8.2 else: branch
            else:
                node, tree, solution, node_hashes = branch(node, node_tours, solution, node_sol, tree, node_hashes,
                                                           branch_on_task_finish_times, yuan_approach)



    elapsed_time = time.time() - start_time

    # 3. evaluate best solution found
    # 3.1 if no integer feasible solutions have been found: return solution, problem deemed infeasible
    if curr_best_tours == []:
        solution.infeasible_time = elapsed_time
        return solution
    # 3.2 if fake tour in solution: problem deemed infeasible
    if math.isclose(curr_best_int_solution[0], 1):
        solution.infeasible_time = elapsed_time
        return solution

    print(f"value of best integer solution found: {curr_best_int_value}")
    solution.add_optimal_solution(curr_best_int_solution, curr_best_int_value, curr_best_tours, elapsed_time,
                                  solution.root_lb)

    solution.tot_columns = len(all_tours)
    solution.avg_forbidden_tours_per_node = cum_forbidden_tours / nodes_count
    solution.time_setup_pricing = get_setup_pricing(pricing_networks)  # runtime for setting up/adjusting pricing networks
    return solution




def update_infeas_sol(disaggr_infeas_solutions, node_sol, node_tours):
    """Add newly found disaggregated-infeasible solution to set of disaggregated-infeasible solutions.

    Parameters
    disaggr_infeas_solutions: list
        List of disaggregated infeasible solutions found so far
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects

    """

    aggr_sol = []
    for i in node_sol:
        if node_sol[i] > 1 - eps_global / 10:
            aggr_sol.append(node_tours[i])
    disaggr_infeas_solutions.append(aggr_sol)


def check_integer_sol(node, node_tours, inst, solution, node_sol, node_val, tree, disaggr_infeas_solutions,
                      curr_best_int_value, curr_best_int_solution, curr_best_tours, use_dmp):
    """Applies the feasibility check to an integer solution that is better than the current incumbent. If the feasibility
    check is passed, updates the incumbent and the upper bound. Else, either a switch to the DMP is performed or a no-good
    cut is added to the master problem.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    node_tours: list
        List of GH_tour tours at current node
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects
    node_val: float
        Objective value at current node
    tree: list
        Current branching tree, list of GH_node objects
    disaggr_infeas_solutions: list
        List of disaggregated infeasible solutions found so far
    curr_best_int_value: float
        Value of best disaggregated-feasible integer solution found (upper bound)
    curr_best_int_solution: GH_solution
        Best disaggregated-feasible integer solution found
    curr_best_tours: list
        Tours that make up the best disaggregated-feasible integer solution found
    use_dmp: bool
        Indicates if switching between AMP and DMP should be used (if False, uses no-good cuts as in Dall'Olio & Kolisch (2024))

    Returns
    -------
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics
    tree: list
        Current branching tree, list of GH_node objects
    node: GH_node
        Current node in the branching tree
    curr_best_int_value: float
        Value of best disaggregated-feasible integer solution found (upper bound)
    curr_best_int_solution: GH_solution
        Best disaggregated-feasible integer solution found
    curr_best_tours: list
        Tours that make up the best disaggregated-feasible integer solution found
    """

    # only count feasibility checks of solutions without the fake column
    if not math.isclose(node_sol[0], 1):
        solution.feas_checks += 1
    # 1. perform feasibitity check
    # if DMP should be used: perform extended feasibility check and update skill compositions
    if use_dmp:
        # need to perform extended feasibility check on deepcopy of node.tours because feas.check modifies
        # the tour set given as an argument. If the feasibility check passes, we do not want to change the
        # current tour set as this might interfere with other tour usages
        tours_feas_check = copy.deepcopy(node_tours)
        (tours_feas_check, changed_tours_hashes_idxs, changed_tours_hashes, feasible, time_feas_check,
         time_setup_feas_check) = is_sol_actually_feasible_extended(node_sol, tours_feas_check, inst.workers,
                                                                    node.forbidden_tours, node.solve_as_dmp)
        # adjust skill comps at sibling node
        if node.sibling is not None:
            for tour in node.sibling.tours:
                if len(changed_tours_hashes) == 0:  # stop after all changed tours have been found
                    break
                tour_hash = tour.get_hash_no_comp()
                if tour_hash in changed_tours_hashes:
                    idx_curr = changed_tours_hashes.index(tour_hash)
                    # only change skill comp. of tour at sibling node if tour is not forced
                    # forced tours always come at the end of node_tours, but are not part of node.tours!
                    if changed_tours_hashes_idxs[idx_curr] < len(node.tours):
                        tour.skill_comp = node.tours[changed_tours_hashes_idxs[idx_curr]].skill_comp
                        tour.skill_comp_cnt = node.tours[changed_tours_hashes_idxs[idx_curr]].skill_comp_cnt
                    del changed_tours_hashes[idx_curr]
                    del changed_tours_hashes_idxs[idx_curr]

    # if no DMP should be used: only check feasibility, do not change skill compositions
    else:
        feasible, time_feas_check, time_setup_feas_check = is_sol_actually_feasible(node_sol, node_tours,
                                                                                    inst.workers)
    # log time spent checking feasibility
    solution.time_feas_check += time_feas_check
    solution.time_setup_feas_check += time_setup_feas_check

    # 2. if solution is feasible: store solution information as best integer solution found so far
    if feasible:
        if use_dmp:
            optimal_tours_skill_comp = tours_feas_check
        else:  # if DMP is not used: need to run feasibility check to get the actual proper skill compositions for each team
            optimal_tours_skill_comp = is_sol_actually_feasible_extended(node_sol, node_tours, inst.workers,
                                                                         node.forbidden_tours,
                                                                         node.solve_as_dmp)[0]

        solution.disaggr_feas_solutions += 1
        # avoid counting feas_checks for solutions with fake column
        if not math.isclose(node_sol[0], 1):
            solution.feas_checks_passed += 1
        curr_best_int_value = node_val  # new better solution found
        curr_best_int_solution = node_sol
        curr_best_tours = optimal_tours_skill_comp
        print("Better integer solution found: " + str(node_val))
        # store TW violation probabilities for each task and tour lengths
        tour_lengths = {}
        tw_viol_probs = {}
        for idx in node_sol:
            if node_sol[idx] > 1 - eps_global * 10:
                tour = optimal_tours_skill_comp[idx]
                if len(tour.tasks) not in tour_lengths:
                    tour_lengths[len(tour.tasks)] = 0
                tour_lengths[len(tour.tasks)] += 1
                for task in tour.tasks:
                    if task not in inst.tasks:
                        continue
                    if tour.tw_viol_prob[task] > eps_global / 100 and tour.tw_viol_prob[task] < 1 - eps_global / 100:
                        tw_viol_probs[task] = tour.tw_viol_prob[task]
        tw_viol_probs = sorted(tw_viol_probs.items(), key=lambda x: x[1], reverse=True)
        tw_viol_probs = dict(tup for tup in tw_viol_probs)
        solution.tw_viol_prob = tw_viol_probs.copy()
        tour_lengths = sorted(tour_lengths.items(), key=lambda x: x[0])
        solution.tour_lengths = dict(tour_lengths)

    # 3. else: solution is infeasible: forbid its combination of tours (if use_dmp = False)
    # or mark node as DMP (if use_dmp = True) and re-insert node into tree such that current node is
    # re-solved in the next iteration
    else:
        solution.disaggr_infeas_solutions += 1
        # if DMP should be used: mark current and sibling node and switch to solving the DMP
        if use_dmp:
            print("Found integer infeasible solution. Switching to DMP in current and sibling branch")
            if node.solve_as_dmp:
                raise Exception("Got an aggregated infeasible solution in an extended master problem")
            node.solve_as_dmp = True
            tree.insert(0, node)

            # solve extended master model in current node and sibling node
            print(
                "No feasible flow of workers, solving disaggregated master problem in current and sibling branch")
            if node.parent is not None:  # root node does not have a sibling branch
                node.parent.left_child.solve_as_dmp = True
                node.parent.right_child.solve_as_dmp = True
                node.parent.left_child.switched_to_disaggr = True
                node.parent.right_child.switched_to_disaggr = True

        # else: forbid current solution explicitly via an additional constraint in the master problem
        else:
            print("Found integer infeasible solution. I am forbidding it and re-solving.")
            update_infeas_sol(disaggr_infeas_solutions, node_sol, node_tours)
            tree.insert(0, node)

    return solution, tree, node, curr_best_int_value, curr_best_int_solution, curr_best_tours

def get_fake_tour(inst):
    """Create fake tour consisting of all available workers that execute all tasks such that they are finished at the
    last time in the time horizon. Fake tour starts at beginning of time horizon and returns at end of time horizon.
    Note: in most cases, this tour is not feasible for the master problem. It can still be used to start the column
    generation procedure and check for infeasibility (if fake tour is part of optimal solution: problem is infeasible)

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    fake_tour: GH_tour
        Tour executing all tasks at the latest possible time, occupying the entire workforce for the entire planning
        horizon
    """

    # 1. get maximum workforce size for each skill level
    fake_formation_w_d = {}
    fake_formation_id = "f_"
    for level in inst.skill_levels:
        fake_formation_w_d[level] = inst.workers_w_d[level]
        fake_formation_id +=  f"{level}:{inst.workers[level]},"
    fake_formation_id = fake_formation_id.rstrip(",")

    # 2. create GH_tour object including all tasks, each finished at their latest finish time
    fake_tour = GH_tour(fake_formation_w_d, None)
    fake_tour.is_fake_tour = True
    fake_tour.tw_viol_prob = {}
    fake_tour.tasks = inst.tasks.copy()
    fake_tour.is_initial_tour = True

    # 3. calculate cost of fake tour
    cost = 0
    # fake_tour.task_cost_dict = {} # debugging only
    for task in inst.tasks:
        cost += inst.weights[task] * (inst.latest_finish[task] + (inst.latest_finish_viol[task] -
                                                                  inst.latest_finish[task])**2)
        fake_tour.tw_viol_prob[task] = 1        # TW viol. prob. for take tour set to 1 (dummy value)
    # scale costs to ensure that fake_tour.cost > cost of any feasible tour
    fake_tour.cost = cost * 1.1

    # 4. set start/finish times of tasks and leave/return time of team
    fake_tour.leave_time = inst.begin_horizon
    fake_tour.quantile_return_time = inst.end_horizon
    for task in fake_tour.tasks:
        fake_tour.worst_case_start_time[task] = inst.earliest_start[task]
        fake_tour.quantile_finish_time[task] = inst.latest_finish_viol[task]

    # 5. set fake skill composition
    skill_comp = {}
    skill_comp_cnt = {}
    for k in inst.workers:
        skill_comp_cnt[k] = inst.workers[k]
        skill_comp[k] = {}
        for kk in inst.workers:
            if kk < k:
                continue
            if kk == k:
                skill_comp[k][kk] = inst.workers[k]
            else:
                skill_comp[k][kk] = 0
    fake_tour.skill_comp = skill_comp
    fake_tour.skill_comp_cnt = skill_comp_cnt
    fake_tour.formation_id = fake_formation_id
    fake_tour.quantile_finish_time["sink"] = fake_tour.quantile_return_time

    
    return fake_tour
        
def get_start_at_earliest_tours(inst):
    """Generates single-task tours for each possible combination of tasks and suitable profiles. Task is assumed
    to be finished at its earliest possible finish time.
    These tours are used to initialize the column set at the beginning of the algorithm.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    tours: list
        List of single-task GH_tour objects

    """
    tours = []
    for formation_id in inst.tasks_per_formation:
        for task in inst.tasks_per_formation[formation_id]:
            finish_time = inst.earliest_start[task] + inst.modes[task][formation_id]

            tour = GH_tour(inst.formations_w_d[formation_id], formation_id)
            tour.is_initial_tour = True
            tour.tw_viol_prob[task] = 0
            tour = get_default_skill_comps([tour], inst)[0]
            tour.tasks = [task]
            tour.cost = finish_time * inst.weights[task]        # best-case finish time is always within task's time window
            
            tour.worst_case_start_time[task] = inst.earliest_start[task]
            tour.quantile_finish_time[task] = finish_time
            # get earliest possible leave time
            leave_time = min([t - max(inst.travel_times_per_bin[inst.bin_per_instant[t]][(inst.depot, task)])
                          for t in range(inst.earliest_start[task], inst.latest_start[task] + 1)])
            tour.leave_time = leave_time
            tour.quantile_return_time = finish_time + max(inst.travel_times_per_bin[inst.bin_per_instant[finish_time]][(inst.depot, task)])
            tour.quantile_finish_time["sink"] = tour.quantile_return_time
            tours.append(tour)
    return tours



class GH_node():

    def __init__(self, inst):
        """Class for nodes within the branch and bound tree. Contains information about branches on variables/tours (i.e.
        forced and forbidden tours/tasks), lower bounds, parent nodes, used formulation (AMP or DMP), and some other
        statistics.

        Parameters
        ----------
        inst: instance_loader.Instance
            Contains all necessary instance data read from input files.

        """
        self.is_root = False
        self.inst = inst
        self.tours = []
        self.t_gr = {}
        self.t_le = {}
        self.forced_arcs = []
        self.forbidden_arcs = []
        self.forced_tours = []
        self.forbidden_tours = []
        self.forced_tasks = []
        self.t_max_gr = {}
        for task in inst.tasks:
            self.t_max_gr[task] = -1
        self.t_max_le = {}
        for task in inst.tasks:
            self.t_max_le[task] = math.inf
        self.gomory_cuts_lhs = []
        self.gomory_cuts_rhs = []
        self.u_task = []
        self.u_kt = []
        self.depth = 0
        self.parent = None
        self.left_child_lb = None
        self.right_child_lb = None
        self.solve_as_dmp = False        # True iff. for each formation: consider all skill comps in the model
        self.sibling = None
        self.contains_fake_tour = True          # True iff. node contains fake tour in its set of tours
        self.switched_to_disaggr = False     # set to True once we switch to the disaggregated master problem

    def __lt__(self, other):
        """Compares depths of self and other.
        Parameters
        ----------
        other: GH_node
            branching tree node to compare depth again

        Returns
        -------
        self_has_depth: bool
            True iff. self has a a smaller depth than other in the branching tree.
        """
        # less than defined based on depth in the B&P tree
        return self.depth < other.depth

    def clone(self):
        """Creates a mutable copy of the current GH_node object. Any changes made to the copy object will not be applied
        to the current object and viceo versa.

        Returns
        -------
        cln: GH_node
            Cloned GH_node object
        """

        # copy all attributes, apply deepcopy only where necessary
        cln = GH_node(self.inst)
        cln.tours = copy.deepcopy(self.tours)
        cln.t_gr = copy.deepcopy(self.t_gr)
        cln.t_le = copy.deepcopy(self.t_le)
        cln.forced_tours = copy.deepcopy(self.forced_tours)
        cln.forbidden_tours = copy.deepcopy(self.forbidden_tours)
        cln.forced_tasks = copy.deepcopy(self.forced_tasks)
        cln.t_max_gr = copy.deepcopy(self.t_max_gr)
        cln.t_max_le = copy.deepcopy(self.t_max_le)
        cln.forced_arcs = copy.deepcopy(self.forced_arcs)
        cln.forbidden_arcs = copy.deepcopy(self.forbidden_arcs)
        cln.solve_as_dmp = self.solve_as_dmp
        cln.gomory_cuts_rhs = self.gomory_cuts_rhs.copy()
        cln.gomory_cuts_lhs = []
        for cut in self.gomory_cuts_lhs:
            cln.gomory_cuts_lhs.append(cut.copy())
        cln.u_task = []
        for cut in self.u_task:
            cln.u_task.append(cut.copy())
        cln.u_kt = []
        for cut in self.u_kt:
            cln.u_kt.append(cut.copy())

        cln.contains_fake_tour = self.contains_fake_tour

        return cln
    
    def to_string(self):
        """Generate string containing information about node that can be printed.

        Returns
        -------
        s: str
            String providing an overview of all branching decisions (incl. forced/forbidden tours/arcs) at current node
        """

        s = ""
        for t_tilde in self.t_le:
            s = s + str(t_tilde) + "<=" + str(self.t_le[t_tilde]) + "\n"
        for t_tilde in self.t_gr:
            s = s + str(t_tilde) + ">=" + str(self.t_gr[t_tilde]) + "\n"
        for t_tilde in self.t_max_le:
            if self.t_max_le[t_tilde] != math.inf:
                s = s + "finish_" + str(t_tilde) + "<=" + str(self.t_max_le[t_tilde]) + "\n"
        for t_tilde in self.t_max_gr:
            if self.t_max_gr[t_tilde] != -1:
                s = s + "finish_" + str(t_tilde) + ">=" + str(self.t_max_gr[t_tilde]) + "\n"
        if self.forced_tours:
            s = s + "Forced:\n"
        for tour in self.forced_tours:
            s = s + "{start:" + f"{tour.leave_time}, "
            for task in tour.tasks:
                s = s + str(task) + ":[" + str(tour.worst_case_start_time[task]) + "," + str(tour.quantile_finish_time[task]) + "], "
            s = s[:-2] + "}" + str(tour.formation_w_d)
            s += "skillcomp:" + str(tour.skill_comp) + "\n"
        if self.forbidden_tours:
            s = s + "Forbidden:\n"
        for tour in self.forbidden_tours:
            s = s + "{start:" + f"{tour.leave_time}, "
            for task in tour.tasks:
                s = s + str(task) + ":[" + str(tour.worst_case_start_time[task]) + "," + str(tour.quantile_finish_time[task]) + "], "
            s = s[:-2] + "}" + "formation:" + str(tour.formation_w_d)
            s += "skillcomp:" + str(tour.skill_comp) + "\n"
        
        return s[:-1]
    
    def get_hash(self):
        """Get hash as string containing basic information about the node.

        Returns
        -------
        s: str
            String-formatted hash that uniquely defines the node
        """

        s = ""
        for t_tilde in self.t_le:
            s = s + str(t_tilde) + "<=" + str(self.t_le[t_tilde]) + ","
        for t_tilde in self.t_gr:
            s = s + str(t_tilde) + ">=" + str(self.t_gr[t_tilde]) + ","
        for t_tilde in self.t_max_le:
            if self.t_max_le[t_tilde] != math.inf:
                s = s + "finish_" + str(t_tilde) + "<=" + str(self.t_max_le[t_tilde]) + ","
        for t_tilde in self.t_max_gr:
            if self.t_max_gr[t_tilde] != -1:
                s = s + "finish_" + str(t_tilde) + ">=" + str(self.t_max_gr[t_tilde]) + ","
        if self.forced_tours:
            s = s + "Forced:"
        for tour in self.forced_tours:
            s = s + "["
            for task in tour.tasks:
                s = s + str(task) + ":" + str(tour.worst_case_start_time[task]) + "," + str(tour.skill_comp) + "; "
            s = s[:-1] + "]"
        if self.forbidden_tours:
            s = s + "Forbidden:"
        for tour in self.forbidden_tours:
            s = s + "["
            for task in tour.tasks:
                s = s + str(task) + ":" + str(tour.worst_case_start_time[task]) + "," + str(tour.skill_comp) + "; "
            s = s[:-2] + "]"

        if self.forced_arcs:
            s = s + "Forced_arcs: ["
            for arc in self.forced_arcs:
                s = s + str(arc) + "; "
            s = s[:-2] + "]"
        if self.forbidden_arcs:
            s = s + "Forbidden_arcs: ["
            for arc in self.forbidden_arcs:
                s = s + str(arc) + "; "
            s = s[:-2] + "]"

        return s
    
    def update_lb(self, lb):
        """Update lower bound of node and its parent node.

        Parameters
        ----------
        lb: float
            New lower bound for current and (potentially) parent node
        """
        # 1. always update own lower bound
        self.lb = lb
        
        if self.parent != None:
            if self == self.parent.left_child:
                self.parent.left_child_lb = lb
            elif self == self.parent.right_child:
                self.parent.right_child_lb = lb
            else:
                raise Exception("Tried update from a non-child")

            # 2. update parent LB if both child nodes were already solved
            if self.parent.left_child_lb != None and self.parent.right_child_lb != None:
                self.parent.update_lb(min(self.parent.left_child_lb, self.parent.right_child_lb))

class GH_solution():

    def __init__(self, inst):
        """Class for solution objects. Contains information about a disaggregated-feasible integer solution, in particular
        tours alongside their leave times, formations, and task sequences.
        Also contains various pieces of information that can be used for analyzing solver behavior/performance:
        - reference to input inst object
        - indicator if current solution is optimal or suboptimal
        - runtime loggings for different parts of the algorithm
        - loggings for different statistics computed during the pricing step
        - loggings for different statistics computed during the branching tree search
        - statistics on branching decisions and tour counts
        Various other statistics are also included. This object is initialized once at the beginning of the algorithm.
        It is then continuously update after each node exploration. Whenever a disaggregated-feasible integer solution
        is found, its information is stored in this object.

        Parameters
        ----------
        inst: instance_loader.Instance
            Contains all necessary instance data read from input files.
        """

        self.inst = inst
        self.optimal = None
        self.heuristic = None
        self.infeasible_time = -1
        self.cannot_solve_time = -1
        self.explored_nodes = 0
        self.disaggr_feas_solutions = 0
        self.disaggr_infeas_solutions = 0
        self.feas_checks = 0
        self.feas_checks_passed = 0
        self.tot_labels = 0
        self.tot_dom_labels = 0
        self.tot_columns = 0
        self.time_setup_feas_check = 0
        self.time_feas_check = 0
        self.time_setup_master = 0
        self.time_master = 0
        self.time_setup_pricing = 0
        self.time_pricing = 0
        self.nr_iterations_cg = 0
        self.avg_forbidden_tours_per_node = 0
        self.max_forbidden_tours = 0
        self.count_branch_on_vehicles = 0
        self.count_branch_on_variable = 0
        self.count_branch_on_tasks = 0
        self.count_branch_on_arcs = 0
        self.count_gc_added = 0
        self.time_dominance_check = 0
        self.time_start_distr_calc = 0
        self.time_root = 0
        self.initial_label_cnt = 0
        self.only_best_task_cnt = {"Only best tasks": 0, "All tasks": 0}
        self.tw_viol_prob = {}
        self.tour_lengths = {}
        self.used_dmp = False
        self.explored_nodes_dmp = 0
        self.explored_nodes_amp = 0

    def get_min_value(self):
        """Returns the minimal possible cost of inst (even if the solution is infeasible).
        Trivial lower bound for objective function.
        Used to compute net objective values. A net objective value is equal to the objective value of a solution minus
        the result of this function. This net value serves as a more meaningful baseline for gap calculations. Otherwise,
        gaps can be artificially deflated by simply scaling the time instants by a constant.

        Returns
        -------
        val: float
            Minimum objective value that serves as a lower bound for the objective value of any feasible solution
            to the problem
        """

        val = sum([self.inst.earliest_finish[task] * self.inst.weights[task] for task in self.inst.tasks])
        return val
    
    def get_max_net_cost(self):
        """Returns the maximal net cost of the instance. Trivial upper bound for objective function

        Returns
        -------
        val: float
            Maximum objective value that serves as an upper bound for the net objective value of any feasible solution
            to the problem
        """
        val = 0
        for task in self.inst.tasks:
            val += (self.inst.latest_finish[task] - self.inst.earliest_finish[task]) * self.inst.weights[task]
        return val

    def add_optimal_solution(self, opt_sol, opt_val, opt_tours, opt_elapsed_time, root_lb):
        """Add optimal solution obtained during B&P to GH_solution object.

        Parameters
        ----------
        opt_sol: dict
            Maps tour indices to their optimal lambda value
        opt_val: float
            Optimal objective value
        opt_tours: list
            List of tours making up the optimal solution
        opt_elapsed_time: float
            Total runtime required by the algorithm to find the optimal solution
        root_lb: float
            Lower bound of the problem after solving the root node

        """

        # 1. store optimal tours and optimal objective
        optimal =  type('Optimal', (object,), {})()
        optimal.value = opt_val     # optimal objective value
        optimal.tours = []          # tours making up the optimal solution
        for i in range(len(opt_sol)):
            if opt_sol[i] > 1 - eps_global / 10:
                optimal.tours.append(opt_tours[i])
        optimal.elapsed_time = opt_elapsed_time
        # 2. store bounds
        optimal.net_value = opt_val - self.get_min_value()  # difference between current objective value and trivial lower bound (is not necessarily = 0 for a globally optimal solution of the problem)
        optimal.root_lb = root_lb
        optimal.net_root_lb = root_lb - self.get_min_value()
        # 3. compute gap
        if optimal.value - self.get_min_value() <= eps_gap:
            optimal.gap_at_root = 0
        else:
            try: # min_value might be 0
                optimal.gap_at_root = 1 - (optimal.root_lb - self.get_min_value()) / (optimal.value - self.get_min_value())
            except:
                optimal.gap_at_root = 0

        # 4. compute relative net value
        if self.get_max_net_cost() == 0:
            optimal.relative_net_value = 0
        else:
            optimal.relative_net_value = (opt_val - self.get_min_value()) / self.get_max_net_cost()
        self.optimal = optimal
        
    def add_heuristic_solution(self, heur_sol, heur_val, heur_tours, lower_bound, elapsed_time, root_lb):
        """Add heuristic solution obtained during B&P to GH_solution object.
        Similar to add_optimal_solution, but this function is called only when the heuristic master problem
        (which is solved after the maximum runtime was exceeded) returns a solution with a nonzero gap.

        Parameters
        ----------
        heur_sol: dict
            Maps tour indices to their lambda value
        heur_val: float
            Ojective value of heuristic solution
        heur_tours: list
            List of tours making up the heuristic solution
        lower_bound: float
            Best lower bound computed
        elapsed_time: float
            Total runtime required by the algorithm to find the optimal solution
        root_lb: float
            Lower bound of the problem after solving the root node

        """
        # 1. store heuristic tours and heuristic objective
        heuristic = type('Heuristic', (object,), {})()
        heuristic.value = heur_val
        heuristic.tours = []
        for i in range(len(heur_sol)):
            if heur_sol[i] > 1 - eps_global / 10:
                heuristic.tours.append(heur_tours[i])
        # 2. store bounds and runtime
        heuristic.lower_bound = lower_bound
        heuristic.net_upper_bound = heur_val - self.get_min_value()
        heuristic.net_lower_bound = lower_bound - self.get_min_value()
        heuristic.elapsed_time = elapsed_time
        heuristic.net_value = heur_val - self.get_min_value()
        heuristic.root_lb = root_lb
        heuristic.net_root_lb = root_lb - self.get_min_value()
        # 3. compute gap at root node
        if heur_val - self.get_min_value() <= eps_gap:
            heuristic.gap_at_root = 0
        else:
            heuristic.gap_at_root = 1 - (heuristic.root_lb - self.get_min_value()) / (heur_val - self.get_min_value())
        # 4. compute net value
        if self.get_max_net_cost() == 0:
            heuristic.relative_net_value = (heur_val - self.get_min_value()) / self.get_max_net_cost()
        else:
            heuristic.relative_net_value = (heur_val - self.get_min_value()) / self.get_max_net_cost()
        # 5. compute gap
        if heur_val - self.get_min_value() <= eps_gap: # can happen due to floating point errors
            heuristic.optimality_gap = 0.0
        else:
            heuristic.optimality_gap = 1 - (lower_bound - self.get_min_value()) / (heur_val - self.get_min_value())
        self.heuristic = heuristic


    def to_string(self):
        """Format solution information to string that can be printed. Contains all logged statistics, as well as a summary
        of all tours that make up the best disaggregated-feasible integer solution found.
        """

        tot_workers = 0
        for k in self.inst.workers:
            tot_workers += self.inst.workers[k]
        
        s = ""
        s+= "Number of tasks: " + str(len(self.inst.tasks)) + "\n"
        s+= "Actual number of instants: " + str(len(self.inst.instants)) + "\n"
        s+= "Total number of workers: " +  str(tot_workers) + "\n"
        for k in self.inst.workers:
            s += "Workers level "+ str(k) + ": " + str(self.inst.workers[k]) + "\n"
        s+= "Number of explored nodes: " + str(self.explored_nodes) + "\n"
        s+= "Number of disaggregated-infeasible solutions: " + str(self.disaggr_infeas_solutions) + "\n"
        s+= "Number of labels: " + str(self.tot_labels) + "\n"
        s+= "Number of dominated labels: " + str(self.tot_dom_labels) + "\n"
        s+= "Percentage of labels dominated: " + str(round(self.tot_dom_labels/self.tot_labels*100, 2)) + "%" + "\n"
        s+= "Number of feasible checks: " + str(self.feas_checks) + "\n"
        s+= "Number of feasible checks passed: " + str(self.feas_checks_passed) + "\n"
        
        s += "Number of columns: " + str(self.tot_columns) + "\n"
        s += "Total time master: " + str(self.time_master) + "\n"
        s += "Total time setup master: " + str(self.time_setup_master) + "\n"
        s += "Total time pricing: " + str(self.time_pricing) + "\n"
        s += "Total time setup pricing: " + str(self.time_setup_pricing) + "\n"
        s += "Total time feasibility check: " + str(self.time_feas_check) + "\n"
        s += "Total time setup feasibility check: " + str(self.time_setup_feas_check) + "\n"
        s += "Total time label dominance check: " + str(self.time_dominance_check) + "\n"
        s += "Total time start time distribution calculation: " + str(self.time_start_distr_calc) + "\n"
        s += "Number of iterations of cg: " + str(self.nr_iterations_cg) + "\n"
        s += "Average forbidden tours per node: " + str(self.avg_forbidden_tours_per_node) + "\n"
        s += "Max forbidden tour in a node: " + str(self.max_forbidden_tours) + "\n"
        s += "Nr branches on vehicles: " + str(self.count_branch_on_vehicles) + "\n"
        s += "Nr branches on variable: " + str(self.count_branch_on_variable) + "\n"
        s += "Nr branches on task finish times: " + str(self.count_branch_on_tasks) + "\n"
        s += "Nr branches on arcs: " + str(self.count_branch_on_arcs) + "\n"
        s += "No. of initial labels created: " + str(self.initial_label_cnt) + "\n"
        s += "No. of networks solved with only best tasks: " + str(self.only_best_task_cnt) + "\n"
        s += "TW violation probabilities: " + str(self.tw_viol_prob) + "\n"
        s += "Tour lengths frequencies: " + str(self.tour_lengths) + "\n"
        s += "\n"
        
        if self.optimal != None:
            value = math.trunc(self.optimal.value *100000000) / 100000000
            net_value = math.trunc(self.optimal.net_value *100000000) / 100000000
            s+= "Optimal solution found in " + str(self.optimal.elapsed_time) + " seconds\n"
            s+= "Optimal value: " + str(value) + "\n"
            s+= "Optimal net value: " + str(net_value) + "\n"
            s+= "Relative net value: " + str(self.optimal.relative_net_value) + "\n\n"
            root_gapPerCent = math.trunc(self.optimal.gap_at_root * 10000) / 100
            s += "Runtime root node: " + str(self.time_root) + "\n"
            s += "LB at root node: " + str(self.optimal.root_lb) + "\n"
            s += "Net LB at root node: " + str(self.optimal.net_root_lb) + "\n"
            s += "Gap in % at root node: " + str(root_gapPerCent) + "%\n\n"
            s+= "Tours:\n"
            for tour in self.optimal.tours:
                s+= tour.to_string() + "\n"
        
        elif self.heuristic != None:
            value = math.trunc(self.heuristic.value *100000000) / 100000000
            net_value = math.trunc(self.heuristic.net_value *100000000) / 100000000
            s+= "Heuristic solution found in " + str(self.heuristic.elapsed_time) + " seconds\n"
            s+= "Heuristic solution value: " + str(value) + "\n"
            s+= "Best lower bound: " + str(self.heuristic.lower_bound) + "\n"
            s+= "Heuristic solution net value: " + str(net_value) + "\n"
            s+= "Best net lower bound: " + str(self.heuristic.net_lower_bound) + "\n"
            gapPerCent = math.trunc(self.heuristic.optimality_gap * 10000) / 100
            s+= "Optimality gap: " + str(gapPerCent) + "%\n"
            s+= "Relative net value: " + str(self.heuristic.relative_net_value) + "\n\n"
            root_gapPerCent = math.trunc(self.heuristic.gap_at_root * 10000) / 100
            s += "Runtime root node: " + str(self.time_root) + "\n"
            s += "LB at root node: " + str(self.heuristic.root_lb) + "\n"
            s += "Net LB at root node: " + str(self.heuristic.net_root_lb) + "\n"
            s += "Gap in % at root node: " + str(root_gapPerCent) + "%\n\n"
            s+= "Tours:\n"
            for tour in self.heuristic.tours:
                s+= tour.to_string() + "\n"
        
        elif self.infeasible_time >= 0:
            s+= "Instance infeasible\n"
            s+= "Time needed to prove infeasibility: " + str(self.infeasible_time)
            
        else:
            s+= "Could not find any solution within the time limit\n"
            s+= "Stopped after: " + str(self.cannot_solve_time)
            
        return s
            


    def __eq__(self, other):

        """Check if two solutions are identical. Solutions are considered identical iff. they contain the same tours.
        Tours can be different GH_tour objects, but their content must match between the solutions.

        Parameters
        ----------
        other: GH_solution
            Solution to compare against

        Returns
        -------
        are_identical: bool
            True iff. self and other are the same solution
        """

        tours_hash_set = set()
        tours_hash_set_other = set()
        for tour in self.optimal.tours:
            tours_hash_set.add(tour.get_hash())
        for tour in other.optimal.tours:
            tours_hash_set_other.add(tour.get_hash())

        for tour in tours_hash_set:
            if tour not in tours_hash_set_other:
                return False
        for tour in tours_hash_set_other:
            if tour not in tours_hash_set:
                return False
        return True

    def to_dict(self):
        """Store data for all tours used by current solution sol in a dict, which can be saved as a .json file.

        Returns
        -------
        tours_dict: dict
            Maps tour indices to dictionaries that encompass all information necessary to uniquely identify a tour
        """
        tours_dict = {}

        it = 0
        if self.optimal:
            for tour in self.optimal.tours:
                it += 1
                tours_dict[it] = {"leave": tour.leave_time, "quantile_return": tour.quantile_return_time,
                                  "formation_w_d": tour.formation_w_d,
                                  "formation": tour.formation_id,
                                  "skill_comp": tour.skill_comp, "skill_comp_cnt": tour.skill_comp_cnt,
                                  "tasks": tour.tasks,
                                  "tw_viol_prob": tour.tw_viol_prob, "worst_case_start_time": tour.worst_case_start_time,
                                  "quantile_finish_time": tour.quantile_finish_time, "cost": tour.cost}
        elif self.heuristic:
            for tour in self.heuristic.tours:
                it += 1
                tours_dict[it] = {"leave": tour.leave_time, "quantile_return": tour.quantile_return_time,
                                  "formation_w_d": tour.formation_w_d,
                                  "formation": tour.formation_id,
                                  "skill_comp": tour.skill_comp, "skill_comp_cnt": tour.skill_comp_cnt,
                                  "tasks": tour.tasks,
                                  "tw_viol_prob": tour.tw_viol_prob, "worst_case_start_time": tour.worst_case_start_time,
                                  "quantile_finish_time": tour.quantile_finish_time}
        return tours_dict

