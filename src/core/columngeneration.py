"""
Column generation (CG) logic for the Branch-Price-and-Cut-and-Switch algorithm.

This module implements the LP relaxation solver used at each node of the branch-and-bound
tree. It coordinates the master problem and the pricing subproblem in the standard
column-generation loop, with optional Gomory cut separation at the root node.

Core procedure (get_CG_LB_no_heuristic):
  1. Master setup: builds the restricted master LP for the current B&B node, accounting
     for forced/forbidden tours, branching constraints on tour counts per time instant
     (t_gr / t_le), arc forcing/forbidding, and any previously found Gomory cuts.
  2. Pricing loop: iteratively solves the ESPPRC-based pricing subproblem (one per worker
     formation) to find negative-reduced-cost tour columns.
     - Optionally restricts extensions to the most promising task arcs first
       (solve_only_with_best_tasks) before falling back to all extensions.
     - New columns are added to the master and the LP is re-solved until no improving
       column is found.
  3. Gomory cut separation (root node only): once the pricing loop converges, checks for
     violated Chvátal-Gomory cuts via check_for_gomory_cuts_only_nonzero_light. If a
     violated cut is found it is added to the master and the pricing loop restarts.
     Repeats until no_gomory_cuts cuts have been added or no violated cut remains.
  4. Returns the optimal LP bound, solution, tours, and detailed runtime/label statistics
     to the caller (branchandprice.py).
"""

import time
import math
import copy
from src.master import master
from src.pricing import pricing
from src.core.utils import tour_in_set
from src.cuts.gomory_cuts import extend_gomory_cuts, check_for_gomory_cuts_only_nonzero_light
from gurobipy import GRB
from config.config import *



def get_CG_LB_no_heuristic(pricing_networks, node, disaggr_infeas_solutions, tl, elapsed_time, earliest_finish_sums,
                           no_gomory_cuts, cores_per_thread, yuan_approach, solve_only_with_best_tasks,
                           best_task_cnt):
    """Solve LP relaxation via column generation. Proceeds as follows:

    1. sets up master problem at current node
    2. Solve pricing problem: repeat until no negative columns are found
        2.1 if only_best_tasks = True: solve pricing problem only allowing extensions along arcs that connect tasks
            with the largest task rewards
        2.2 if only_best_tasks = False or 2.1 did not return a negative column: solve pricing problem allow all
            extensions
    3. if gomory cuts are desired: look for violated gomory cuts
        3.1 if a violated cut was found: add it to the problem and go to Step 2
        3.2 if no cut was found or sufficiently many cuts are already added to the master problem: go to Step 4
    4. return current solution as optimal solution to the LP relaxation of the master problem at the current node


    Parameters
    ----------
    pricing_networks: dict
        Maps formation IDs to their workers.pricing.graph.PricingNetwork object
    node: GH_node
        Current node in the branching tree
    disaggr_infeas_solutions: list
        List of disaggregated infeasible solutions found so far
    tl: float
        Time limit of the algorithm (in seconds)
    elapsed_time: float
        Time spent in BPC&S routine (in seconds) prior to current function call
    earliest_finish_sums: float
            Minimum objective value that serves as a lower bound for the objective value of any feasible solution
            to the problem (computed using GH_solution.get_min_value())
    no_gomory_cuts: int
        Maximum no. of gomory cuts to be added at the root node.
    cores_per_thread: int
        Number of cores to use by gurobi when solving the master problem
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
        branch_on_task_finish_times = False, and use_dmp = True
    solve_only_with_best_tasks: bool
        If set to True, the pricing problem is first solved only allowing extensions along arcs that connect tasks with
        very large task rewards. If set to False, the pricing problem is always solved allowing all extensions.
    best_task_cnt: int
            No. of incident tasks for each node that should be considered if only_best_tasks is set to True
    Returns
    -------
    cg_sol: dict
        Maps tour indices to their lambda value of optimal solution of the current LP relaxation
    cg_val: float
        Optimal objective value of the current LP relaxation
    cg_tours: list
        List of tours at the current node
    tot_labels: int
        Total no. of labels constructed in the current pricing step
    tot_dom_labels: int
        Total no. of dominated labels in the current pricing step
    total_time: float
        Total runtime spent to solve the current node
    tot_master_time: float
        Total runtime spent to solve the master problem at the current node
    tot_pricing_time: float
        Total runtime spent to solve the pricing subproblems at the current node
    tot_dominance_time: float
        Total time spent to check label dominance
    tot_start_time_distr_time: float
        Total time spent to compute start time distributions
    master_model.master_setup_time: float
        Total runtime spent to setup the master problem at the current node
    nr_iterations_cg: int
        No. of iterations of column generation at the current node
    total_initial_label_cnt: int
        Total no. of initial labels constructed
    only_best_task_cnt: dict
        Counts the number of times the pricing subproblem was found considering only the most promising extensions
        (key True) or considering all extensions (key False), respectively
    """

    # 1. initialize algorithm
    # 1.1 define some results logging parameters
    total_initial_label_cnt = 0
    total_best_task_cnt = {True: 0, False: 0}       # counts the no. of times the SPPRC has been solved using all or only the best tasks
    tour_hashes = []
    tot_labels = 0  # total labels
    tot_dom_labels = 0  # total dominated labels
    for tour in node.tours:
        tour_hashes.append(tour.get_hash())
    # runtime of master/pricing problem and no. of column generation calls
    start_time = time.time()
    runtime_exceeded = False # set to True once current CG step consumed more runtime than what is available for BPC&S
    tot_master_time = 0 # total runtime of master problem
    tot_pricing_time = 0    # total runtime of pricing
    nr_iterations_cg = 1    # no. of column generation calls
    tot_dominance_time = 0  # total time spent checking labels for dominance
    tot_start_distr_calc_time = 0       # total time spent calculating start time distribution

    # 1.2 compute cost of forced tours
    forced_cost = get_forced_tours_cost(node.forced_tours)   # cost of forced tours, i.e. tours that must be selected

    # 1.3 get all unforced_tasks and workers_kt (available workers with/without dominance for each skill level k and
    # time instant t, after removing workers that are occupied with forced tours)
    unforced_tasks = copy.copy(node.inst.tasks) # copy all tasks
    forced_tasks = set()
    # 1.3.1 if skill compositions are to be considered: also get all possible skill compositions
    if node.solve_as_dmp:    # no downgrading considered if all_skill_comps is True
        workers_kt = {}  # workers without downgrading for each instant t
        for k in node.inst.skill_levels:
            workers_kt[k] = {}
            for t in node.inst.instants:
                workers_kt[k][t] = node.inst.workers[k]
        for forced_tour in node.forced_tours:  # remove forced tasks from unforced_tasks and subtract required workers from all available workers
            for forced_task in forced_tour.tasks:
                unforced_tasks.remove(forced_task)
                forced_tasks.add(forced_task)
            for k in forced_tour.skill_comp_cnt:
                for t in range(forced_tour.leave_time, forced_tour.quantile_return_time):
                    workers_kt[k][t] -= forced_tour.skill_comp_cnt[k]
        # if node is root node: also keep track of workers with downgrading which are used to find violated gomory cuts
        # note: there are no forced tours at the root node, hence we do not need to remove workers occupied with forced
        # tours from the available workforce with level >= k (i.e. node.inst.workers_w_d[k])
        if node.is_root:
            workers_kt_gc = {}
            for k in node.inst.skill_levels:
                workers_kt_gc[k] = {}
                for t in node.inst.instants:
                    workers_kt_gc[k][t] = node.inst.workers_w_d[k]
            if len(node.forced_tours) > 0:
                raise Exception(f"Found {len(node.forced_tours)} forced tours at the root node DMP formulation.")
    # 1.3.2 else: only compute worker availability
    else:
        workers_kt = {} # workers with downgrading for each instant t
        for k in node.inst.skill_levels:
            workers_kt[k] = {}
            for t in node.inst.instants:
                workers_kt[k][t] = node.inst.workers_w_d[k]
        for forced_tour in node.forced_tours:   # remove forced tasks from unforced_tasks and subtract required workers from all available workers
            for forced_task in forced_tour.tasks:
                unforced_tasks.remove(forced_task)
                forced_tasks.add(forced_task)
            for k in forced_tour.formation_w_d:
                for t in range(forced_tour.leave_time, forced_tour.quantile_return_time):
                    workers_kt[k][t] -= forced_tour.formation_w_d[k]



    # 1.4 get infeasible aggregated solutions and restriction on no. of tours (derived from branching / failed
    # feasibility checks)
    t_gr, t_le = get_branching_constr_after_forcing(node)
    infeas_aggr_sol_sets, infeas_aggr_sol_superset = get_infeas_aggr_sol_sets(node, disaggr_infeas_solutions)


    # 1.5 create list forced_arcs_excl_tours that only contain the forced arcs which are not already included in a
    # forced tour
    arcs_forced_by_tours = []       # contains all arcs that are already enforced by forced tours
    for tour in node.forced_tours:
        sequence = ["source"] + tour.tasks + ["sink"]
        for idx in range(len(sequence)-2):
            arcs_forced_by_tours.append((sequence[idx], sequence[idx+1], tour.quantile_finish_time[sequence[idx+1]]))
        # sink arc treated separately, as tour.finish_time does not contain an item for key 'sink'
        arcs_forced_by_tours.append((sequence[idx+1], sequence[idx + 2], tour.quantile_return_time))
    forced_arcs_excl_tours = []
    for arc in node.forced_arcs:
        if arc not in arcs_forced_by_tours:
            forced_arcs_excl_tours.append(arc)


    # 1.6 create and solve master problem
    master_model = master.Master_model(unforced_tasks, node.tours, list(range(len(node.tours))), workers_kt, t_gr,
                                       t_le, forced_arcs_excl_tours, infeas_aggr_sol_sets, GRB.CONTINUOUS,
                                       node.solve_as_dmp, node.gomory_cuts_lhs, node.gomory_cuts_rhs,
                                       cores_per_thread)


    # 2. find negative columns by solving the pricing problem
    # 2.1 create pricing network for each profile (worker formation)
    for formation_id in pricing_networks:
        pricing_networks[formation_id].remove_forced_task(forced_tasks)

    # 2.2 concatenate forbidden tours and infeasible aggregated solutions
    forbidden_tours = set(node.forbidden_tours)
    forbidden_tours = forbidden_tours.union(infeas_aggr_sol_superset)

    found_violated_cut = True       # set to False once no more negative columns have been found and no gomory cuts are violated
    gomory_cut_lhs = None           # initially set to None, usually contains columns and coefficients for last gomory cut found
    gomory_cut_rhs = None
    gomory_cuts_added = 0

    while not runtime_exceeded and found_violated_cut:
        pricing_iter_cnt = 0 # tracks no. of times that all pricing subproblems have been solved in current fctn call (cycle breaker)

        # 2.3 add gomory cut if one has been found
        if gomory_cut_lhs is not None:
            gomory_cuts_added += 1
            master_model.add_gomory_cut(gomory_cut_lhs, gomory_cut_rhs)

        # 2.4 solve master problem and get duals
        opt_sol, opt_val, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, time_master = master_model.optimize_master(return_duals = True)
        current_sol_tours = [node.tours[idx] for idx in opt_sol if opt_sol[idx] > eps_global]
        tot_master_time += time_master

        # 2.5 if master problem is infeasible: return
        if not opt_sol:
            total_time = time.time() - start_time
            return {}, math.inf, [], 0, 0, total_time, tot_master_time, tot_pricing_time, tot_dominance_time, tot_start_distr_calc_time, \
                master_model.master_setup_time, nr_iterations_cg, total_initial_label_cnt, total_best_task_cnt

        print(f"Current value of master problem: {opt_val + forced_cost} (net {opt_val + forced_cost - earliest_finish_sums})")
        # 2.6 solve pricing problem
        neg_tours = []
        # 2.6.1 if desired: first solve pricing problem only allowing the most promising extensions
        if solve_only_with_best_tasks:
            only_best_tasks = True
            (neg_tours, count_labels, count_dom_labels, time_pricing, time_dominance, time_start_distr_calculation,
             initial_label_cnt) = pricing.solve_pricing(node.inst, pricing_networks, mu, delta, rho_gr, rho_le, psi, zeta_le,
                                                        zeta_gr, forbidden_tours, only_best_tasks, best_task_cnt, node.t_max_le,
                                                        node.t_max_gr, node.solve_as_dmp, node, current_sol_tours,
                                                        yuan_approach)
            total_initial_label_cnt += initial_label_cnt
            total_best_task_cnt[True] += 1
        # 2.6.2 else or if no columns were found: solve pricing problem allowing all extensions
        if neg_tours == []:
            only_best_tasks = False
            (neg_tours, count_labels, count_dom_labels, time_pricing, time_dominance, time_start_distr_calculation,
             initial_label_cnt) = pricing.solve_pricing(node.inst, pricing_networks, mu, delta, rho_gr, rho_le,
                                                        psi, zeta_le, zeta_gr, forbidden_tours, only_best_tasks, best_task_cnt,
                                                        node.t_max_le, node.t_max_gr, node.solve_as_dmp,
                                                        node, current_sol_tours, yuan_approach)
            total_initial_label_cnt += initial_label_cnt
            total_best_task_cnt[False] += 1

        # 2.6.3 log some runtimes and label counts
        tot_pricing_time += time_pricing
        tot_dominance_time += time_dominance
        tot_start_distr_calc_time += time_start_distr_calculation
        tot_labels += count_labels
        tot_dom_labels += count_dom_labels

        # 2.7 repeat until no more negative columns are found
        while not runtime_exceeded and neg_tours:
            # 2.7.1 cycle breaker
            # Note: this can theoretically happen when reduced cost calculation is subject to (significant) numerical
            # errors. in our tests it did not happen, but we still keep it as a sanity check
            if len(node.gomory_cuts_lhs) > 0:
                pricing_iter_cnt += 1
                if pricing_iter_cnt > 100:
                    raise Exception("Pricing probably started cycling")


            # 2.7.2 if BPC&S runtime is exceeded: break and return
            nr_iterations_cg += 1
            total_time = time.time() - start_time
            if total_time + elapsed_time > elapsed_time_buffer_factor * tl: # add slight buffer to allow algo. to finish solving the current node
                runtime_exceeded = True

            # 2.7.3 add columns and include them into existing gomory cuts
            node.gomory_cuts_lhs = extend_gomory_cuts(neg_tours, node.gomory_cuts_lhs,  node.u_task, node.u_kt,
                                                      node.tours)
            node.tours.extend(neg_tours)
            infeas_aggr_sol_sets, infeas_aggr_sol_superset = get_infeas_aggr_sol_sets(node, disaggr_infeas_solutions)

            # 2.7.4 add columns to master problem and re-solve
            master_model.add_tours(neg_tours, node.solve_as_dmp, node.gomory_cuts_lhs)
            opt_sol, opt_val, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, time_master = master_model.optimize_master(return_duals = True)
            current_sol_tours = [node.tours[idx] for idx in opt_sol if opt_sol[idx] > eps_global]
            tot_master_time += time_master
            print(f"Current value of master problem: {opt_val + forced_cost} (net {opt_val + forced_cost - earliest_finish_sums})")

            # 2.7.5 get forbidden tours
            forbidden_tours = set(node.forbidden_tours)
            forbidden_tours = forbidden_tours.union(infeas_aggr_sol_superset)

            # 2.7.6 if desired: first solve pricing problem only allowing the most promising extensions
            if solve_only_with_best_tasks:
                only_best_tasks = True
                (neg_tours, count_labels, count_dom_labels, time_pricing, time_dominance, time_start_distr_calculation,
                 initial_label_cnt)= pricing.solve_pricing(node.inst, pricing_networks, mu, delta, rho_gr, rho_le,
                                                           psi, zeta_le, zeta_gr, forbidden_tours, only_best_tasks,
                                                           best_task_cnt, node.t_max_le, node.t_max_gr, node.solve_as_dmp,
                                                           node, current_sol_tours, yuan_approach)
                total_initial_label_cnt += initial_label_cnt
                total_best_task_cnt[True] += 1

            # 2.7.7 else or if no columns were found: solve pricing problem allowing all extensions
            if neg_tours == []:
                only_best_tasks = False
                (neg_tours, count_labels, count_dom_labels, time_pricing, time_dominance, time_start_distr_calculation,
                 initial_label_cnt) = pricing.solve_pricing(node.inst, pricing_networks, mu, delta, rho_gr,
                                                            rho_le, psi, zeta_le, zeta_gr, forbidden_tours, only_best_tasks,
                                                            best_task_cnt, node.t_max_le, node.t_max_gr,
                                                            node.solve_as_dmp, node, current_sol_tours,
                                                            yuan_approach)
                total_initial_label_cnt += initial_label_cnt
                total_best_task_cnt[False] += 1

            # 2.7.8 log some runtimes and label counts
            tot_pricing_time += time_pricing
            tot_dominance_time += time_dominance
            tot_start_distr_calc_time += time_start_distr_calculation
            tot_labels += count_labels
            tot_dom_labels += count_dom_labels

        # 2.8 skip if maximum no. of GCs to be added is reached
        if gomory_cuts_added == no_gomory_cuts or yuan_approach:
            break

        # 2.9 check for violated gomory cuts
        tours_opt_sol = {}
        tours_to_idxs = {}
        idxs_to_tours = {}
        if (node.is_root) and len(node.gomory_cuts_lhs) < no_gomory_cuts:
            for i in opt_sol:
                if node.contains_fake_tour and i == 0:      # skip fake tour for gomory cuts
                    continue
                if opt_sol[i] > eps_global:
                    tours_opt_sol[node.tours[i]] = opt_sol[i]
                    tours_to_idxs[node.tours[i]] = i
                    idxs_to_tours[i] = node.tours[i]
            # 2.9.1 if root node is DMP node: look for violated cuts considering workers_kt_gc (i.e. workers w/ downgrading)
            # rather than with workers_kt (i.e. workers without downgrading)
            # Note: in Hagn et al.(2026), the root is always solved with the AMP
            if node.solve_as_dmp:
                found_violated_cut, opt_alpha, opt_alpha_zero, u_task, u_kt = check_for_gomory_cuts_only_nonzero_light(node,
                                                                        opt_sol, workers_kt_gc)
            else:
                found_violated_cut, opt_alpha, opt_alpha_zero, u_task, u_kt = check_for_gomory_cuts_only_nonzero_light(node,
                                                                        opt_sol, workers_kt)


            # 2.9.2 if violated gomory cut has been found: add it to the master problem and re-solve
            if found_violated_cut:
                cut_lhs = {}
                for i in opt_sol:
                    if node.contains_fake_tour and i == 0:  # skip fake tour for gomory cuts
                        continue
                    tour = node.tours[i]
                    if opt_sol[i] > eps_global:
                        if abs(opt_alpha[i]) > eps_global:
                            cut_lhs[i] = opt_alpha[i]
                    else:
                        alpha_tour = sum([u_task[task] for task in tour.tasks])
                        # create list of tuples (k,t) that are in u_kt, i.e. their values are nontrivial
                        entries_u_kt = [(k,t) for (k,t) in u_kt if t >= tour.leave_time and t < tour.quantile_return_time and k in tour.formation_w_d]
                        alpha_tour += sum([u_kt[(k,t)]*tour.formation_w_d[k] for (k,t) in entries_u_kt])
                        alpha_tour = math.floor(alpha_tour)
                        if abs(alpha_tour) > eps_global:
                            cut_lhs[i] = alpha_tour
                node.gomory_cuts_lhs.append(cut_lhs)
                node.gomory_cuts_rhs.append(opt_alpha_zero)
                node.u_task.append(u_task)
                node.u_kt.append(u_kt)
                gomory_cut_lhs = cut_lhs
                gomory_cut_rhs = opt_alpha_zero
                print("Added gomory cut")

        # 2.10 if no cut has been found: set parameter to False and break out of loop
        else:
            found_violated_cut = False



    total_time = time.time() - start_time
    if not runtime_exceeded:
        print(f"Found LB (from LP relaxation) of value {opt_val + forced_cost} "
              f"(net {opt_val + forced_cost - earliest_finish_sums}) in {total_time} s")
    else:
        print(f"Terminating node: BPC&S runtime exceeded. Continuing with solving the heuristic master problem...")

    # 3. parse solution
    cg_val = opt_val + forced_cost
    cg_sol = {}
    cg_tours = []
    for i in range(0, len(node.tours)):
        cg_sol[i] = opt_sol[i]
        cg_tours.append(node.tours[i])
    for i in range(len(node.tours), len(node.tours) + len(node.forced_tours)):
        cg_sol[i] = 1.0
        cg_tours.append(node.forced_tours[i-len(node.tours)])

    # 4. restore removed tasks in network
    for formation_id in pricing_networks:
        pricing_networks[formation_id].restore_removed_tasks()

    print(f"No. of gomory cuts at node: {len(node.gomory_cuts_rhs)}")

    return (cg_sol, cg_val, cg_tours, tot_labels, tot_dom_labels, total_time, tot_master_time, tot_pricing_time,
            tot_dominance_time, tot_start_distr_calc_time, master_model.master_setup_time, nr_iterations_cg,
            total_initial_label_cnt, total_best_task_cnt, runtime_exceeded)





def get_forced_tours_cost(forced_tours):
    """Calculate cost of all forced tours.

    Parameters
    ----------
    forced_tours: list
        List of forced tours (GH_tour objects)

    Returns
    -------
    cost: float
        Total cost (objective value) of all forced tours
    """

    cost = 0
    for tour in forced_tours:
        cost += tour.cost
    return cost

def get_branching_constr_after_forcing(node):
    """Calculate maximum and minimum no. of tours at a specific time, which are stored in node.t_le[t] and node.t_gr[t]
    for time instants t, after considering forced tours.
    Note: t_le and t_gr contain bounds on the number of tours active at a certain time. Their value is not automatically
    updated to account for forced tours. Therefore, we always have to adjust their value before calling the pricing
    routine.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    Returns
    -------
    t_gr: dict
        Maps time instants to the minimum number of active tours (yielded by branching on tour counts).
        Value is adjusted by the number of forced tours.
    t_le: dict
        Similar to t_gr, but corresponds to <= constraints (i.e., maximum number of active tours at a time instant).
        Value is adjusted by the number of forced tours.

    """

    t_gr = {}
    t_le = {}
    # 1. t_gr: reduce tour count by the number of forced tours
    for t in node.t_gr:
        l = node.t_gr[t]
        for forced_tour in node.forced_tours:
            if t in list(range(forced_tour.leave_time, forced_tour.quantile_return_time)):
                l -= 1
        t_gr[t] = l
    # 2. t_le: reduce tour count by the number of forced tours
    for t in node.t_le:
        l = node.t_le[t]
        for forced_tour in node.forced_tours:
            if t in list(range(forced_tour.leave_time, forced_tour.quantile_return_time)):
                l -= 1
        t_le[t] = l        
    
    return t_gr, t_le

def get_infeas_aggr_sol_sets(node, disaggr_infeas_solutions):
    """Create the sets of indices corresponding to the disaggr_infeas_solutions and a set (superset) of all tours
    in disaggr_infeas_solutions that are also in node.tours.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    disaggr_infeas_solutions: list
        List of disaggregated infeasible solutions found so far

    Returns
    -------
    sets: list
        List of lists containing all tours in each infeasible aggregated solution that are also in the current set
        of tours
    superset: list
        List of all tours in an infeasible aggregated solution that are also in node.tours (concatenation of
        objects in sets)
    """

    sets = []   # list of lists
    superset = set()    # list derived by concatenating all objects in sets
    for aggr_sol in disaggr_infeas_solutions:
        ias_set = []
        for tour in aggr_sol:
            for i in range(len(node.tours)):
                found = False
                if tour_in_set(node.forced_tours, tour):    # if tour is forced at current node: set found to True
                    found = True
                    break
                elif tour.get_hash() == node.tours[i].get_hash():   # else: check if tour exists at current node
                    ias_set.append(i)   # add i to list of nodes contained in both nodes.tours and aggr_sol
                    found = True
                    break
            if not found:
                ias_set = None
                break
        if ias_set != None:
            sets.append(ias_set)
            superset |= set(aggr_sol)
    return sets, superset


class CG_timeout_exception(Exception):

    def __init__(self, tot_labels, tot_dom_labels):
        """Exception raised when reaching the time limit for exact solving.

        Parameters
        ----------
        tot_labels: int
            Total no. of labels constructed in the current pricing step
        tot_dom_labels: int
            Total no. of dominated labels in the current pricing step
        """

        self.tot_labels = tot_labels
        self.tot_dom_labels = tot_dom_labels
