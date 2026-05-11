"""This script contains smaller utility functions used by branchandprice.py and columngeneration.py.
"""
from math import floor
from config.config import *

def all_weights_int(inst):
    """Check if all weights of given instance inst are integer

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    all_integer: bool
        True iff. all weights are integer
    """
    for task in inst.weights:
        w = inst.weights[task]
        if w - floor(w) != 0:
            return False
    return True

def is_sol_integer(node_sol):
    """Check if given solution is integer, i.e. feasible for the IP.

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value

    Returns
    -------
    is_integer: bool
        True iff. current solution is integer
    """

    for i in node_sol:
        if node_sol[i] > eps_global and node_sol[i] < 1 - eps_global:
            return False
    return True


def tour_in_set(tour_set, new_tour):
    """Check if new_tour is in tour_set.

    Parameters
    ----------
    tour_set: list
        List of GH_tour objects
    new_tour: GH_tour
        Tour to check

    Returns
    -------
    is_in_set: bool
        True iff. new_tour is in tour_set
    """

    found = False
    for tour in tour_set:
        if tour.get_hash() == new_tour.get_hash():
            found = True
            break
    return found

def get_default_skill_comps(tours, inst):
    """Get a default skill composition for all tours in tour. Default skill compositions act as placeholders until
    a skill composition is computed by solving the DMP at the current node, or by applying the feasibility check.

    Parameters
    ----------
    tours: list
        List of GH_tour objects
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    tours: list
        List of GH_tour objects
    """
    for tour in tours:
        skill_comp = {}
        skill_comp_cnt = {}
        formation_id = tour.formation_id
        formation_dict = inst.formations[formation_id]
        for k in inst.workers:
            if k in inst.formations[formation_id]:
                skill_comp_cnt[k] = inst.formations[formation_id][k]
            else:
                skill_comp_cnt[k] = 0
            skill_comp[k] = {}
            for kk in inst.workers:
                if kk < k:
                   continue
                if kk == k:
                    if k in formation_dict:
                        skill_comp[k][kk] = formation_dict[k]
                    else:   # skill level not needed in formation
                        skill_comp[k][kk] = 0
                else:
                    skill_comp[k][kk] = 0
        tour.skill_comp = skill_comp
        tour.skill_comp_cnt = skill_comp_cnt

    return tours


def store_solution_and_stats(solution, node, root, disaggr_infeas_solutions, node_sol, node_val, nodes_count, total_time,
                             master_setup_time, time_master, time_pricing, nr_iterations_cg, initial_label_cnt,
                             count_labels, count_dom_labels, time_dominance, time_start_distr_calc, only_best_task_cnt,
                             cum_forbidden_tours, runtime_exceeded):
    """After solving a node in the branching tree, stores all relevant information on solution progress, as well as the
    found solution.

    Parameters
    ----------
    solution: GH_solution
        GH_solution object
    node: GH_node
        Current node in the branching tree
    root: GH_node
        Root node of the brnaching tree
    disaggr_infeas_solutions: list
        List of disaggregated-infeasible solutions found so far
    node_sol: dict
        Maps tour indices to their lambda values in the current node
    node_val: float
        Objective value of current node
    nodes_count: int
        Number of branching tree nodes explored so far
    total_time: float
        Total runtime so far
    master_setup_time: float
        Total time spent to set up and adjust the master problem
    time_master: float
        Total master problem solving time
    time_pricing: float
        Total time spent solving the pricing subproblems
    nr_iterations_cg: int
        No. of iterations of column generation at the current node
    initial_label_cnt: int
        Total no. of initial single-task labels creating during pricing subroutine
    count_labels: int
        Total no. of labels creating during pricing subroutine
    count_dom_labels: int
        Total no. of labels dominated during the subroutine
    time_dominance: float
        Total time spent to check label dominance
    time_start_distr_calc: float
        Total time spent to compute start time distributions
    only_best_task_cnt: dict
        Counts the number of times the pricing subproblem was found considering only the most promising extensions (key True)
        or considering all extensions (key False), respectively
    cum_forbidden_tours: int
        Total no. of tours that were forbidden at at least one node
    runtime_exceeded: bool
        Indicates if BPC&S runtime was exceeded during the last pricing step. If this is the case, node_sol is not
        the optimal solution of the LP relaxation at node, and can thus not be used for lower bound computation
    Returns
    -------
    solution: GH_solution
        GH_solution object
    node: GH_node
        Current node in the branching tree
    nodes_count: int
        Number of branching tree nodes explored so far
    cum_forbidden_tours: int
        Total no. of tours that were forbidden at at least one node

    """

    # 2.2.3 track no. of nodes solved via the DMP / AMP formulations
    if node.solve_as_dmp:
        solution.used_dmp = True
        solution.explored_nodes_dmp += 1
    else:
        solution.explored_nodes_amp += 1
    if node.is_root:  # store no. of gomory cuts created
        solution.count_gc_added = len(node.gomory_cuts_lhs)
    # 2.2.4 store found solution in GH_node object
    node.iter = nodes_count
    node.sol = node_sol
    node.val = node_val

    if node == root:  # store relaxation LB at root node
        if runtime_exceeded:
            solution.root_lb = solution.get_min_value()
        else:
            solution.root_lb = node.val
        solution.time_root = total_time

    # 1.2.2 store several metrics regarding runtimes and problem size
    solution.time_setup_master += master_setup_time
    solution.time_master += time_master
    solution.time_pricing += time_pricing
    solution.nr_iterations_cg += nr_iterations_cg
    # label and node counts
    solution.initial_label_cnt += initial_label_cnt  # no. of initial labels created
    solution.tot_labels += count_labels
    solution.tot_dom_labels += count_dom_labels
    solution.time_dominance_check += time_dominance
    solution.time_start_distr_calc += time_start_distr_calc
    nodes_count += 1
    solution.explored_nodes += 1
    # count if networks have been solved with smallest_mus = True or False
    solution.only_best_task_cnt["Only best tasks"] += only_best_task_cnt[True]
    solution.only_best_task_cnt["All tasks"] += only_best_task_cnt[False]

    # compare no. of forbidden tours at current node to maximum no. of forbidden tours at all nodes already explored
    n_forb_tours = len(node.forbidden_tours) + len(disaggr_infeas_solutions)  # no. of forbidden tours
    cum_forbidden_tours += n_forb_tours
    if n_forb_tours > solution.max_forbidden_tours:  # max_forbidden_tours: maximum no. of forbidden tours at any explored node
        solution.max_forbidden_tours = n_forb_tours

    return solution, node, nodes_count, cum_forbidden_tours

def get_setup_pricing(pricing_networks):
    """Calculate total time spent on setting up/adjusting pricing networks.

    Parameters
    ----------
    pricing_networks: dict
        Maps formation IDs to their src.pricing.graph.PricingNetwork object

    Returns
    -------
    tot_time: float
        Total setup time for all pricing networks
    """

    tot_time = 0
    for formation_id in pricing_networks:
        tot_time += pricing_networks[formation_id].time_setup
    return tot_time
