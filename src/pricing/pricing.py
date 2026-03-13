"""This script contains the master function solve_pricing, which starts the pricing procedure.
For the sake of readability, most subroutines required during the pricing step are fractionalized into several scripts.
All necessary scripts (except for higher-level functions that are also needed by other parts of the algorithm) are
located in this same directory (workers.pricing).
"""

from src.pricing.heuristic_columns import find_columns_heuristics
import time
import numpy as np
from config.config import *

def solve_pricing(inst, pricing_networks, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, forbidden_tours,
                  only_best_tasks, best_task_cnt, t_max_le, t_max_gr, solve_as_dmp, node, current_sol_tours,
                  yuan_approach):
    """Solve the pricing problem by solving the elementary shortest path problem with resource constraints (ESPPRC)
    on all provided pricing networks. Iteratively solves each pricing network. When negative columns are found,
    the column with the most negative reduced cost is added to the column pool and the master problem is resolved.
    The process terminates once no negative columns can be found in any of the pricing networks.
    Note: when the approach by Yuan et al. (2015) is to be applied, i.e. when yuan_approach = True, all columns with
    negative reduced costs are added to the column pool in each iteration.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    pricing_networks: list
        List of workers.pricing.graph.PricingNetwork objects
    mu: dict
        Maps task execution constraints to their dual values
    delta: dict
        Maps workforce constraints to their dual values
    rho_gr: dict
        Maps >= tour count constraints to their dual values
    rho_le: dict
        Maps >= tour count constraints to their dual values
    psi: dict
        Maps gomory cuts to their dual values
    zeta_le: dict
        Maps <= 1 arc usage constraints to their dual values
    zeta_gr: dict
        Maps >= 1 arc usage constraints to their dual values
    forbidden_tours: list
        List of tours that are forbidden by branching on tours
    only_best_tasks: bool
        Indicates if each pricing network should initially only be solved using the tasks with the most negative
        dual value
    best_task_cnt: int
        No. of incident tasks for each node that should be considered if only_best_tasks is set to True
    t_max_le: dict
        Maps tasks to their latest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
    t_max_gr: dict
        Maps tasks to their earliest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).
    node: GH_node
        Current node in the branching tree
    current_sol_tours: list
        List of GH_tours at the current node that were selected in the current LP relaxation's solution (i.e., whose
         current objective value is > 0)
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
        branch_on_task_finish_times = False, and use_dmp = True.

    Returns
    -------
    neg_tours: list
        List of tours with negative reduced costs that were found
    tot_labels: int
        Total number of labels constructed
    solve_time: float
        Total time spent to solve the pricing problem
    tot_dominance_time: float
        Total time spent to check label dominance
    tot_start_time_distr_time: float
        Total time spent to compute start time distributions
    total_initial_label_cnt: int
        Total no. of initial labels constructed
    """
    # 1. Intialization
    # 1.1 results logging for label and column counts
    tot_labels = 0
    tot_dom_labels = 0 # only dominated labels
    sink_labels = 0
    total_initial_label_cnt = 0

    # 1.2 runtime logging
    tot_dominance_time = 0
    solve_time = 0
    tot_start_distr_time = 0        # time spent calculating start time distributions

    # 1.3 get forbidden tours for each formation
    forbidden_tours_per_formation = {}
    for formation_id in inst.formations_w_d:
        forbidden_tours_per_formation[formation_id] = []
        for forb_tour in forbidden_tours:
            if inst.formations_w_d[formation_id] == forb_tour.formation_w_d:
                forbidden_tours_per_formation[formation_id].append(forb_tour)


    # 2. compute columns
    neg_tours = []

    # 2.1 if approach by Yuan et al. (2015) is used: first call heuristics for all pricing networks
    if yuan_approach:
        neg_tours = find_columns_heuristics(pricing_networks, forbidden_tours_per_formation, mu,
                                                                delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, t_max_le,
                                                                t_max_gr, solve_as_dmp, node, yuan_approach,
                                                                current_sol_tours, inst)

    # 2.2 if approach by Yuan et al. (2015) not used or no negative heuristic columns are found: solve pricing networks exactly
    if neg_tours == []:
        # solve pricing problem for each formation
        for formation_id in inst.formations:
            network = pricing_networks[formation_id]
            network.set_resources_consumption(inst, forbidden_tours_per_formation[formation_id])
            network.task_resources_consd = []        # reset tasks for which resources will be considered at the beginning of each new iteration

            # 2.2.1 get all tours that are part of the basis solution and use the corresponding formation
            current_sol_tours_per_formation = [tour for tour in current_sol_tours if tour.formation_id == formation_id]

            # 2.2.2 solve ESPPRC
            start_time = time.time()
            (min_path, min_cost, min_labels, count_labels, count_dom, tour_costs, sink_labels_network,
             no_initial_labels) =  network.get_sprc(mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, only_best_tasks,
                                                    best_task_cnt, t_max_le, t_max_gr, solve_as_dmp, node, yuan_approach)

            # 2.2.3 log some results stats
            total_initial_label_cnt += no_initial_labels
            sink_labels += sink_labels_network
            solve_time += time.time() - start_time
            tot_labels += count_labels
            tot_dom_labels += count_dom
            tot_dominance_time += network.check_dominance_time
            tot_start_distr_time += network.calculate_start_distr_time
            network.check_dominance_time = 0    # reset dominance check runtime, since pricing networks are reused in every iteration
            network.calculate_start_distr_time = 0  # same for start time distribution calculation runtime



            # 2.2.4 add tour(s) with negative reduced cost to list of negative tours
            # 2.2.4.1 if approach by Yuan et al. (2015) is used: add all tours with negative reduced cost if desired
            if yuan_approach and min_cost < np.inf:
                # sanity check: check if any forbidden tours have been re-generated
                for tour in forbidden_tours_per_formation[formation_id]:
                    for idx in range(len(min_path)):
                        path = min_path[idx]
                        label = min_labels[idx]
                        if path == tour.tasks and label.start_time_from_depot == tour.leave_time:
                            if label.min_skill_comp == tour.skill_comp:
                                raise Exception(f"ERROR: re-generated forbidden tour {path}")
                for label in min_labels:
                    if label.cost < eps_col_neg * 10:
                        new_tour = network.build_tour(label, inst)
                        neg_tours.append(new_tour)

            # 2.2.4.2 else: only add tour with most negative reduced cost to list of negative tours (if it exists)
            else:
                if min_cost < eps_col_neg * 10:
                    new_tour = network.build_tour(min_labels[0], inst)
                    neg_tours.append(new_tour)


    return neg_tours, tot_labels, tot_dom_labels, solve_time, tot_dominance_time, tot_start_distr_time, total_initial_label_cnt
