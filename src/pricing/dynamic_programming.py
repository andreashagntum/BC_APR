"""
Dynamic programming (labeling algorithm) for the stochastic pricing subproblem.

This module solves the Elementary Shortest Path Problem with Resource Constraints (ESPPRC)
on each formation's pricing network using a forward labeling algorithm. Each label tracks a
partial tour (depot -> task sequence) together with all resources needed for dominance checks
and reduced-cost computation under stochastic, time-bin-dependent travel times.

Core algorithm (solve_pricing_network):
  1. Initial label creation (create_initial_labels): for each task node and every feasible
     depot leave time, a seed label [depot, task] is generated. Leave times that would
     trivially violate chance constraints or extended time windows are pruned upfront.
  2. Label extension: labels are extended along arcs of the pricing network. At each step
     the start-time PMF/CDF at the new task is computed via discrete convolution of the
     current PMF with the (bin-dependent) stochastic travel-time distribution. Extensions
     violating chance constraints (alpha-service-level), extended time windows, branching
     constraints (t_max_le / t_max_gr), or forbidden arcs are discarded immediately.
  3. Dominance check (Label.dominates): before storing a new label it is checked against
     all existing labels at the same node with the same quantile finish time. A label
     dominates another if it is no worse in reduced cost, return time, unvisited task set,
     start-time CDF (stochastic dominance), and Gomory cut coefficients. Dominated labels
     are pruned to keep the label set tractable.
  4. Sink evaluation: once a label reaches the sink, its reduced cost is computed. For AMP
     nodes the single best label is returned (find_best_label); for DMP nodes the optimal
     skill composition is selected ex-post for each sink label (find_best_label_skill_comp);
     for the Yuan et al. (2015) approach all negative-cost labels are returned
     (find_best_label_all_labels).
  5. Elementarity enforcement: if the best sink label contains repeated tasks, the
     corresponding tasks are added to the set of explicit task resources and the network
     is re-solved until an elementary path is found.

"""
import numpy as np
import time
import math
from src.pricing.utils import *
from config.config import *


def create_initial_labels(pricing_network, forb_tour_idxs, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr,
                        t_max_le, t_max_gr, skill_comp_cnt, solve_as_dmp, node_in_tree, yuan_approach):
    """Create initial labels for each task node and possible depot leave time. Leave times that would incur
    unnecessary waiting at the first task are automatically skipped.
    Each initial label corresponds to a single-tasks subtour [depot, task_i] for some task_i.

    Parameters
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which initial labels are to be computed
    forb_tour_idxs: list
        List of indices of forbidden tours in current pricing network. Indices always start at 0 and increase incrementally.
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
        t_max_le: dict
        Maps tasks to their latest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
    t_max_gr: dict
        Maps tasks to their earliest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
    skill_comp_cnt: dict
        Maps skill levels to the number of required workers (from skill composition)
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).
    node_in_tree: GH_node
        Current node in the branching tree
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used

    Returns
    -------
    initial_labels: list
        List of initial Label objects
    node_labels
        Maps tasks to the list of initial labels containing that task
    labels_to_finish_times
        Maps tasks and finish times to a list of initial labels containing that task and finish it at the respective
        finish time
    """

    # 1. initialization
    # 1.1 define some counters
    labels_created = 0
    labels_kept = 0
    labels_skipped = 0
    # 1.2 dictionary to map nodes and finish times to the corresponding labels (used for dominance check)
    labels_to_finish_times = {}     # store all labels in list based on their first task and their worst-case finish time
    for node in pricing_network.tasks:
        labels_to_finish_times[node] = {}
        for t in range(pricing_network.earliest_finishes[node], pricing_network.latest_finishes_viol[node] + 1):
            labels_to_finish_times[node][t] = []
    # 1.3 list of initial labels and list of labels per node
    initial_labels = []
    node_labels = {}    # keys: node names, values: labels present at current node (one for each path that ends at node)
    for node in pricing_network.tasks:
        node_labels[node] = []

    # 2. for each node in the pricing network: compute its initial labels
    for node in node_labels:
        resources_arc = pricing_network.resources[pricing_network.source, node]
        # 2.1 get latest leave time that can potentially lead to a feasible label (i.e., a label that satisfies the chance
        # constraint and time windows)
        latest_leave_time = -1
        for time_bin in pricing_network.travel_times_per_bin:
            # 2.1.1 get alpha-quantile for travel time in current bin
            prob_reach_in_time = 0  # probability of reaching task node in time
            for tt in pricing_network.travel_times_per_bin[time_bin][(pricing_network.source, node)]:
                prob_reach_in_time += pricing_network.travel_times_per_bin[time_bin][(pricing_network.source, node)][tt]
                if prob_reach_in_time >= pricing_network.alpha:
                    latest_leave_time_per_bin = pricing_network.latest_starts[node] - tt      # latest leave time such that chance constraint satisfied at task node
                    break

            # 2.1.2 update latest leave time for current bin to ensure TW satisfaction
            latest_leave_time_per_bin = min(latest_leave_time_per_bin, pricing_network.latest_starts_viol[node] -
                                        pricing_network.max_travel_times_per_bin[time_bin][pricing_network.source, node])

            # 2.1.3 if latest leave time is before time bin: current time bin will always lead to chance constraint/TW violations => continue with next bin
            if latest_leave_time_per_bin < pricing_network.instant_per_bin[time_bin][0]:
                continue

            # 2.1.4 if latest leave time is contained in bin: update latest leave time
            if latest_leave_time_per_bin <= pricing_network.instant_per_bin[time_bin][-1]:
                latest_leave_time = max(latest_leave_time, latest_leave_time_per_bin)

            # 2.1.5 if latest leave time is after time bin: update latest leave time to the end of the bin
            else:
                latest_leave_time = max(latest_leave_time, pricing_network.instant_per_bin[time_bin][-1])



        # 2.2 get lower bound on earliest leave time assuming no waiting
        # note: for time-dependant travel times, waiting can be an efficient decision because one can avoid worse
        # travel time distributions in the subsequent bin
        max_travel_time = max([pricing_network.max_travel_times_per_bin[time_bin][(pricing_network.source, node)] for time_bin in
                        pricing_network.max_travel_times_per_bin])
        earliest_leave_no_waiting = pricing_network.earliest_starts[node] - max_travel_time

        # 2.3 get all bins between earliest leave time w/o waiting and latest leave time
        bins_no_waiting = [b for b in range(pricing_network.bin_per_instant[earliest_leave_no_waiting],
                                            pricing_network.bin_per_instant[latest_leave_time] + 1)]

        # 2.3 get list of potential leave times
        # 2.3.1 check each instant in the relevant bins
        potential_leave_times = []
        found_earliest_start_no_waiting = False # true iff. a leave time was found, with which the task is started at its earliest start and no waiting is incurred
        max_leave_time_with_waiting_time = 0 # latest leave time that would imply waiting (and thus guarantees starting at earliest start)
        for time_bin in bins_no_waiting:
            for t in pricing_network.instant_per_bin[time_bin]:
                latest_arrival_time = t + pricing_network.max_travel_times_per_bin[time_bin][(pricing_network.source, node)]
                # if leaving at t incurs no waiting at task and does not violate LF^v: always add it to potential leave times
                if latest_arrival_time > pricing_network.earliest_starts[node]:
                    if latest_arrival_time <= pricing_network.latest_starts_viol[node]:
                        potential_leave_times.append(t)
                # if leaving at t implies task is started at its earliest start: remember this
                elif latest_arrival_time == pricing_network.earliest_starts[node]:
                    potential_leave_times.append(t)
                    found_earliest_start_no_waiting = True
                # else: remember leave time as candidate for earliest leave that guarantees start at earliest start
                else:
                    max_leave_time_with_waiting_time = max(max_leave_time_with_waiting_time, t)
        # 2.3.2 if no time instant was found that ensures starting at earliest start: also add latest potential leave time
        # that guarantees a start at earliest start
        if not found_earliest_start_no_waiting:
            potential_leave_times.insert(0, max_leave_time_with_waiting_time)


        # 2.4 Create labels for each possible depot leave time
        for t in potential_leave_times:
            # note: waiting might be an optimal decision when time bins are used. if time bins are not used or only a
            # single time bin exists, one can safely prune any time instants t that would incur waiting time
            labels_created += 1
            # 2.4.1 create Label objects, set initial data
            node_t_label = Label(forb_tour_idxs, pricing_network.formation, t, node_in_tree)
            node_t_label.median_finish_per_task[pricing_network.source] = node_t_label.median_finish
            for i in range(len(node_in_tree.gomory_cuts_lhs)):
                node_t_label.frac_coeffs_per_cut[i] = 0
            node_t_label.sequence = [pricing_network.source]
            node_t_label.length = 1
            # calculate start time distribution at node
            # set initial start time distribution at source node
            node_t_label.start_time_pmf = {t: 1}
            node_t_label.start_time_cdf = {t: 1}
            node_t_label.start_time_cdf_per_task[pricing_network.source] = {t: 1}
            # calculate start time distribution at first task node
            start_time = time.time()
            node_t_label.add_node(node)  # add to sequence
            label_t_bin_pred = pricing_network.bin_per_instant[node_t_label.median_finish]
            start_time_pmf, start_time_cdf, is_feasible, infeasibility_str, infeasibility_reason = node_t_label.get_start_time_distr(node,
                                                    pricing_network.travel_times_per_bin[label_t_bin_pred][(pricing_network.source, node)],
                                                    0, pricing_network.earliest_starts[node],
                                                    pricing_network.latest_finishes[node],
                                                    pricing_network.latest_finishes_viol[node],
                                                    pricing_network.task_execution_times[node],
                                                    pricing_network.weights[node], pricing_network.alpha,
                                                    pricing_network.max_travel_times_per_bin[label_t_bin_pred][(pricing_network.source, node)],
                                                    pricing_network.min_travel_times_per_bin[label_t_bin_pred][(pricing_network.source, node)],
                                                    pricing_network.quantile_travel_times_per_bin[label_t_bin_pred][(pricing_network.source, node)])

            pricing_network.calculate_start_distr_time += time.time() - start_time
            # 2.4.2 continue if extension is not feasible, since extension will also be infeasible for every instant > t
            # (this can happen when using time bin-travel times. similar to the comment of step 1.2, especially at the
            # last time step of a bin chance constraints are sometimes violated, while they are satisfied at the first
            # time step of the subsequent bin)
            if not is_feasible:
                if infeasibility_str in ["chance_constr_violated", "chance_constr_precheck_violated"]:
                    continue
                else:
                    raise Exception(f"Initial label deemed infeasible due to non-chance constraint related reasons: {infeasibility_str}")
            # 2.4.3 skip label if label violates branching on finish time constraints
            if node_t_label.quantile_case_finish < t_max_gr[node] or node_t_label.quantile_case_finish > t_max_le[node]:
                continue
            # 2.4.4 skip arcs that are forbidden
            if ("source", node, node_t_label.quantile_case_finish) in node_in_tree.forbidden_arcs:
                continue

            label_t_bin_succ = pricing_network.bin_per_instant[node_t_label.median_finish]
            node_t_label.start_time_cdf_per_task[node] = start_time_cdf # store start time CDF for each task to evaluate TW viol. probability
            node_t_label.start_time_pmf = start_time_pmf
            node_t_label.start_time_cdf = start_time_cdf


            # 2.4.5 calculate reduced costs
            # expected route costs (finish times + penalty)
            node_t_label.cost += node_t_label.task_costs  # add costs
            node_t_label.tour_cost += node_t_label.task_costs
            node_t_label.task_cost_dict[node] = node_t_label.task_costs
            # branching and busy penalty
            t_from = t      # team leaves depot at time t
            t_to = max(t + pricing_network.quantile_travel_times_per_bin[label_t_bin_pred][(pricing_network.source, node)],
                       pricing_network.earliest_starts[node]) + pricing_network.task_execution_times[node]
            if t_to != node_t_label.quantile_case_finish:
                raise Exception("Initial label creation: bad quantile finish time calculation")
            branching_penalty = get_branching_penalty(rho_gr, rho_le, t_from, t_to)
            busy_penalty = get_busy_penalty(pricing_network.formation, delta, t_from, t_to, skill_comp_cnt, solve_as_dmp)
            arc_penalty = 0
            if (node_t_label.sequence[-2], node_t_label.sequence[-1], node_t_label.quantile_case_finish) in zeta_le:
                arc_penalty = (zeta_le[(node_t_label.sequence[-2], node_t_label.sequence[-1], node_t_label.quantile_case_finish)] -
                               zeta_gr[(node_t_label.sequence[-2], node_t_label.sequence[-1], node_t_label.quantile_case_finish)])
            node_t_label.cost += branching_penalty + busy_penalty + arc_penalty
            node_t_label.cost -= mu[node]
            node_t_label.task_reward += mu[node]
            node_t_label.total_busy_penalty += busy_penalty
            # if approach by Yuan et al. (2015) is used: calculate reduced cost w.r.t each skill composition
            if yuan_approach and solve_as_dmp:
                for skill_comp_id in pricing_network.skill_comps_cnts_ids:
                    busy_penalty_skill_comp = get_busy_penalty(node_t_label.formation, delta, t_from, t_to,
                                                    pricing_network.skill_comps_cnts_ids[skill_comp_id], solve_as_dmp)
                    node_t_label.cost_per_skill_comp[skill_comp_id] = busy_penalty_skill_comp + branching_penalty + arc_penalty + node_t_label.task_costs - mu[node]

            # gomory cut costs
            for i in range(len(node_in_tree.gomory_cuts_lhs)):
                # skip trivial psi values to avoid unnecessary calculations
                if psi[i] < eps_global / 10:
                    continue
                # adjust cut coefficients
                coeff = get_gomory_cut_coeff_increase(pricing_network.formation, node_in_tree.u_kt[i], t_from, t_to)
                node_t_label.frac_coeffs_per_cut[i] += coeff + node_in_tree.u_task[i][node]
                # if coefficient is almost integer: re-calculate coefficient and update it
                # this is done to avoid numerical problems propagating when extending a label multiple times
                if abs(round(node_t_label.frac_coeffs_per_cut[i], 0) - node_t_label.frac_coeffs_per_cut[i]) < eps_gc_recalc:
                    node_t_label = recompute_gomory_cut_coeff(node_t_label, i, node_in_tree, pricing_network, t_to, psi)
                # else: if coefficient is almost integer: round it to avoid numerical problems
                else:
                    if abs(round(node_t_label.frac_coeffs_per_cut[i], 0) - node_t_label.frac_coeffs_per_cut[i]) < eps_gc_round:
                        node_t_label.frac_coeffs_per_cut[i] = round(node_t_label.frac_coeffs_per_cut[i], 0)
                    # increase reduced cost in case fractional coefficient is now >= 1
                    if node_t_label.frac_coeffs_per_cut[i] >= 1:
                        integer_part = math.floor(node_t_label.frac_coeffs_per_cut[i])
                        node_t_label.cost += integer_part * psi[i]
                        node_t_label.integer_coeffs_per_cut[i] += integer_part
                        node_t_label.gomory_penalty[i] += integer_part * psi[i]
                        node_t_label.frac_coeffs_per_cut[i] -= integer_part


            # 2.4.6 if mu[task] < task_cost: task reward is smaller than expected cost of finishing -> do not create node
            # this is only valid if there are no ">=" branches on vehicle counts after the tour has started, as the
            # reduced cost of these constraints are non-positive
            relevant_t_gr_branches = [tt for tt in node_in_tree.t_gr if tt >= node_t_label.start_time_from_depot]
            # Note: this only works for the approach by Hagn et al. (2026).
            # when arcs are forced, it might be beneficial to use 'bad' arcs first in order to be able to use forced
            # arcs with large negative duals afterwards
            if not yuan_approach and len(relevant_t_gr_branches) == 0 and mu[node] < node_t_label.task_costs:
                labels_skipped += 1
                break

            # 2.4.7. if probability distribution of start times at any other task is guaranteed to be below alpha:
            # task can not be visited anymore -> add task to list of unreachable tasks
            # bisect.bisect_left slower than using find_alpha_quantile if start_time_cdf is small
            max_prob_leq_alpha = find_alpha_quantile(start_time_cdf, pricing_network.alpha)
            max_prob_leq_alpha_finish = max_prob_leq_alpha + pricing_network.task_execution_times[node]
            for edge in pricing_network.graph.out_edges(node):
                other_task = edge[1]
                if other_task == pricing_network.sink:
                    continue
                if max_prob_leq_alpha_finish +  pricing_network.min_travel_times_per_bin[label_t_bin_succ][node, other_task] > pricing_network.latest_starts[other_task]:
                    node_t_label.unreachable_tasks_prob += [other_task]
                    node_t_label.length += 1        # see Righini, Salani p. 160

            # 2.4.7 check if new label is dominated by any previous label. New label cannot dominate old labels as its
            # starting time is larger than all previously generated labels', thus stochastic dominance is not possible.
            start_time = time.time()
            new_dominated = False
            for other_label in node_labels[node]:
                if other_label.dominates(node_t_label, pricing_network.task_resources_consd,
                                     psi, solve_as_dmp, yuan_approach):  # check if old label dominates new label
                    new_dominated = True
                    break
            pricing_network.check_dominance_time += time.time() - start_time

            # 2.4.8 increase ext.non_dom_formation by 1 if node is non-dominated
            node_t_label.non_dom_formation += resources_arc.non_dom_formation  # add 1 if node is non-dominated

            # 2.4.9 set is_forbidden_subpath to 0 for all tours that do not use the current first edge
            if not new_dominated:
                # get all indices of forbidden tours that use the current edge
                resources = pricing_network.resources[(pricing_network.source, node)]
                # if edge (source, node) is not part of forbidden tour => can't be a subpath => set is_forbidden_subpath to 0
                for forb_tour_id in node_t_label.is_forbidden_subpath:
                    if forb_tour_id not in resources.forb_res:
                        node_t_label.is_forbidden_subpath[forb_tour_id] = False

            # 2.4.10 if label is not dominated: add it to initial_labels[node] and node_labels
            if not new_dominated:
                initial_labels.append(node_t_label)
                node_labels[node].append(node_t_label)
                labels_to_finish_times[node][node_t_label.quantile_case_finish].append(node_t_label)
                labels_kept += 1

    return node_labels, initial_labels, labels_to_finish_times

def spprc_algorithm(pricing_network, forb_tour_idxs, forb_skill_comps, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr,
                    only_best_tasks, delta_tasks, t_max_le, t_max_gr,
                    solve_as_dmp, node_in_tree, yuan_approach):
    """Label algorithm for the ESPPRC. Solves the ESPPRC for a single input pricing network. Workflow is as follows:
    1. creates initial single-task labels for each possible depot leave time
    2. iteratively extends labels along arcs of the pricing network.
        2.1 Infeasible labels are immediately pruned
        2.2 Whenever a new label is created, it is verified if the label is dominated by any other label (-> new label is pruned)
            or if it dominates any other label (-> other label is pruned)
        2.3 repeats until no more feasible extensions are left for any label
    3. computes the best (for approach by Yuan et al. (2015): all) label and its corresponding skill composition
    4. if the best label has negative reduced cost: returns it and terminates
    5. else: returns an empty result indicating no negative columns exist

    Two additional features are implemented:
    - only_best_tasks: if enabled, Steps 2 to 5 are first performed while only considering extensions along arcs (i, j) where
      j is among the tasks with the best task reward (mu) values of all nodes incident to i. If no label with negative
      reduced cost is found with this approach, Steps 2 to 5 are repeated, then considering all possible extensions.
    - elementary condition relaxed: paths are allowed to visit tasks multiple times. If at the end of Step 5, all
      labels with negative reduced costs contain duplicate tasks, the respective tasks' resources are introduced and Steps 2 to 5
      are repeated, this time forbidden the repetition of the respective tasks. This step is repeated until either
        (a) all tasks resources are considered, i.e. no tasks are allowed to be duplicated
        (b) a label with negative reduced cost, which does not contain duplicate tasks, was found
        (b) no labels with negative reduced cost were found.
      This procedure is called decremental state space relaxation and was introduced by Righini and Salani (2008).

    These two additional features are applied in the following order: first, task resources are introduced if necessary.
    If a negative column is found, it is returned. Else, additional task resources are introduced. This process repeats
    until all task resources are introduced or a negative column is found. If still no negative column was found,
    this function returns a corresponding result and is then called again, this time with only_best_tasks = False.
    Again, the same logic is applied.


    Parameters:
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed
    forb_tour_idxs: list
        List of indices of forbidden tours in current pricing network. Indices always start at 0 and increase incrementally.
    forb_skill_comps:
        Maps indices of forbidden tours to their skill composition (used for identification of forbidden tours when current
        node is solved using the DMP)
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
    only_best_tasks: bool
        Indicates if each pricing network should initially only be solved using the tasks with the most negative
        dual value
    delta_tasks: int
        No. of incident tasks for each node that should be considered if only_best_tasks is set to True
    t_max_le: dict
        Maps tasks to their latest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
    t_max_gr: dict
        Maps tasks to their earliest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).
    node_in_tree: GH_node
        Current node in the branching tree
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used

    Returns
    -------
    min_path: list
        Task sequence (including depot) of all labels in min_label
    min_cost: float
        Minimum reduced cost of any found label
    min_label: list
        List of Label objects with negative reduced costs that are to be returned (can contain multiple or just one element,
        depending on if the approach by Yuan et al. (2015) is used or not)
    count_labels: int
        Number of labels created during the pricing step
    count_dom: int
        Number of labels dominated
    tour_costs: float
        Objective function value of the (reduced cost-wise) minimal label
    sink_labels_network: int
        Number of non-dominated labels at the sink during the pricing step
    no_initial_labels: int
        Number of initial labels created by workers.pricing.dynamic_programming.create_initial_labels()
    """
    # 1. Initialization
    # 1.1 define some logging variables
    no_initial_labels = 0
    count_labels = 1 # no. of labels generated
    count_dom = 0   # no. of labels found to be dominated


    # 1.3 if all tasks in the current network are part of forced tours: skip the network
    if pricing_network.tasks == []:
        return [], math.inf, None, count_labels, count_dom, math.inf,  0, 0

    # 1.4 if only_best_tasks == True, we only allow extensions to delta_tasks many tasks with maximum task execution reward
    if only_best_tasks:
        find_best_tasks(pricing_network, mu, delta_tasks)

    # 1.5 if node is solved using the AMP: define default skill comp. set as they do not play a role in the algorithm
    if not solve_as_dmp:
        skill_comps = [None]
        skill_comps_cnt = [None]
        last_skill_comp_idx = 0
    # else: copy all skill compositions
    else:
        skill_comps = pricing_network.skill_comps.copy()
        skill_comps_cnt = pricing_network.skill_comps_cnt.copy()
        last_skill_comp_idx = 0     # only solve pricing network for a single skill comp. as dominance relations transfer to other skill comps.

    # 1.6 if no skill comps exist (i.e. formation can not be built using the total available workforce): skip pricing network
    if len(skill_comps) == 0:
        # this should never happen but is kept as a fallback option
        return [], math.inf, None, count_labels, count_dom, math.inf, 0, 0


    # 1.7 generate all initial labels
    # create containers for labels w.r.t. their worst-case finish times
    node_labels, initial_labels, labels_to_finish_times = create_initial_labels(pricing_network, forb_tour_idxs,
                                                                            mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr,
                                                                            t_max_le, t_max_gr, skill_comps_cnt[0],
                                                                            solve_as_dmp, node_in_tree, yuan_approach)


    # 2. perform labeling algorithm
    forbidden_skill_comps_per_sink_label = {}  # keys: list indexes of sink labels, values: list of skill comps. for which label/tour is forbidden
    skill_comp_cnt = skill_comps_cnt[last_skill_comp_idx]

    no_of_tasks = len(pricing_network.tasks)

    # 2.1 create initial labels for each task
    node_labels[pricing_network.sink] = []
    node_labels[pricing_network.source] = []
    no_initial_labels += len(initial_labels)
    solution_found = False
    # 2.2 repeatedly look for negative labels until either (a) tour with red. costs < 0 has been found or (b) all tours found have red. costs >= 0
    while not solution_found: # repeat until either (a) tour with red. costs < 0 has been found or (b) all tours found have red. costs >= 0
        unexp = initial_labels.copy()  # list of unexplored labels, initialize with initial labels

        while unexp:
            curr_label = unexp.pop(0)   # remove label that is now extended
            curr_node = curr_label.get_last_node()  # get last node of current label (=path)
            curr_task = curr_node
            execution_time_head = pricing_network.task_execution_times[curr_task]       # execution time of current last task
            # 2.2.1 for each incident node: try to extend label
            for edge in pricing_network.graph.edges(curr_node):   # try extending path along all directions
                succ_node = edge[1]
                # a. get relevant information regarding the tail node (earliest start, task name, execution time, task weight)
                earliest_start_tail = pricing_network.earliest_starts[succ_node]
                execution_time_tail = pricing_network.task_execution_times[succ_node]
                tail_weight = pricing_network.weights[succ_node]
                # b. try to extend label towards tail node
                label_t_bin_pred = pricing_network.bin_per_instant[curr_label.median_finish]
                new_label = curr_label.extend(succ_node, pricing_network.sink, pricing_network.resources[curr_node, succ_node],
                                              pricing_network.travel_times_per_bin[label_t_bin_pred],
                                              pricing_network.min_travel_times_per_bin[label_t_bin_pred],
                                              pricing_network.max_travel_times_per_bin[label_t_bin_pred],
                                              pricing_network.quantile_travel_times_per_bin[label_t_bin_pred],
                                              execution_time_head,
                                              earliest_start_tail, pricing_network.latest_starts, pricing_network.latest_finishes,
                                              pricing_network.latest_finishes_viol,
                                              tail_weight, execution_time_tail, pricing_network.alpha,
                                              pricing_network.end_horizon,
                                              pricing_network.task_resources_consd, pricing_network, mu, delta, rho_gr, rho_le, psi,
                                              zeta_le, zeta_gr, only_best_tasks, t_max_le, t_max_gr, skill_comp_cnt,
                                              solve_as_dmp, node_in_tree, yuan_approach, node_in_tree.forbidden_arcs,
                                              node_in_tree.forced_arcs, pricing_network.source)
                # c. skip forbidden arcs
                if new_label is not None:
                    if (edge[0], edge[1], new_label.quantile_case_finish) in node_in_tree.forbidden_arcs:
                        continue
                if new_label == None:   # label cannot be extended (e.g. task already performed, forbidden tour created)
                    continue
                count_labels += 1
                # d. if succ is the sink we treat it more easily
                if succ_node == pricing_network.sink:
                    node_labels[pricing_network.sink].append(new_label)
                    # if node is solved using the DMP: if tour is equal to forbidden tour -> remember skill comp. of
                    # forbidden tour (will be skipped when looking for the optimal skill composition and label w.r.t. reduced costs)
                    if solve_as_dmp:
                        forbidden_skill_comps_per_sink_label[len(node_labels[pricing_network.sink])-1] = []
                        for forb_tour_id in new_label.is_forbidden_subpath:
                            if new_label.is_forbidden_subpath[forb_tour_id]:
                                forbidden_skill_comps_per_sink_label[len(node_labels[pricing_network.sink])-1].append(forb_skill_comps[forb_tour_id])
                    continue

                # e. check if labels at incident node (succ_node) dominate the new label
                new_dominated = False       # True iff. new_label is dominated
                old_dominated = []
                start_time = time.time()
                for old_label in labels_to_finish_times[succ_node][new_label.quantile_case_finish]:
                    if old_label.dominates(new_label, pricing_network.task_resources_consd, psi,
                                           solve_as_dmp, yuan_approach):  # check if old label dominates new label
                        new_dominated = True
                        count_dom += 1
                        break
                    elif new_label.dominates(old_label, pricing_network.task_resources_consd,
                                             psi, solve_as_dmp, yuan_approach):
                        old_dominated.append(old_label) # check if new label dominates old label
                pricing_network.check_dominance_time += time.time() - start_time

                # f. if new label not dominated: add to list of labels at succ_node and mark as unexplored
                if not new_dominated:
                    # store label
                    labels_to_finish_times[succ_node][new_label.quantile_case_finish].append(new_label)
                    node_labels[succ_node].append(new_label)
                    # if maximum route length is reached: extend to sink
                    if new_label.length == no_of_tasks:
                        new_node = new_label.get_last_node()  # get last node of current label (=path)
                        new_task = succ_node
                        new_execution_time_head = pricing_network.task_execution_times[
                            new_task]  # execution time of current last task
                        # get relevant information regarding the tail node (earliest start, task name, execution time, task weight)
                        earliest_start_tail = pricing_network.earliest_starts[pricing_network.sink]
                        execution_time_tail = pricing_network.task_execution_times[pricing_network.sink]
                        tail_weight = pricing_network.weights[pricing_network.sink]
                        label_t_bin_pred = pricing_network.bin_per_instant[new_label.median_finish]
                        sink_label = new_label.extend(pricing_network.sink, pricing_network.sink,
                                                      pricing_network.resources[new_node, pricing_network.sink],
                                                      pricing_network.travel_times_per_bin[label_t_bin_pred],
                                                      pricing_network.min_travel_times_per_bin[label_t_bin_pred],
                                                      pricing_network.max_travel_times_per_bin[label_t_bin_pred],
                                                      pricing_network.quantile_travel_times_per_bin[label_t_bin_pred],
                                                      new_execution_time_head, earliest_start_tail, pricing_network.latest_starts,
                                                      pricing_network.latest_finishes,
                                                      pricing_network.latest_finishes_viol, tail_weight, execution_time_tail,
                                                      pricing_network.alpha, pricing_network.end_horizon,
                                                      pricing_network.task_resources_consd, pricing_network, mu, delta, rho_gr,
                                                      rho_le, psi, zeta_le, zeta_gr, only_best_tasks, t_max_le, t_max_gr,
                                                      skill_comp_cnt, solve_as_dmp, node_in_tree, yuan_approach,
                                                      node_in_tree.forbidden_arcs, node_in_tree.forced_arcs,
                                                      pricing_network.source)
                        # skip forbidden arcs
                        if sink_label is not None:
                            if (new_node, pricing_network.sink, sink_label.quantile_case_finish) in node_in_tree.forbidden_arcs:
                                continue
                        if sink_label != None:
                            node_labels[pricing_network.sink].append(sink_label)
                            # if node is solved using the DMP: if tour is equal to forbidden tour -> remember skill
                            # comp. of forbidden tour (will be skipped when looking for the optimal skill composition
                            # and label w.r.t. reduced costs)
                            if solve_as_dmp:
                                forbidden_skill_comps_per_sink_label[len(node_labels[pricing_network.sink])-1] = []
                                for forb_tour_id in sink_label.is_forbidden_subpath:
                                    if sink_label.is_forbidden_subpath[forb_tour_id]:
                                        forbidden_skill_comps_per_sink_label[len(node_labels[pricing_network.sink])-1].append(
                                            forb_skill_comps[forb_tour_id])

                    # else: mark node as unexplored since it has not fully consumed the "length" resource
                    elif new_label.length < no_of_tasks:
                        unexp.append(new_label)
                    else:
                        raise ValueError
                    # remove old labels dominated by new label (old path cannot be part of an optimal solution)
                    for old_dom_label in old_dominated:
                        node_labels[succ_node].remove(old_dom_label)
                        labels_to_finish_times[succ_node][new_label.quantile_case_finish].remove(old_dom_label)
                        if old_dom_label in unexp:
                            unexp.remove(old_dom_label)
                            count_dom += 1

        # 2.3 find best label (path) which is elementary
        # if node is solved using the AMP: simply find all relevant negative columns
        if not solve_as_dmp:
            no_of_sink_labels = len(node_labels[pricing_network.sink])
            # 2.3.1 if approach by Yuan et al. (2015) is not used -> return a single column
            if not yuan_approach:
                min_label, min_cost, tour_cost, min_cost_all = find_best_label(node_labels[pricing_network.sink], pricing_network)
                if min_label is not None:
                    min_label.min_skill_comp = pricing_network.default_comp
                    min_label.min_skill_comp_cnt = pricing_network.default_comp_cnt
                    min_label = [min_label]
                else:
                    min_label = []
            # 2.2 else: return all negative columns at the sink
            else:
                min_label, min_cost, tour_cost, min_cost_all = find_best_label_all_labels(node_labels[pricing_network.sink],
                                                                                          pricing_network)
                if min_label != []:
                    for label in min_label:
                        label.min_skill_comp = pricing_network.default_comp
                        label.min_skill_comp_cnt = pricing_network.default_comp_cnt

        # 2.4 if node is solved using the AMP: also find best composition for all labels at sink
        else:
            no_of_sink_labels = len(node_labels[pricing_network.sink])
            min_label, min_cost, tour_cost, min_cost_all = find_best_label_skill_comp(node_labels[pricing_network.sink],
                                                                                      pricing_network,
                                                                                      delta, skill_comps,
                                                                                      skill_comps_cnt,
                                                                                      forbidden_skill_comps_per_sink_label)

        # 2.5 if feasible (w.r.t time windows, chance constraint, and path elementarity) tour with red. costs < 0 has
        # been found: return column
        if min_cost < eps_col_neg * 10:
            min_path = [label.sequence[1:-1] for label in min_label]
            sink_labels_network = no_of_sink_labels

            return min_path, min_cost, min_label, count_labels, count_dom, tour_cost, sink_labels_network, \
                   no_initial_labels

        # 2.6 else: if min. costs of all feasible and infeasible (e.g. duplicate) tours are non-negative:
        # no column with red. costs < 0 exists -> set solution_found to true, return empty solution
        elif min_cost_all >= eps_col_neg * 10:
            solution_found = True
        # 2.7 else: reset list of labels present at each node, the list of initial labels (done at the beginning of the
        # 'while not solution found'-loop, and perform labeling algorithm again, this time considering additional
        # task resources
        else:
            for node in node_labels:
                node_labels[node] = []

    # 3. if no tour with red. costs < 0 has been found: return empty solution
    return [], math.inf, None, count_labels, count_dom, math.inf, 0, no_initial_labels


def recompute_gomory_cut_coeff(label, i, node_in_tree, pricing_network, t_to, psi):
    """Recompute the gomory cut coefficient for a given label and a given gomory cut. This function is called whenever
    a gomory cut coefficient is almost integer to avoid numerical issues that could lead to faulty reduced cost calculations.

    Parameters
    ----------
    label: workers.pricing.dynamic_programming.Label
        Label for which gomory cut coefficient should be recomputed
    i: int
        Index of target gomory cut
    node_in_tree: GH_node
        Current node in the branching tree
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed
    t_to: int
        End of the considered time interval (quantile-case finish time of subpath described by label)
    psi: dict
        Maps gomory cuts to their dual values

    Returns
    -------
    label: workers.pricing.dynamic_programming.Label
        Input label with recomputed gomory cut coefficient
    """
    # re-calculate coefficient
    # 1. task coefficients
    coeff_recalc = sum([node_in_tree.u_task[i][task] for task in label.sequence[1:] if task != pricing_network.sink])
    # 2. workforce coefficient
    coeff_recalc += get_gomory_cut_coeff_increase(pricing_network.formation, node_in_tree.u_kt[i],
                                                  label.start_time_from_depot, t_to)
    # update coefficient and reduced cost to new coefficient, independent of whether it differs
    # from the current one
    if abs(round(coeff_recalc, 0) - coeff_recalc) < eps_gc_round:
        coeff_recalc = round(coeff_recalc, 0)
    # update reduced cost
    label.cost = label.cost - label.gomory_penalty[i] + psi[i] * math.floor(coeff_recalc)
    # update all other relevant values (fractional/integer coefficients, GC penalties)
    label.integer_coeffs_per_cut[i] = math.floor(coeff_recalc)
    label.frac_coeffs_per_cut[i] = coeff_recalc - math.floor(coeff_recalc)
    label.gomory_penalty[i] = psi[i] * math.floor(coeff_recalc)

    return label

def find_best_tasks(pricing_network, mu, delta_tasks):
    """Identifies the 'delta_tasks' tasks with the smallest task execution rewards (i.e. task duals) reachable by
    each task.

    Parameters
    ----------
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed
    mu: dict
        Maps task execution constraints to their dual values
    delta_tasks: int
        Number of tasks with the smallest mu values that should be identified

    Returns
    -------
    pricing_network.smallest_mus: dict
        Dictionary with two-stage key structure that contains bool indicating which arcs in the pricing network
        connect two tasks where the tail node is among the 'delta_tasks' smallest mus incident to the head node.
    """
    # 1. initialize dummy mu values for source and sink
    # Note: mu value are always >= 0
    mu[pricing_network.source] = -1
    mu[pricing_network.sink] = -1

    # 2. set up dict structures for logging mu values of all tasks reachable by each task
    mus = {}
    pricing_network.smallest_mus = {}
    delta_largest_mu = {}
    for node in pricing_network.graph.nodes:
        mus[node] = []
        pricing_network.smallest_mus[node] = {}
    # fill dict
    for (task_i, task_j) in pricing_network.graph.edges:
        mus[task_i].append(mu[task_j])
    # 3. for each task: get 'delta_tasks' tasks reachable by the current task with minimum mu
    for task in mus:
        if len(mus[task]) == 0:
            delta_largest_mu[task] = -math.inf
        else:
            delta_largest_mu[task] = sorted(mus[task])[-min(delta_tasks, len(mus[task]))]
    # 4. for each arc: store bool indicating if the tail node's mu is among the 'delta_tasks' smallest mus of all tasks
    # reachable from the head node
    for (task_i, task_j) in pricing_network.graph.edges:
        if task_j == pricing_network.sink:
            pricing_network.smallest_mus[task_i][task_j] = True
        else:
            pricing_network.smallest_mus[task_i][task_j] = mu[task_j] >= delta_largest_mu[task_i]

def find_best_label_all_labels(sink_labels, pricing_network):
    """Find all labels with negative reduced costs if no skill compositions are considered. Needed for the approach by
    Yuan et al. (2015), where all columns with negative reduced costs are returned.

    Parameters
    ----------
    sink_labels: dict
        List of labels that end at the sink
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed

    Returns
    -------
    neg_labels: list
        List of labels with negative reduced cost
    min_cost: float
        Minimum reduced cost of any sink label
    tour_cost: float
        Objective value of the (reduced cost-wise) minimal sink label
    min_cost_all: float
        Objective value of the (reduced cost-wise) minimal sink label, including labels that have duplicate tasks

    """
    # 1. initialize values that specify the best label
    min_cost = math.inf
    min_cost_all = math.inf  # minimum red. cost of any tours, including tours with duplicate tasks
    tour_cost = math.inf
    neg_labels = []
    # 2. iterate over all sink labels, remember all negative ones and identify labels with duplicate tasks
    # note: if all negative labels contain duplicate tasks, a re-solving of the pricing network while considering additional
    # task resources is necessary
    for sink_label in sink_labels:
        # 2.1 check for duplicate tasks and update general minimum label cost
        dup = check_for_duplicate_tasks(sink_label)
        if sink_label.cost < min_cost_all:
            min_cost_all = sink_label.cost
        # 2.2 if label contains no duplicate (i.e. path is elementary): also update min_cost and tour_cost
        if len(dup) == 0:
            if sink_label.cost < min_cost:
                min_cost = sink_label.cost
                tour_cost = sink_label.tour_cost
            if sink_label.cost < eps_col_neg:
                neg_labels.append(sink_label)
        # 2.3 if label has duplicate tasks: update list of task resources in case the network needs to be re-solved
        else:
            for task in dup:
                if task not in pricing_network.task_resources_consd:
                    pricing_network.task_resources_consd += [task]

    # 3. sort negative labels descending w.r.t. reduced cost and return
    neg_labels = sorted(neg_labels, key = lambda x: x.cost)

    return neg_labels, min_cost, tour_cost, min_cost_all

def find_best_label(sink_labels, pricing_network):
    """Find label with min. reduced costs if no skill compositions are considered, i.e., if the node is solved using
    the AMP.

    Parameters
    ----------
    sink_labels: dict
        List of labels that end at the sink
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed

    Returns
    -------
    neg_labels: list
        List of labels with negative reduced cost
    min_cost: float
        Minimum reduced cost of any sink label
    tour_cost: float
        Objective value of the (reduced cost-wise) minimal sink label
    min_cost_all: float
        Objective value of the (reduced cost-wise) minimal sink label, including labels that have duplicate tasks

    """
    # 1. initialize values that specify the best label
    min_cost = math.inf
    min_cost_all = math.inf  # minimum red. cost of any tours, including tours with duplicate tasks
    tour_cost = math.inf
    min_label = None
    # 2. iterate over sink labels and look for the best one
    # note: if all negative labels contain duplicate tasks, a re-solving of the pricing network while considering additional
    # task resources is necessary
    for sink_label in sink_labels:
        # 2.1 check for duplicate tasks and update general minimum label cost
        dup = check_for_duplicate_tasks(sink_label)
        if sink_label.cost < min_cost_all:
            min_cost_all = sink_label.cost
        # 2.2 if label contains no duplicate (i.e. path is elementary): also update min_cost and tour_cost
        if len(dup) == 0:  # only consider elementary paths
            if sink_label.cost < min_cost:
                min_label = sink_label
                min_cost = sink_label.cost
                tour_cost = sink_label.tour_cost
        # 2.3 if label has duplicate tasks: update list of task resources in case the network needs to be re-solved
        else:
            for task in dup:
                if task not in pricing_network.task_resources_consd:
                    pricing_network.task_resources_consd += [task]

    return min_label, min_cost, tour_cost, min_cost_all


def find_best_label_skill_comp(sink_labels, pricing_network, delta, skill_comps, skill_comps_cnt,
                               forbidden_skill_comps_per_sink_label):
    """Find label and corresponding skill comp. with min. reduced costs. Used when a node is solved using the DMP.
    For each sink label, the reduced cost of the label for each suitable skill composition is computed ex-post.
    The sink label plus the corresponding skill composition that yield the smallest reduced cost is returned.
    For additional details, see Section 5.3 and Appendix E see Hagn et al. (2026).

    Parameters
    ----------
    sink_labels: dict
        List of labels that end at the sink
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed
    delta: dict
        Maps workforce constraints to their dual values
    skill_comps: list
        List of skill compositions suitable for the current pricing network, including the assignment of skill levels
        to jobs within the formation (i.e. which worker is assigned to a job with which skill level)
    skill_comps_cnt: list
        List of skill compositions suitable for the current pricing network
        Maps skill levels to the number of required workers (from skill composition)
    forbidden_skill_comps_per_sink_label: dict
        Maps sink labels to a list of skill compositions that are not suitable for the given label, because the label
        would then be equal to a forbidden tour

    Returns
    -------
    [min_label]: single-item list
        Label with minimum reduced cost
    min_cost: float
        Minimum reduced cost of any sink label
    tour_cost: float
        Objective value of the (reduced cost-wise) minimal sink label
    min_cost_all: float
        Objective value of the (reduced cost-wise) minimal sink label, including labels that have duplicate tasks
    """
    # 1. initialize values that specify the best label
    min_cost = math.inf
    min_cost_all = math.inf  # minimum red. cost of any tours, including tours with duplicate tasks
    tour_cost = math.inf
    min_label = None
    min_skill_comp = None
    min_skill_comp_cnt = None
    # 2. check each label and get optimal skill composition w.r.t reduced costs
    for i in range(len(sink_labels)):
        sink_label = sink_labels[i]
        # 2.1 check for duplicate tasks
        dup = check_for_duplicate_tasks(sink_label)
        min_skill_comp_sink_label = None
        min_skill_comp_cnt_sink_label = None
        min_cost_sink_label = math.inf

        # 2.1 get cost of 1 worker of any skill level
        t_from = sink_label.start_time_from_depot
        t_to = sink_label.quantile_case_finish
        cost_per_worker = {}
        for k in pricing_network.workers:
            skill_comp_cnt = {k: 1}
            cost_per_worker[k] = get_busy_penalty(pricing_network.formation, delta, t_from, t_to, skill_comp_cnt, True)
        # 2.2 for each skill comp: calculate reduced costs of tour and update optimal skill comp. if necessary
        for j in range(len(skill_comps)):
            skill_comp = skill_comps[j]
            skill_comp_cnt = skill_comps_cnt[j]
            # skip skill comps. that would lead to a forbidden tour
            if skill_comp in forbidden_skill_comps_per_sink_label[i]:
                continue
            label_cost = sink_label.cost - sink_label.total_busy_penalty + sum([skill_comp_cnt[k] * cost_per_worker[k] for
                                                                                k in skill_comp_cnt])
            # if skill comp. better than best known skill comp: save it
            if label_cost < min_cost_sink_label:
                min_cost_sink_label = label_cost
                min_skill_comp_sink_label = skill_comp
                min_skill_comp_cnt_sink_label = skill_comp_cnt

        # 2.3 if label better than best known label and does not contain duplicate tasks: save it
        if min_cost_sink_label < min_cost_all:
            min_cost_all = min_cost_sink_label
        if len(dup) == 0:  # only consider elementary paths
            if min_cost_sink_label < min_cost:
                min_label = sink_label
                min_label.cost = min_cost_sink_label
                min_cost = min_cost_sink_label
                tour_cost = sink_label.tour_cost
                min_skill_comp = min_skill_comp_sink_label
                min_skill_comp_cnt = min_skill_comp_cnt_sink_label
                min_label_index = i
        # 2.3 if label has duplicate tasks: update list of task resources in case the network needs to be re-solved
        else:
            for task in dup:
                if task not in pricing_network.task_resources_consd:
                    pricing_network.task_resources_consd += [task]
            print(f"tour with repetition detected: {sink_label.sequence[1:-1]}")

    # 3. update min skill comp. of best label and return
    if min_label is not None:
        min_label.min_skill_comp = min_skill_comp
        min_label.min_skill_comp_cnt = min_skill_comp_cnt

    return [min_label],  min_cost, tour_cost, min_cost_all

def check_for_duplicate_tasks(label):
    """Check for duplicate tasks in a given label.

    Parameters
    ----------
    label: workers.pricing.dynamic_programming.Label
        Label for which gomory cut coefficient should be recomputed

    Returns
    -------
    seen_dup: list
        List of duplicate tasks in label
    """

    seen = set()
    seen_dup = set()
    for task in label.sequence:
        if task in seen:
            seen_dup.add(task)
        else:
            seen.add(task)
    return list(seen_dup)


class Label():

    def __init__(self, forb_tour_idxs, formation, start_time_from_depot, node):
        """Class for objects that comprise a subpath in a given pricing network, which starts at the depot at a certain
        time and executes a sequence of tasks.

        Parameters
        forb_tour_idxs: list
            List of indices of forbidden tours in current pricing network. Indices always start at 0 and increase incrementally.
        formation: dict
            Maps skill levels to the number of required workers (including downgrading).
        start_time_from_depot: int
            Leave time of the subtour described by the label
        node: GH_node
            Current node in the branching tree
        """
        self.sequence = []  # sequence of nodes making up the label
        self.cost = 0  # total cost of sequence
        self.tasks = []  # list of task resources in chronological order
        self.hash = ""  # hash of path
        self.non_dom_formation = 0  # >0 iff. all tasks along path are non-dominated
        self.start_time_pmf = {}  # probability distribution of start time at most recent node
        self.start_time_cdf = {}  # probability distribution of start time at most recent node
        self.tour_cost = 0  # expected tour cost E(c_r) for tour associated to given label
        self.total_busy_penalty = 0     # total workforce penalty accumulated
        self.unreachable_tasks_prob = []  # tasks unreachable since LB(P(task reached in time window)) < alpha with LB() being a lower bound (see section 2.6)
        self.earliest_start_head = start_time_from_depot
        self.latest_start_head = start_time_from_depot
        self.start_time_from_depot = start_time_from_depot  # time at which team leaves the depot
        self.formation = formation      # dict with keys = workers levels, values = no. of workers of specific skill level (without downgrading)
        self.quantile_case_finish = start_time_from_depot  # worst-case finish time of last task, initially set to depot leave time
        self.median_finish = start_time_from_depot # median finish time of last task, can be used for reduced cost calculation and dominance checks
        self.median_finish_per_task = {} # used for construct GH_tour object from Label object later on
        self.length = 0     # length of route (excluding source/sink)
        self.task_cost_dict = {}
        self.tw_viol_prob = {}  # keys: tasks along path, value: probability of violating the time window
        self.start_time_cdf_per_task = {}
        self.frac_coeffs_per_cut = {}  # keys: indices of gomory cuts, values: fractional part of coefficient in each gomory cut
        self.integer_coeffs_per_cut = {}
        self.is_forbidden_subpath = {}
        self.cost_per_skill_comp = {}     # only for yuan approach: reduced cost of label for each skill comp.
        for forb_idx in forb_tour_idxs:
            self.is_forbidden_subpath[forb_idx] = True
        self.task_reward = 0
        self.gomory_penalty = {}
        for i in range(len(node.gomory_cuts_lhs)):
            self.gomory_penalty[i] = 0
            self.integer_coeffs_per_cut[i] = 0


    def get_last_node(self):
        """Get the current last node of the label.
        """
        return self.sequence[-1]

    def add_node(self, node):
        """Add a node to the subpath described by the current label. Also updates the label hash.
        """

        self.sequence.append(node)
        self.hash += node

    def clone(self, node):
        """Creates a mutable clone of the current label. Ensures data sovereignty, i.e., any changes made to the cloned
        Label object are NOT applied to the original Label object.

        Parameters
        ----------
        node: GH_node
            Current node in the branching tree

        Returns
        -------
        cln: Label
            Cloned label
        """
        cln = Label(list(self.is_forbidden_subpath.keys()), self.formation, self.start_time_from_depot, node)
        cln.cost = self.cost
        cln.tour_cost = self.tour_cost
        cln.total_busy_penalty = self.total_busy_penalty
        cln.non_dom_formation = self.non_dom_formation
        cln.start_time_pmf = self.start_time_pmf.copy()
        cln.start_time_cdf = self.start_time_cdf.copy()
        cln.earliest_start_head = self.earliest_start_head
        cln.latest_start_head = self.latest_start_head
        cln.unreachable_tasks_prob = self.unreachable_tasks_prob.copy()
        cln.formation = self.formation.copy()
        cln.quantile_case_finish = self.quantile_case_finish
        cln.median_finish = self.median_finish
        cln.length = self.length
        cln.sequence = self.sequence.copy()
        cln.tasks = self.tasks.copy()
        cln.task_cost_dict = self.task_cost_dict.copy()
        cln.tw_viol_prob = self.tw_viol_prob.copy()
        cln.start_time_cdf_per_task = self.start_time_cdf_per_task.copy()
        cln.frac_coeffs_per_cut = self.frac_coeffs_per_cut.copy()
        cln.integer_coeffs_per_cut = self.integer_coeffs_per_cut.copy()
        cln.task_reward = self.task_reward
        cln.gomory_penalty = self.gomory_penalty.copy()
        cln.is_forbidden_subpath = self.is_forbidden_subpath.copy()
        cln.cost_per_skill_comp = self.cost_per_skill_comp.copy()
        cln.median_finish_per_task = self.median_finish_per_task.copy()
        return cln

    def to_string(self):
        """Convert label to a printable string. Similar to GH_tour.to_string()
        """
        str_out = "Formation-> "
        for skill_level in self.formation:
            str_out += "level " + str(skill_level) + ":" + str(self.formation[skill_level]) + " "
        str_out += "\n"
        if hasattr(self, "min_skill_comp_cnt"):
            str_out += "Skill comp.: "
            for sl_req in self.min_skill_comp_cnt:
                if self.min_skill_comp_cnt[sl_req] > 0:
                    str_out += str(sl_req) + "<-" + str(self.min_skill_comp_cnt[sl_req])
                    str_out = str_out.rstrip(",")
                    str_out += "; "
            str_out = str_out.rstrip(";")
            str_out += "\n"

        for task in self.sequence[1:-1]:
            str_out += "task" + str(task) + " [" + str(min(self.start_time_cdf_per_task[task].values())) + "," + str(
                max(self.start_time_cdf_per_task[task].values())) + "[\n"
        str_out += "leave time: " + str(self.start_time_from_depot) + ", (quantile) return time: " + str(self.quantile_case_finish) + "\n"
        str_out += "tour cost: " + str(self.tour_cost) + "\n"
        str_out += "reduced cost: " + str(self.cost)
        str_out += "task reward: " + str(self.task_reward) + "\nbusy penalty: " + str(self.total_busy_penalty) + "\n"
        str_out += "total gomory penalty: " + str(sum(self.gomory_penalty.values()))

        return str_out

    def calculate_cdf(self, pmf):
        """Computes a cumulative distribution function (CDF) given a probability mass function (PMF).

        Parameters
        ----------
        pmf: dict
            Maps time instants to probabilities. Must be a PMF

        Returns
        -------
        cdf: dict
            CDF corresponding to PMF
        """

        cdf = {}
        cdf_val_last = 0
        for t in pmf:
            cdf_val_last = cdf_val_last + pmf[t]  # last nontrivial CDF value
            cdf[t] = cdf_val_last
        return cdf

    def get_start_time_distr(self, task, travel_times, execution_time_head, earliest_start_tail, latest_finish_tail,
                             latest_finish_viol_tail, execution_time_tail, tail_weight, alpha, max_travel_time, min_travel_time,
                             quantile_dist):
        """Calculate probability distribution of start times at tail node of an extension.

        Parameters
        ----------
        task: str
            Tail node of target extension. Can also be the sink
        travel_times: dict
            Maps travel_times to to their probabilities
        execution_time_head: int
            Execution time of the head node of the target extension (i.e., last node of current label)
        earliest_start_tail: int
            Earliest possible start time of the tail node
        latest_finish_tail: int
            Latest finish time of the tail node that does not incur a penalty
        latest_finish_viol_tail: int
            Latest feasible finish time of the tail node
        execution_time_tail: int
            Execution time of the tail node
        tail_weight: float
            Weight of tail node (0 if tail node is sink)
        alpha: float in [0, 1]
            Minimum service level per task
        max_travel_time: int
            Maximum travel time between head and tail node
        min_travel_time: int
            Minimum travel time between head and tail node
        quantile_dist: int
            gamma-quantile of travel time distribution between head and tail node

        Returns
        -------
        start_time_pmf_tail: dict
            Maps start times of current label at tail node to its probabilities
        start_time_cdf_tail: dict
            CDF corresponding to start_time_pmf_tail
        is_feasible: bool
            Indicates if the target extension satisfies the extended time windows and the alpha chance constraint
        infeasibility_reason: str | None
            Quick description of the reason for infeasibility if extension is infeasible, else None
        infeasibility_reason_detailed: str | None
            More expressive description of the reason for infeasibility if extension is infeasible, else None

        """


        earliest_arrival_tail = self.earliest_start_head + execution_time_head + min_travel_time
        latest_arrival_tail = self.latest_start_head + execution_time_head + max_travel_time
        # 1. if latest arrival at tail node < earliest_start_tail: trivial PMF, can create start time PMF and CDF directly
        if latest_arrival_tail < earliest_start_tail:
            start_time_pmf_tail = {earliest_start_tail: 1}
            start_time_cdf_tail = {earliest_start_tail: 1}
            self.tw_viol_prob[task] = 0
            is_feasible = True
        # 2. else: if earliest arrival at tail node > latest start at tail node: cannot satisfy chance constraint
        elif earliest_arrival_tail > latest_finish_tail - execution_time_tail:
            return {}, {}, False, "chance_constr_precheck_violated", "earliest_arrival > latest_finish - execution_time => CC violated"
        # 3. else: create arrival time PMF using convolution and transform into start time PMF, calculate CDF and
        # check for feasibility
        else:
            # 3.1 arrival time PMF
            idx_pmf = [i for i in range(earliest_arrival_tail, latest_arrival_tail + 1)]  # all possible start times at tail node
            val_pmf = np.convolve(list(self.start_time_pmf.values()), list(travel_times.values()))
            start_time_pmf_tail = {k: v for (k, v) in zip(idx_pmf, val_pmf)}        # initialized as arrival time pmf
            # 3.2 start time PMF
            # if earliest arrival at tail node < earliest_start_tail: compute start time PMF
            # else: arrival time PMF = start time PMF (no early arrivals possible), thus we have to do nothing
            if earliest_arrival_tail < earliest_start_tail:
                for i in list(start_time_pmf_tail):  # break after reaching earliest start of tail node
                    if i >= earliest_start_tail:
                        break
                    start_time_pmf_tail[earliest_start_tail] += start_time_pmf_tail[i]
                    del start_time_pmf_tail[i]  # remove "arrival before time window opens" event

            # 3.3 get CDF
            start_time_cdf_tail = self.calculate_cdf(start_time_pmf_tail)

            # 3.4 feasibility check
            # extension is feasible iff. latest finish time > latest feasible finish time and CDF at latest feasible finish time <= alpha
            # and P(finishing > latest finish time w/ violation) = 0
            if max(latest_arrival_tail, earliest_start_tail) + execution_time_tail > latest_finish_viol_tail:
                return {}, {}, False, "lf_viol_violated", "CC violated: max(latest_arrival, earliest_start) + execution_time > latest_finish_viol"
            if max(latest_arrival_tail, earliest_start_tail) + execution_time_tail > latest_finish_tail and start_time_cdf_tail[latest_finish_tail-execution_time_tail] < alpha:
                return {}, {}, False, "chance_constr_violated", "CC violated: max(latest_arrival, earliest_start) + execution_time > latest_finish AND start_time_cdf[latest_finish-execution_time] < alpha"
            if latest_finish_tail-execution_time_tail in start_time_cdf_tail:
                self.tw_viol_prob[task] = 1 - start_time_cdf_tail[latest_finish_tail-execution_time_tail]
            else:
                self.tw_viol_prob[task] = 0
        # 4. update costs
        self.task_costs = 0  # expected costs of performing task (expected finish time + lateness penalty)
        for t in start_time_pmf_tail:
            if start_time_pmf_tail[t] == 0:   # skip zero probability start times
                continue
            finish_time_tail = t + execution_time_tail
            self.task_costs += tail_weight * start_time_pmf_tail[t] * finish_time_tail  # weighted finish time is always accrued
            if finish_time_tail > latest_finish_tail:   # quadratic penalty for finish outside time window
                self.task_costs += tail_weight * start_time_pmf_tail[t] * (finish_time_tail - latest_finish_tail) ** 2  # prob * lateness penalty (quadratic)


        # 5. update Label attributes
        self.quantile_case_finish = max(self.quantile_case_finish + quantile_dist, earliest_start_tail) + execution_time_tail
        self.median_finish = find_alpha_quantile(start_time_cdf_tail, 0.5) + execution_time_tail
        self.median_finish_per_task[task] = self.median_finish
        self.earliest_start_head = max(earliest_arrival_tail, earliest_start_tail)
        self.latest_start_head = max(latest_arrival_tail, earliest_start_tail)



        return start_time_pmf_tail, start_time_cdf_tail, True, None, None



    def extend(self, ext_task, sink, resources, travel_times, min_travel_times, max_travel_times, quantile_travel_times,
               execution_time_head, earliest_start_tail, latest_starts,
               latest_finishes, latest_finishes_viol, tail_weight, execution_time_tail, alpha, end_horizon,
               resources_consd, pricing_network, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr, only_best_tasks, t_max_le, t_max_gr,
               skill_comp_cnt, solve_as_dmp, node_in_tree, yuan_approach, forbidden_arcs, forced_arcs, source):
        """Extend path along (self.sequence[-1], node). Computes the start time CDF at the subsequent task node, and
        verifies if time windows and alpha chance constraints are satisfied. Also performs several sanity checks
        along the way.

        Parameters
        ----------
        ext_task: str
            Tail node of the target extension
        sink: str
            Name of the sink node in the current pricing network
        resources: workers.pricing.graph.Resources
            ResourceObjects that contains information necessary to identify forbidden (sub)paths
        travel_times: dict
            Maps travel times along the target extension to their probabilities for each arc in the pricing network
        min_travel_times: dict
            Maps arcs to their minimum travel time
        max_travel_times: dict
            Maps arcs to their maximum travel time
        quantile_travel_times: dict
            Maps arcs to the gamma-quantile of their travel time distribution
        execution_time_head: int
            Execution time of the head node of the target extension (i.e., last node of current label)
        earliest_start_tail: int
            Earliest possible start time of the tail node
        latest_starts: dict
            Maps tasks to their latest start times (that do not incur penalties for the current formation)
        latest_finishes: dict
            Maps tasks to their latest finish times (that do not incur penalties)
        latest_finishes: dict
            Maps tasks to their latest feasible finish times (i.e., that do not violate the extended time window)
        tail_weight: float
            Weight of tail node (0 if tail node is sink)
        alpha: float in [0, 1]
            Minimum service level per task
        end_horizon: int
            Last time instant of the considered planning horizon
        resources_consd: list
            List of tasks whose task resources are currently considered
        pricing_network: workers.pricing.dynamic_programming.PricingNetwork
            Pricing network for which negative columns are to be computed:
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
        only_best_tasks: bool
            Indicates if each pricing network should initially only be solved using the tasks with the most negative
            dual value
        t_max_le: dict
            Maps tasks to their latest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
        t_max_gr: dict
            Maps tasks to their earliest finish time (in the omega_gamma-scenario of travel times) derived by branching on task finish times
        skill_comp_cnt: dict
            Maps skill levels to the number of required workers (from skill composition)
        solve_as_dmp: bool
            Indicates if the current node is solved as a DMP (True) or an AMP (False).
        node_in_tree: GH_node
            Current node in the branching tree
        yuan_approach: bool
            Indicates if the approach by Yuan et al. (2015) should be used
        forbidden_arcs: list
            List of arcs that are forbidden at the current branching tree node (from branching on arcs, only for approach
            by Yuan et al. (2015))
        forced_arcs: list
            List of arcs that are forced at the current branching tree node (from branching on arcs, only for approach
            by Yuan et al. (2015))
        source: str
            Name of the source node in the current pricing network

        Returns
        -------
        ext: Label
            Label obtained by extending self along (self.sequence[-1], node)

        """

        # 1. preprocessing: check several easy-to-verify cases that always imply an infeasible extension
        # 1.1 skip any extensions along forbidden arcs
        # need to calculate theoretical quantile-case finish time at node
        quantile_case_finish = self.quantile_case_finish + quantile_travel_times[(self.sequence[-1], ext_task)] + execution_time_tail
        if (self.sequence[-1], ext_task, quantile_case_finish) in forbidden_arcs:
            return None

        # 1.2 skip any extensions along arcs that conflict with forced arcs
        for arc in forced_arcs:
            # case 1 (ONLY when predecessor != source): predecessor needs to be extended to a different successor
            if self.sequence[-1] != source and self.sequence[-1] == arc[0] and ext_task != arc[1]:
                return None
            # case 2 (ONLY when successor != sink: successor needs to be visited starting from a different predecessor
            if ext_task != sink and ext_task == arc[1] and self.sequence[-1] != arc[0]:
                return None

        # 2. if extension is towards sink: calculate busy and branching penalty, update value and return label
        # other parameters (such as start time distributions) do not have to be updated
        if ext_task == sink:
            ext = self.clone(node_in_tree)
            ext.add_node(ext_task)  # add node to path
            # 2.1 calculate arrival time CDF at sink node to allow for calculation of median depot arrival time
            dists = travel_times[(self.sequence[-1], ext_task)]
            max_travel_time = max_travel_times[(self.sequence[-1], ext_task)]
            min_travel_time = min_travel_times[(self.sequence[-1], ext_task)]
            quant_travel_time = quantile_travel_times[(self.sequence[-1], ext_task)]
            _, start_time_cdf, is_feasible, _, _  = ext.get_start_time_distr(ext_task, dists,
                                                                                execution_time_head, earliest_start_tail,
                                                                                end_horizon + 1, end_horizon + 1,
                                                                                execution_time_tail, tail_weight, alpha,
                                                                                max_travel_time, min_travel_time, quant_travel_time)
            if not is_feasible:# quick sanity check
                raise Exception("Sink extension is infeasible")

            # 2.2 calculate busy & branching & arc usage penalty, update tour costs
            t_from = self.quantile_case_finish
            t_to = t_from + quantile_travel_times[(self.sequence[-1], sink)]
            if t_to != ext.quantile_case_finish:
                raise Exception("Sink extension: bad quantile finish time calculation")
            # workforce penalty
            busy_penalty = get_busy_penalty(self.formation, delta, t_from, t_to, skill_comp_cnt, solve_as_dmp)
            # branching penalty
            branching_penalty = get_branching_penalty(rho_gr, rho_le, t_from, t_to)
            # arc usage penalty
            arc_penalty = 0
            if (ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish) in zeta_le:
                arc_penalty = (zeta_le[(ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish)] -
                               zeta_gr[(ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish)])
            ext.cost += busy_penalty + branching_penalty + arc_penalty
            ext.total_busy_penalty += busy_penalty
            # if approach by Yuan et al. (2015) is used: calculate reduced cost w.r.t each skill composition
            if yuan_approach and solve_as_dmp:
                for skill_comp_id in pricing_network.skill_comps_cnts_ids:
                    busy_penalty_skill_comp = get_busy_penalty(self.formation, delta, t_from, t_to,
                                                    pricing_network.skill_comps_cnts_ids[skill_comp_id], solve_as_dmp)
                    ext.cost_per_skill_comp[skill_comp_id] += busy_penalty_skill_comp + branching_penalty + arc_penalty

            # gomory cut penalty
            for i in range(len(node_in_tree.gomory_cuts_lhs)):
                # skip trivial psi values to avoid unnecessary calculations
                if psi[i] < eps_global / 10:
                    continue
                # 1. adjust cut coefficients
                coeff = get_gomory_cut_coeff_increase(pricing_network.formation, node_in_tree.u_kt[i], t_from, t_to)
                ext.frac_coeffs_per_cut[i] += coeff

                # re-calculate coefficient if it is almost integer to avoid numerical inaccuracies leading to wrong
                # reduced costs
                if abs(round(ext.frac_coeffs_per_cut[i], 0) - ext.frac_coeffs_per_cut[i]) < eps_gc_recalc:
                    ext = recompute_gomory_cut_coeff(ext, i, node_in_tree, pricing_network, t_to, psi)
                # else: if coefficient is almost integer: round it to avoid numerical problems
                else:
                    if abs(round(ext.frac_coeffs_per_cut[i], 0) - ext.frac_coeffs_per_cut[i]) < eps_gc_round:
                        ext.frac_coeffs_per_cut[i] = round(ext.frac_coeffs_per_cut[i], 0)
                    # increase reduced cost in case fractional coefficient is now > 1
                    if ext.frac_coeffs_per_cut[i] >= 1:
                        integer_part = math.floor(ext.frac_coeffs_per_cut[i])
                        ext.cost += integer_part * psi[i]
                        ext.integer_coeffs_per_cut[i] += integer_part
                        ext.frac_coeffs_per_cut[i] -= integer_part
                        ext.gomory_penalty[i] += integer_part * psi[i]

            # 2.3 set other necessary attribute values
            ext.quantile_case_finish = t_to
            ext.tasks.append(ext_task)

            # 2.4 skip extension if arc is forbidden
            if (ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish) in forbidden_arcs:
                return None

            # 2.5 check if label is now equal to a forbidden tour and update ext.is_forbidden_subpath
            for forb_tour_id in ext.is_forbidden_subpath:
                # 2.5.1 if edge (source, node) is not part of forbidden tour => can't be a subpath => set is_forbidden_subpath to 0
                if forb_tour_id not in resources.forb_res:
                    ext.is_forbidden_subpath[forb_tour_id] = False
                # 2.5.2 else: if start times differ: tour does not equal forbidden (sub)tour => also set is_forbidden_subpath to 0
                else:
                    # if start times differ: tour is not equal to a forbidden tour
                    if resources.forb_res[forb_tour_id][0] != ext.start_time_from_depot:
                        ext.is_forbidden_subpath[forb_tour_id] = False
                    # else: tour is equal to a forbidden tour => return empty extension if AMP is solved
                    else:
                        if not solve_as_dmp:
                            return None

            return ext

        # 3. if extension passed prechecks and ext_task is not the sink: compute extensions
        else:
            # 3.1 if task reward < lower bound for expected finish time: extension cannot be part of an optimal solution
            # as reduced cost would increase if we include this task into current path
            # NOTE: this is only valid in the approach by Hagn et al. (2026) and if there are no ">=" branches on vehicle counts
            # after the tour has started, as the reduced cost of these constraints are non-positive
            relevant_t_gr_branches = [tt for tt in node_in_tree.t_gr if tt >= self.start_time_from_depot]
            if not yuan_approach and len(relevant_t_gr_branches) == 0 and mu[ext_task] < tail_weight * (earliest_start_tail + execution_time_tail):
                return None
            # 3.2 if only extension towards the (mu-wise) best tasks are allowed and ext_task is not among the most
            # profitable tasks reachable from self.sequence[-1]: do not allow extension
            if only_best_tasks and not pricing_network.smallest_mus[self.sequence[-1]][ext_task]:
                return None
            # 4.4 else: extension is feasible if a) task is still reachable and b) task resource is ignored OR task
            # resource is not ignored and task has not been visited before
            if ext_task not in self.unreachable_tasks_prob:
                if ext_task not in resources_consd or ext_task not in self.tasks:
                    ext = self.clone(node_in_tree)
                    # get start time distribution of tail node and check for feasibility w.r.t. service level alpha
                    dists = travel_times[(self.sequence[-1], ext_task)]
                    max_travel_time = max_travel_times[(self.sequence[-1], ext_task)]
                    min_travel_time = min_travel_times[(self.sequence[-1], ext_task)]
                    quant_travel_time = quantile_travel_times[(self.sequence[-1], ext_task)]
                    latest_finish_tail = latest_finishes[ext_task]
                    latest_finish_viol_tail = latest_finishes_viol[ext_task]
                    start_time = time.time()
                    ext.add_node(ext_task)  # add node to path
                    # forbid label extensions that would violate task finish time branching constraints
                    if ext_task != pricing_network.sink:
                        quantile_case_start_tail = max(self.quantile_case_finish + quant_travel_time, earliest_start_tail)
                        quantile_case_finish_tail  = quantile_case_start_tail + execution_time_tail
                        if quantile_case_finish_tail < t_max_gr[ext_task] or quantile_case_finish_tail > t_max_le[ext_task]:
                            return None
                    # compute start time distribution
                    start_time_pmf, start_time_cdf, is_feasible, infeasibility_str, infeasibility_reason = ext.get_start_time_distr(ext_task, dists,
                                                                                    execution_time_head, earliest_start_tail,
                                                                                    latest_finish_tail, latest_finish_viol_tail,
                                                                                    execution_time_tail, tail_weight, alpha,
                                                                                    max_travel_time, min_travel_time, quant_travel_time)
                    # skip extension if arc is forbidden
                    if (ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish) in forbidden_arcs:
                        return None
                    pricing_network.calculate_start_distr_time += time.time() - start_time
                    if not is_feasible:
                        return None
                    if min(start_time_cdf.keys()) < earliest_start_tail: # quick sanity check
                        raise Exception(f"ERROR: node {ext_task}: label's earliest start at {min(start_time_cdf.keys())} vs. {earliest_start_tail}")
                    ext.length += 1
                    # 4.5 if expected weighted finish time of task > task reward: extension cannot be part of an optimal route
                    # NOTE: this is only valid in the approach by Hagn et al. (2026) and if there are no ">=" branches on vehicle counts
                    # after the tour has started, as the reduced cost of these constraints are non-positive
                    relevant_t_gr_branches = [tt for tt in node_in_tree.t_gr if tt >= ext.start_time_from_depot]
                    if not yuan_approach and len(relevant_t_gr_branches) == 0 and mu[ext_task] < ext.task_costs:
                        return None
                    # 4.6 add costs, consisting of worker occupation costs, branching costs
                    # and expected route costs (finish time+lateness penalty)
                    ext.start_time_cdf_per_task[ext_task] = start_time_cdf
                    ext.start_time_pmf = start_time_pmf
                    ext.start_time_cdf = start_time_cdf
                    # busy and branching penalty
                    t_from = self.quantile_case_finish
                    t_to = ext.quantile_case_finish
                    busy_penalty = get_busy_penalty(self.formation, delta, t_from, t_to, skill_comp_cnt, solve_as_dmp)
                    branching_penalty = get_branching_penalty(rho_gr, rho_le, t_from, t_to)
                    # arc usage penalty
                    arc_penalty = 0
                    if (ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish) in zeta_le:
                        arc_penalty = zeta_le[(ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish)] - zeta_gr[(ext.sequence[-2], ext.sequence[-1], ext.quantile_case_finish)]
                    ext.cost = ext.cost - mu[ext_task] + ext.task_costs + busy_penalty + branching_penalty + arc_penalty
                    ext.task_reward += mu[ext_task]
                    ext.tour_cost += ext.task_costs
                    ext.task_cost_dict[ext_task] = ext.task_costs
                    ext.total_busy_penalty += busy_penalty
                    # if approach by Yuan et al. (2015) is used: calculate reduced cost w.r.t each skill composition
                    if yuan_approach and solve_as_dmp:
                        for skill_comp_id in pricing_network.skill_comps_cnts_ids:
                            busy_penalty_skill_comp = get_busy_penalty(self.formation, delta, t_from, t_to,
                                                            pricing_network.skill_comps_cnts_ids[skill_comp_id], solve_as_dmp)
                            ext.cost_per_skill_comp[skill_comp_id] += busy_penalty_skill_comp + branching_penalty + arc_penalty + ext.task_costs - mu[ext_task]
                    # gomory cut penalty
                    for i in range(len(node_in_tree.gomory_cuts_lhs)):
                        # skip trivial psi values to avoid unnecessary calculations
                        if psi[i] < eps_global / 10:
                            continue
                        #  adjust cut coefficients
                        coeff = get_gomory_cut_coeff_increase(pricing_network.formation, node_in_tree.u_kt[i], t_from, t_to)
                        ext.frac_coeffs_per_cut[i] += coeff + node_in_tree.u_task[i][ext_task]

                        # re-calculate coefficient if it is almost integer to avoid numerical problems
                        if abs(round(ext.frac_coeffs_per_cut[i], 0) - ext.frac_coeffs_per_cut[i]) < eps_gc_recalc:
                            ext = recompute_gomory_cut_coeff(ext, i, node_in_tree, pricing_network, t_to, psi)
                        # else: if coefficient is almost integer: round it to avoid numerical problems
                        else:
                            if abs(round(ext.frac_coeffs_per_cut[i], 0) - ext.frac_coeffs_per_cut[i]) < eps_gc_round:
                                ext.frac_coeffs_per_cut[i] = round(ext.frac_coeffs_per_cut[i], 0)
                            # increase reduced cost in case fractional coefficient is now >= 1
                            if ext.frac_coeffs_per_cut[i] >= 1:
                                integer_part = math.floor(ext.frac_coeffs_per_cut[i])
                                ext.cost += integer_part * psi[i]
                                ext.integer_coeffs_per_cut[i] += integer_part
                                ext.gomory_penalty[i] += integer_part * psi[i]
                                ext.frac_coeffs_per_cut[i] -= integer_part


                    ext.tasks.append(ext_task)  # add task corresponding to node to path

                    # 4.7 if probability distribution of start times at previously visited node is guaranteed to be below alpha:
                    # task can not be visited again anyway -> do not consume resource
                    max_prob_leq_alpha = find_alpha_quantile(start_time_cdf, alpha)
                    max_prob_leq_alpha_finish = max_prob_leq_alpha + execution_time_tail
                    for other_task in ext.tasks:
                        if max_prob_leq_alpha_finish +  min_travel_times[other_task, ext_task] > latest_starts[other_task]:
                            ext.tasks.remove(other_task)
                            ext.unreachable_tasks_prob += [other_task]



                    # 4.8 increase ext.non_dom_formation by 1 if node is non-dominated
                    ext.non_dom_formation += resources.non_dom_formation  # add 1 if node is non-dominated

                    # 4.9 check if label is now equal to a forbidden tour and update ext.is_forbidden_subpath
                    for forb_tour_id in ext.is_forbidden_subpath:
                        # 4.9.1 if edge (source, node) is not part of forbidden tour => can't be a subpath => set is_forbidden_subpath to 0
                        if forb_tour_id not in resources.forb_res:
                            ext.is_forbidden_subpath[forb_tour_id] = False
                        # 4.9.2 else: if start times differ: tour does not equal forbidden (sub)tour => also set is_forbidden_subpath to 0
                        else:
                            # if start times differ: tour is not equal to a forbidden tour
                            if resources.forb_res[forb_tour_id][0] != ext.start_time_from_depot:
                                ext.is_forbidden_subpath[forb_tour_id] = False
                    return ext

                else:
                    if ext_task in self.tasks:
                        print(f"tried extending route {self.tasks} back to task {ext_task}. Extension deemed impossible")
                    return None

    def dominates(self, other, resources_consd, psi, solve_as_dmp, yuan_approach):
        """Check if a given label "self" dominates another label "other".

        Parameters
        other: Label
            Label to check dominance against
        resources_consd: list
            List of tasks whose task resources are currently considered
        psi: dict
            Maps gomory cuts to their dual values
        solve_as_dmp: bool
            Indicates if the current node is solved as a DMP (True) or an AMP (False).
        yuan_approach: bool
            Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
            branch_on_task_finish_times = False, and use_dmp = True.

        Returns
        -------
        is_dominating: bool
            True if self dominates other, False else
        """

        # 1. check forbidden resources
        # if current subpath is equal to a forbidden subpath -> cannot dominate any other label
        # as this might lead to cutting off optimal labels
        for tour_idx in self.is_forbidden_subpath:
            if self.is_forbidden_subpath[tour_idx]:
                return False

        # 2. compare quantile-case finish times (for dual cost dominance)
        if self.quantile_case_finish != other.quantile_case_finish:
            return False
        # 3. compare median finish times (for same distributions during extensions)
        if self.median_finish != other.median_finish:
            return False
        # 4. gomory cut coefficients
        # self dominates other if for each coefficient either (a) the fractional coefficient of self is <= the coefficient
        # of other or (b) the reduced cost of self offset by 1*sum(psi[i] for i in gomory_cuts if self.coeff[i] > other.coeff[i])
        # is less or equal to the reduced cost of other
        # For details, see Section 5.6 of Hagn et al. (2026)
        for cut_idx in self.frac_coeffs_per_cut:
            gomory_cut_offset = 0
            if self.frac_coeffs_per_cut[cut_idx] > other.frac_coeffs_per_cut[cut_idx]:
                gomory_cut_offset += psi[cut_idx]
            if self.cost + gomory_cut_offset > other.cost:
                return False

        # 5. compare return times and reduced costs
        # 5.1 if node is solved as DMP: need to offset reduced costs by workforce penalty
        if solve_as_dmp:
            # 5.1.1 if approach by Yuan et al. (2015) is used: compare reduced cost for each skill composition
            if yuan_approach:
                for skill_comp_id in self.cost_per_skill_comp:
                    if self.cost_per_skill_comp[skill_comp_id] > other.cost_per_skill_comp[skill_comp_id]:
                        return False
            # 5.1.2 else: compare depot leave times and offset reduced cost
            else:
                if self.start_time_from_depot < other.start_time_from_depot:
                    return False
                if self.cost - self.total_busy_penalty  > other.cost - other.total_busy_penalty:
                    return False
        # 5.2 else: can directly compare reduced costs
        else:
            if self.cost > other.cost:
                return False

        # 6. check if tour consists of only dominated tasks
        if self.non_dom_formation == 0 and other.non_dom_formation > 0:  # can then obtain "self" label from pricing graph of another profile
            return False
        # 7. check if unvisited tasks of self are superset of unvisited tasks of other
        if self.length > other.length:
            return False
        for task in resources_consd:
            if task in self.tasks and task not in other.tasks:
                return False
        # 8. check for stochastic dominance of start time distributions
        if self.earliest_start_head > other.earliest_start_head:
            return False
        for t in range(other.earliest_start_head, self.latest_start_head):
            if self.start_time_cdf[t] < other.start_time_cdf[t]:
                return False

        return True