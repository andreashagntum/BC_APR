"""This script contains functions to heuristically compute labels with (hopefully) negative reduced costs.
Implemented are:
- greedy heuristic
- VND (with single-delete and single-insert operators)
"""

from src.pricing.dynamic_programming import Label, create_initial_labels
from config.config import *

def find_column_greedy(all_initial_labels, pricing_network, node_in_tree, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr,
                       t_max_le, t_max_gr, skill_comp_cnt, skill_comp, solve_as_dmp, yuan_approach):
    """Greedy heuristic to find paths of negative reduced cost. Start at the depot and extend to the next task with
    the lowest reduced cost increase.

    Parameters
    -----------
    all_initial_labels: list
        List of all initial labels obtained using workers.pricing.
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed
    node_in_tree: GH_node
        Current node in the branching tree
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
    skill_comp: dict
        Maps skill levels to the actual qualification of workers used  (from skill composition, based on skill_comp_cnt)
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used

    Returns
    -------
    curr_label: workers.pricing.dynamic_programming.Label | None
        Label found by the greedy algorithm (if it found any) | None if no label was found
    """

    # 1. find best label leaving the sink and select it as the starting point of the subpath
    all_initial_labels = sorted(all_initial_labels, key = lambda x: x.cost)
    curr_label = all_initial_labels[0]

    # 2. as long as a feasible extension exists: extend current label to all possible neighbors in the graph and check
    # reduced cost change. select the next task as the one with the minimum reduced cost increase
    while True:
        # 2.1 get label information at current last task
        curr_node = curr_label.get_last_node()  # last node of current label (=path)
        curr_task = curr_node
        execution_time_head = pricing_network.task_execution_times[curr_task]

        # 2.2 for each incident node: check if extension is feasible and, if so, calculate its reduced cost
        extensions = []     # list of all feasible extensions of the current label
        for edge in pricing_network.graph.edges(curr_node):
            succ_node = edge[1]
            # 2.2.1 get relevant information regarding the tail node (earliest start, task name, execution time, task weight)
            earliest_start_tail = pricing_network.earliest_starts[succ_node]
            execution_time_tail = pricing_network.task_execution_times[succ_node]
            tail_weight = pricing_network.weights[succ_node]
            # try to extend label towards tail node
            label_t_bin_pred = pricing_network.bin_per_instant[curr_label.median_finish]
            new_label = curr_label.extend(succ_node, pricing_network.sink,
                                          pricing_network.resources[curr_node, succ_node],
                                          pricing_network.travel_times_per_bin[label_t_bin_pred],
                                          pricing_network.min_travel_times_per_bin[label_t_bin_pred],
                                          pricing_network.max_travel_times_per_bin[label_t_bin_pred],
                                          pricing_network.quantile_travel_times_per_bin[label_t_bin_pred],
                                          execution_time_head,
                                          earliest_start_tail, pricing_network.latest_starts,
                                          pricing_network.latest_finishes,
                                          pricing_network.latest_finishes_viol,
                                          tail_weight, execution_time_tail, pricing_network.alpha,
                                          pricing_network.end_horizon,
                                          [], pricing_network, mu, delta, rho_gr, rho_le, psi,
                                          zeta_le, zeta_gr, False, t_max_le, t_max_gr, skill_comp_cnt,
                                          solve_as_dmp, node_in_tree, yuan_approach, node_in_tree.forbidden_arcs,
                                          node_in_tree.forced_arcs, pricing_network.source)
            if new_label is None:  # label cannot be extended (e.g. task already performed, forbidden tour created)
                continue
            # 2.2.2 check if forbidden arc was used
            if new_label is not None:
                if (edge[0], edge[1], new_label.quantile_case_finish) in node_in_tree.forbidden_arcs:
                    continue

            # 2.2.3 if label is not None, i.e. label is feasible: store it in list extensions
            extensions.append(new_label)

        # 2.3 if no feasible extensions have been found: check feasibility and return the label
        if len(extensions) == 0:
            # if label does not end at sink: sink arc is forbidden (can happen e.g. if tour is forbidden) -> return None
            if curr_label.get_last_node() != "sink":
                return None
            # assign skill_comp to it
            curr_label.min_skill_comp_cnt = skill_comp_cnt
            curr_label.min_skill_comp = skill_comp
            curr_label.cost = min(curr_label.cost_per_skill_comp.values())
            return curr_label
        # 2.4 else: sort all possible extensions w.r.t. reduced cost, pick the cheapest one as the new current label
        # and go to Step 2.1
        else:
            extensions = sorted(extensions, key = lambda x: x.cost)
            curr_label = extensions[0]


def find_column_vnd(current_sol_tours, pricing_network, node_in_tree, forb_tour_idxs, mu, delta, rho_gr, rho_le, psi, zeta_le,
                    zeta_gr, t_max_le,  t_max_gr, skill_comp_cnt, solve_as_dmp, yuan_approach, greedy_labels):
    """Search for new columns by applying the VND algorithm proposed by Yuan et al. (2015).
    Procedure: For each tour that is part of the current basic solution, apply all potential single-task delete and
    insert operations. If a negative column is found, it is returned.

    Parameters
    -----------
    current_sol_tours: list
        List of GH_tours at the current node that were selected in the current LP relaxation's solution (i.e., whose
         current objective value is > 0)
    pricing_network: workers.pricing.dynamic_programming.PricingNetwork
        Pricing network for which negative columns are to be computed
    node_in_tree: GH_node
        Current node in the branching tree
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
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used
    greedy_labels: list
        List of labels computed by find_column_greedy (if any has been found)

    Returns
    -------
    neg_labels: list
        All negative labels found by the VND algorithm
    """

    # 1. set up list of labels with negative reduced costs and filter all tours to which the VND should be applied
    neg_labels = []     # contains all labels w/ negative reduced cost found by the VND algorithm
    current_sol_tours_formation = [tour for tour in current_sol_tours if tour.formation_id == pricing_network.formation_id]

    # 2. perform all possible operations
    if len(current_sol_tours_formation) > 0:
        # 2.1 single-delete operations
        for tour in current_sol_tours_formation:
            # 2.1.1 delete one task at a time only if the tour contains more than one task (otherwise, the tour will be empty)
            if len(tour.tasks) == 1:
                continue
            for task_to_delete in tour.tasks:
                tasks = tour.tasks.copy()
                tasks.remove(task_to_delete)
                sequence = ["source"] + tasks + ["sink"]
                # 2.1.2 create label object and initialize it
                curr_label = Label(forb_tour_idxs, pricing_network.formation, tour.leave_time, node_in_tree)
                curr_label.median_finish_per_task[pricing_network.source] = curr_label.median_finish
                for i in range(len(node_in_tree.gomory_cuts_lhs)):
                    curr_label.frac_coeffs_per_cut[i] = 0
                curr_label.sequence = [pricing_network.source]
                curr_label.length = 1
                # 2.1.3 calculate start time distribution at node
                # set initial start time distribution at source node
                curr_label.start_time_pmf = {tour.leave_time: 1}
                curr_label.start_time_cdf = {tour.leave_time: 1}
                curr_label.start_time_cdf_per_task[pricing_network.source] = {tour.leave_time: 1}
                # 2.1.4 initialize reduced cost for each possible skill composition
                for skill_comps_cnts_id in pricing_network.skill_comps_cnts_ids:  # reduced cost of label per skill comp.
                    curr_label.cost_per_skill_comp[skill_comps_cnts_id] = 0

                # 2.1.5 extend label along the sequence, compute distributions and reduced cost
                i = 0
                while i < len(sequence) - 1:
                    curr_node = curr_label.get_last_node()  # get last node of current label (=path)
                    curr_task = curr_node
                    if i == 0:  # executing time of source node is zero
                        execution_time_head = 0
                    else:
                        execution_time_head = pricing_network.task_execution_times[curr_task]  # execution time of current last task

                    edge = (sequence[i], sequence[i+1])

                    # skip arcs that are not in the pricing network (extension guaranteed to be infeasible)
                    if edge not in pricing_network.graph.edges:
                        curr_label = None
                        i = len(sequence) - 1

                    # perform extension
                    if curr_label is not None:
                        succ_node = edge[1]
                        # get relevant information regarding the tail node (earliest start, task name, execution time, task weight)
                        earliest_start_tail = pricing_network.earliest_starts[succ_node]
                        execution_time_tail = pricing_network.task_execution_times[succ_node]
                        tail_weight = pricing_network.weights[succ_node]
                        # extend
                        label_t_bin_pred = pricing_network.bin_per_instant[curr_label.median_finish]
                        new_label = curr_label.extend(succ_node, pricing_network.sink,
                                                      pricing_network.resources[curr_node, succ_node],
                                                      pricing_network.travel_times_per_bin[label_t_bin_pred],
                                                      pricing_network.min_travel_times_per_bin[label_t_bin_pred],
                                                      pricing_network.max_travel_times_per_bin[label_t_bin_pred],
                                                      pricing_network.quantile_travel_times_per_bin[label_t_bin_pred],
                                                      execution_time_head,
                                                      earliest_start_tail, pricing_network.latest_starts,
                                                      pricing_network.latest_finishes,
                                                      pricing_network.latest_finishes_viol,
                                                      tail_weight, execution_time_tail, pricing_network.alpha,
                                                      pricing_network.end_horizon,
                                                      pricing_network.task_resources_consd, pricing_network, mu, delta, rho_gr,
                                                      rho_le, psi,
                                                      zeta_le, zeta_gr, False, t_max_le, t_max_gr, skill_comp_cnt,
                                                      solve_as_dmp, node_in_tree, yuan_approach, node_in_tree.forbidden_arcs,
                                                      node_in_tree.forced_arcs, pricing_network.source)
                        curr_label = new_label
                        # if arc is forbidden: break
                        if curr_label is not None:
                            if (edge[0], edge[1], new_label.quantile_case_finish) in node_in_tree.forbidden_arcs:
                                curr_label = None
                                i = len(sequence) - 1
                        # if extension is infeasible -> stop while loop and continue with the next tour
                        if new_label is None:
                            i = len(sequence)-1
                        else:
                            i += 1

                # 2.1.6 for each skill comp. with negative reduced cost: add curr_label to neg_labels
                if curr_label is not None:
                    if curr_label.sequence[-1] != "sink": # quick sanity check
                        raise Exception("VND algorithm, delete operator: label does not end at sink")
                    # check cost for each skill comp
                    for skill_comp_cnt_id in curr_label.cost_per_skill_comp:
                        if curr_label.cost_per_skill_comp[skill_comp_cnt_id] < eps_col_neg:
                            # set skill composition and store
                            neg_label = curr_label.clone(node_in_tree)
                            neg_label.min_skill_comp_cnt = pricing_network.skill_comps_cnts_ids[skill_comp_cnt_id]
                            neg_label.min_skill_comp = pricing_network.skill_comps_ids[skill_comp_cnt_id]
                            neg_label.cost = min(neg_label.cost_per_skill_comp.values())
                            neg_labels.append(neg_label)

        # 2.2 insert operators: logic analogous to delete operators
        for tour in current_sol_tours_formation:
            insertable_tasks = [task for task in pricing_network.tasks if task not in tour.tasks]
            for task_to_insert in insertable_tasks:
                insert_positions = [i for i in range(len(tour.tasks))]
                for pos_idx in insert_positions:
                    # 2.2.1 create new task sequence
                    tasks = tour.tasks.copy()
                    tasks.insert(pos_idx, task_to_insert)
                    sequence = ["source"] + tasks + ["sink"]
                    # 2.2.2 create label object and initialize it
                    curr_label = Label(forb_tour_idxs, pricing_network.formation, tour.leave_time, node_in_tree)
                    curr_label.median_finish_per_task[pricing_network.source] = curr_label.median_finish
                    for i in range(len(node_in_tree.gomory_cuts_lhs)):
                        curr_label.frac_coeffs_per_cut[i] = 0
                    curr_label.sequence = [pricing_network.source]
                    curr_label.length = 1
                    # 2.2.3 set initial start time distribution at source node
                    curr_label.start_time_pmf = {tour.leave_time: 1}
                    curr_label.start_time_cdf = {tour.leave_time: 1}
                    curr_label.start_time_cdf_per_task[pricing_network.source] = {tour.leave_time: 1}
                    # 2.2.4 initialize reduced cost for each possible skill composition
                    for skill_comps_cnts_id in pricing_network.skill_comps_cnts_ids:
                        curr_label.cost_per_skill_comp[skill_comps_cnts_id] = 0

                    # 2.2.5 extend label along the sequence, compute distributions and reduced cost
                    i = 0
                    while i < len(sequence) - 1:
                        curr_node = curr_label.get_last_node()  # get last node of current label (=path)
                        curr_task = curr_node
                        if i == 0:  # executing time of source node is zero
                            execution_time_head = 0
                        else:
                            execution_time_head = pricing_network.task_execution_times[curr_task]  # execution time of current last task

                        edge = (sequence[i], sequence[i + 1])
                        # skip arcs that are not in the pricing network (extension guaranteed to be infeasible)
                        if edge not in pricing_network.graph.edges:
                            curr_label = None
                            i = len(sequence) - 1

                        # perform extension
                        if curr_label is not None:
                            succ_node = edge[1]
                            # get relevant information regarding the tail node (earliest start, task name, execution time, task weight)
                            earliest_start_tail = pricing_network.earliest_starts[succ_node]
                            execution_time_tail = pricing_network.task_execution_times[succ_node]
                            tail_weight = pricing_network.weights[succ_node]
                            # extend
                            label_t_bin_pred = pricing_network.bin_per_instant[curr_label.median_finish]
                            new_label = curr_label.extend(succ_node, pricing_network.sink,
                                                          pricing_network.resources[curr_node, succ_node],
                                                          pricing_network.travel_times_per_bin[label_t_bin_pred],
                                                          pricing_network.min_travel_times_per_bin[label_t_bin_pred],
                                                          pricing_network.max_travel_times_per_bin[label_t_bin_pred],
                                                          pricing_network.quantile_travel_times_per_bin[label_t_bin_pred],
                                                          execution_time_head,
                                                          earliest_start_tail, pricing_network.latest_starts,
                                                          pricing_network.latest_finishes,
                                                          pricing_network.latest_finishes_viol,
                                                          tail_weight, execution_time_tail, pricing_network.alpha,
                                                          pricing_network.end_horizon,
                                                          pricing_network.task_resources_consd, pricing_network, mu,
                                                          delta, rho_gr,
                                                          rho_le, psi,
                                                          zeta_le, zeta_gr, False, t_max_le, t_max_gr,
                                                          skill_comp_cnt,
                                                          solve_as_dmp, node_in_tree, yuan_approach,
                                                          node_in_tree.forbidden_arcs, node_in_tree.forced_arcs,
                                                          pricing_network.source)
                            curr_label = new_label
                            # if arc is forbidden: break
                            if curr_label is not None:
                                if (edge[0], edge[1], new_label.quantile_case_finish) in node_in_tree.forbidden_arcs:
                                    curr_label = None
                                    i = len(sequence) - 1
                            # if extension is infeasible -> stop while loop and continue with the next tour
                            if curr_label is None:
                                i = len(sequence) - 1
                            else:
                                i += 1
                    # 2.2.6 for each skill comp. with negative reduced cost: add curr_label to neg_labels
                    if curr_label is not None:
                        if curr_label.sequence[-1] != "sink":  # quick sanity check
                            raise Exception("VND algorithm, delete operator: label does not end at sink")
                        # check cost for each skill comp
                        for skill_comp_cnt_id in curr_label.cost_per_skill_comp:
                            if curr_label.cost_per_skill_comp[skill_comp_cnt_id] < eps_col_neg:
                                # set skill composition and store
                                neg_label = curr_label.clone(node_in_tree)
                                neg_label.min_skill_comp_cnt = pricing_network.skill_comps_cnts_ids[skill_comp_cnt_id]
                                neg_label.min_skill_comp = pricing_network.skill_comps_ids[skill_comp_cnt_id]
                                neg_label.cost = min(neg_label.cost_per_skill_comp.values())
                                neg_labels.append(neg_label)

    # 3. postprocessing: remove duplicate labels
    # 3.1 remove all labels that are equal to the greedy label (if greedy label is not None)
    label_idxs_to_remove = []
    for greedy_label in greedy_labels:
        for i in range(len(neg_labels)):
            label_i = neg_labels[i]
            if label_i.sequence == greedy_label.sequence and label_i.start_time_from_depot == greedy_label.start_time_from_depot:
                if label_i.formation == greedy_label.formation and label_i.min_skill_comp_cnt == greedy_label.min_skill_comp_cnt:
                    label_idxs_to_remove.append(i)
    # remove duplicate indices in label_idxs_to_remove (e.g. happens when 3 tours with 2 tasks all remove their last
    # task and end up creating the same single-task route) and then sort list in descending order
    label_idxs_to_remove = sorted(list(set(label_idxs_to_remove)), reverse = True)
    # pop duplicate labels of greedy label
    for idx in label_idxs_to_remove:
        neg_labels.pop(idx)

    # 3.2 remove all duplicate labels (e.g. tours t1= [dep, task_1, task_2, task_3, dep] and t2=[dep, task_1, dep]
    # can yield duplicates when task_3 is deleted from t2 and task_2 is inserted into t2 after task_1)
    label_idxs_to_remove = []
    for i in range(len(neg_labels) - 1):
        label_i = neg_labels[i]
        #  check if label has a duplicate VND label
        for j in range(i+1, len(neg_labels)):
            label_j = neg_labels[j]
            if label_i.sequence == label_j.sequence and label_i.start_time_from_depot == label_j.start_time_from_depot:
                if label_i.formation == label_j.formation and label_i.min_skill_comp_cnt == label_j.min_skill_comp_cnt:
                    label_idxs_to_remove.append(j)
    # remove duplicate indices in label_idxs_to_remove and then sort list in descending order
    label_idxs_to_remove = sorted(list(set(label_idxs_to_remove)), reverse = True)
    # pop duplicate labels
    for idx in label_idxs_to_remove:
        neg_labels.pop(idx)

    # 3.3 remove labels that are equal to forbidden tours
    label_idxs_to_remove = []
    for i in range(len(neg_labels)):
        label = neg_labels[i]
        for forb_tour in node_in_tree.forbidden_tours:
            if (label.start_time_from_depot == forb_tour.leave_time and label.sequence[1:-1] == forb_tour.tasks and
                    pricing_network.formation_id == forb_tour.formation_id):
                if label.min_skill_comp_cnt == forb_tour.skill_comp_cnt:
                    label_idxs_to_remove.append(i)
    label_idxs_to_remove = sorted(label_idxs_to_remove, reverse = True)
    # pop duplicate labels
    for idx in label_idxs_to_remove:
        neg_labels.pop(idx)

    return neg_labels


def find_columns_heuristics(pricing_networks, forbidden_tours_per_formation, mu, delta, rho_gr, rho_le, psi, zeta_le, zeta_gr,
                            t_max_le, t_max_gr, solve_as_dmp, node_in_tree, yuan_approach, current_sol_tours, inst):
    """Find negative columns using greedy and VND heuristics. Only called when approach by Yuan et al. (2015) is used.
    Also, in line with Yuan et al. (2015), greedy heuristic is only called at the root node.

    Parameters
    -----------
    pricing_networks: list
        List of workers.pricing.graph.PricingNetwork objects
    forbidden_tours_per_formation: dict
        Maps formations to the list of tours that use the formation and are forbidden (by branching on tours)
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
    solve_as_dmp: bool
        Indicates if the current node is solved as a DMP (True) or an AMP (False).
    node_in_tree: GH_node
        Current node in the branching tree
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used
    current_sol_tours: list
        List of GH_tours at the current node that were selected in the current LP relaxation's solution (i.e., whose
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.

    Returns
    -------
    neg_tours: list
        All negative tours (GH_tour objects) found by the VND algorithm
    """
    # 0. sanity check: heuristic column sarch only correctly implemented if yuan_approach AND solve_as_dmp are True
    if not yuan_approach or not solve_as_dmp:
        raise Exception(f"Heuristic column serch only implemented for yuan_approach = True AND solve_as_dmp = True.")

    # 1. initialization
    # 1.1 datastructures to store negative labels and tours
    neg_heur_labels_per_formation = {}
    neg_tours = []

    # 2. apply greedy and VND heuristics to each formation
    for formation_id in inst.formations:
        # 2.1 preprocessing
        # 2.1.1 get pricing network and set dummy value for skill comps
        pricing_network = pricing_networks[formation_id]
        pricing_network.set_resources_consumption(inst, forbidden_tours_per_formation[formation_id])
        pricing_network.task_resources_consd = []  # reset tasks for which resources will be considered in labeling algorithm
        neg_heur_labels_per_formation[pricing_network.formation_id] = []
        # 2.1.2 get indices of forbidden tours (needed for label extensions during VND algorithm)
        forb_tour_idxs = list(range(len(pricing_network.forb_tours))) # also get their indices

        greedy_label = None
        # 2.2 if current node is the root: call greedy for each skill composition
        if node_in_tree.is_root:
            if pricing_network.tasks != []:  # only call greedy heuristic when at least one task is present in the pricing network
                for skill_comp_cnt_id in pricing_network.skill_comps_cnts_ids:
                    skill_comps_cnt = pricing_network.skill_comps_cnts_ids[skill_comp_cnt_id]
                    skill_comp = pricing_network.skill_comps_ids[skill_comp_cnt_id]

                    # 2.2.1 get all initial labels, i.e. single-task labels that leave the sink, and sort them w.r.t their reduced cost
                    # compute initial labels
                    _, all_initial_labels, _ = create_initial_labels(pricing_network, forb_tour_idxs, mu, delta, rho_gr,
                                                                       rho_le, psi, zeta_le, zeta_gr,
                                                                       t_max_le, t_max_gr, skill_comps_cnt,
                                                                       solve_as_dmp, node_in_tree, yuan_approach)

                    # 2.2.2 apply greedy heuristic
                    if len(all_initial_labels) > 0:
                        greedy_label = find_column_greedy(all_initial_labels, pricing_network, node_in_tree, mu, delta, rho_gr,
                                                          rho_le, psi,
                                                          zeta_le, zeta_gr, t_max_le, t_max_gr, skill_comps_cnt,
                                                          skill_comp, solve_as_dmp, yuan_approach)
                        # Note: greedy heuristic always returns at most one label
                        if greedy_label is not None:
                            if greedy_label.cost < eps_col_neg * 10:
                                neg_heur_labels_per_formation[pricing_network.formation_id].append(greedy_label)

        # 2.3 apply VND to all basis columns that use the current formation
        # 2.3.1 set dummy value for skill_comps_cnt (will only be used to compute label.cost values, but we will
        # only use label.cost_per_skill_comp anyway)
        skill_comps_cnt = [pricing_network.default_comp_cnt]
        # 2.3.2 apply VND
        heur_labels = find_column_vnd(current_sol_tours, pricing_network, node_in_tree, forb_tour_idxs, mu, delta,
                                      rho_gr, rho_le, psi, zeta_le,
                                      zeta_gr, t_max_le, t_max_gr, skill_comps_cnt[0], solve_as_dmp,
                                      yuan_approach, neg_heur_labels_per_formation[pricing_network.formation_id])

        # 2.3.1 add all negative columns
        # Note: reduced cost negativity was already checked in find_column_vnd, so we don't check it again here
        if heur_labels != []:
            for label in heur_labels:
                neg_heur_labels_per_formation[pricing_network.formation_id].append(label)


        # 2.4 create tour objects for all negative labels
        ex_hashes = [tour.get_hash() for tour in node_in_tree.tours] # hashes of all already-existing tours
        for label in neg_heur_labels_per_formation[pricing_network.formation_id]:
            if label.cost < eps_col_neg / 10:
                new_tour = pricing_network.build_tour(label, inst)
                new_tour_hash = new_tour.get_hash()
                # 2.4.1 skip label if its already present in the node's column pool or in the list of negative tours
                # Note: this can happen due to floating point imprecision
                if new_tour_hash not in ex_hashes:
                    neg_tours.append(new_tour)
                    ex_hashes.append(new_tour_hash)


    return neg_tours
