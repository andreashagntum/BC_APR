"""This script contains function to generate child nodes from a parent node, given a branching decision.
Currently, all branching rules defined in src.branching.branching_rules are supported.
"""
from src.branching.branching_rules import *
import bisect

def branch(node, node_tours, solution, node_sol, tree, node_hashes, branch_on_task_finish_times, yuan_approach):
    """Branch on current solution at current node. Searches for a nontrivial branch and applies it. Branching strategies
    are searched in the following sequence:
    1. if yuan_approach = False (i.e. approach by Hagn et al. (2026) is used):
        1.1 branching on task finish times
        1.2 branching on no. of tours at a given time
        1.3 branching on variables
    2. if yuan_approach = True (i.e. approach by Yuan et al. (2015) is used):
        2.1 branching on arcs
        2.2 branching on variables

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    node_tours: list
        List of GH_tour tours at current node
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics
    node_sol: dict
        Maps tour indices to their lambda value
    tree: list
        Current branching tree, list of GH_node objects
    node_hashes: list
        List of hashes of all nodes explored so far (used for sanity checks)
    branch_on_task_finish_times: bool
        Indicates if branching on task finish time should be used.
    yuan_approach: bool
        Indicates if the approach by Yuan et al. (2015) should be used. If set to True, overrides no_gomory_cuts = 0,
            branch_on_task_finish_times = False, and use_dmp = True.

    Returns
    -------
    node: GH_node
        Current node in the branching tree
    tree: list
        Current branching tree, list of GH_node objects
    solution: GH_solution
        Solution object containing the best integer solution found alongside some statistics
    node_hashes: list
        List of hashes of all nodes explored so far (used for sanity checks)

    """

    # 1. if approach by Yuan et al. (2015) should be used: firs try to branch on arcs, use variable branching
    # as fallback
    if yuan_approach:
        # 1.1 get value of each arc based on current solution and find most fractional value
        (arc_value, arc, sol_tours_using_arc, forbidden_tour_idxs_to_remove,
         forced_tour_idxs_to_remove) = find_most_frac_arc(node_sol, node_tours)
        if arc_value > 0.4999:
            # 1.2 if no non-trivial branch has been found: use fallback-branching on variables
            # get most fractional tour variable
            branch_var_index = find_most_frac_variable(node_sol)
            fixed_tour = node_tours[branch_var_index]
            print(f"fallback-branching on tour index {branch_var_index}")
            left_child, right_child = generate_children_b_on_variable(node, fixed_tour, branch_var_index)
            solution.count_branch_on_variable += 1
        else:
            # 1.3 branch on it, right child: arc forbidden, left child: arc forced
            print(f"branching on arc {arc}")
            left_child, right_child = generate_children_b_on_arcs(node, arc, forbidden_tour_idxs_to_remove,
                                                                  forced_tour_idxs_to_remove)
            solution.count_branch_on_arcs += 1

    # 2. approach by Hagn et al. (2026): first branch on task finish times, then on tours, then on variables
    else:
        # 2.1 branch on task finish times
        if branch_on_task_finish_times:
            task, t_tilde, tours_to_remove, tour_idxs_to_remove = find_most_std_frac_task_finish_time_alt(
                node_sol, node_tours, node.inst)
        else:
            task = None
        # 2.2 if nontrivial branch was found: branch
        if task != None:
            left_child, right_child = generate_children_b_on_task_finish_times(node, task, t_tilde,
                                                                               tour_idxs_to_remove)
            print(f"branching on task {task} and worst-case finish time {t_tilde}")
            solution.count_branch_on_tasks += 1
        # 2.3 else: check if branching on vehicles tour counts is possible
        else:
            l, t_tilde, no_tours_t_tilde = find_most_freq_frac_instant(node_sol, node_tours)
            # 2.3.1 if nontrivial branch was found: branch
            if l != None:
                print(
                    f"branching on instant {t_tilde} with {no_tours_t_tilde} fractional tours and a lambda sum of {l}")
                left_child, right_child = generate_children_b_on_tour_counts(node, l, t_tilde)
                solution.count_branch_on_vehicles += 1

            # 2.3.2 else: fallback branch on variables
            else:
                # get most fractional tour variable
                branch_var_index = find_most_frac_variable(node_sol)
                fixed_tour = node_tours[branch_var_index]
                print(f"fallback-branching on tour index {branch_var_index}")
                # generate child/branched nodes using branching rule 2.
                left_child, right_child = generate_children_b_on_variable(node, fixed_tour,
                                                                          branch_var_index)
                solution.count_branch_on_variable += 1

    # 3. validity check: check if node with current branching constraints has been generated before
    left_child_hash = left_child.get_hash()
    right_child_hash = right_child.get_hash()
    if left_child_hash in node_hashes or right_child_hash in node_hashes:
        raise Exception("Node with same branching constraints already generated")
    else:
        node_hashes.append(left_child_hash)
        node_hashes.append(right_child_hash)

    # 4. set parent and child node attributes, add child nodes to tree
    left_child.parent = node
    right_child.parent = node
    node.left_child = left_child
    node.right_child = right_child
    node.left_child.sibling = right_child
    node.right_child.sibling = left_child
    bisect.insort(tree, right_child)  # tree.append(right_child) also possible
    bisect.insort(tree, left_child)  # tree.append(left_child) also possible
    tree.sort(key=lambda x: x.parent.lb, reverse=False)

    return node, tree, solution, node_hashes


def generate_children_b_on_task_finish_times(node, task, t_tilde, tour_idxs_to_remove):
    """Create child nodes by branching on gamma-quantile case finish times of a given task.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    task: str
        Task to branch on
    t_tilde: int
        Finish time to branch on
    tour_idxs_to_remove: list
        List of indices of tours that need to be removed in exactly one of the child nodes

    Returns
    -------
    left_child: GH_node
        Child node of node with quantile-case finish time of task enforced to be <= t_tilde
    right_child: GH_node
        Child node of node with quantile-case finish time of task enforced to be > t_tilde
    """

    # 1. clone parent node
    left_child = node.clone()
    right_child = node.clone()

    # 2. update branching constraints (left child gets <= inequality, right child >= inequality
    left_child.t_max_le[task] = t_tilde
    right_child.t_max_gr[task] = t_tilde + 1

    # 3. remove infeasible tours from child nodes
    removed_idxs_per_child = {left_child: [],
                              right_child: []}  # only contains list of indices that have been removed in each child node
    for idx in tour_idxs_to_remove:
        tour = node.tours[idx]
        if idx == 0:
            continue
        if tour.quantile_finish_time[task] > t_tilde:
            removed_idxs_per_child[left_child].append(idx)
            left_child.tours.remove(tour)
        elif tour.quantile_finish_time[task] <= t_tilde:
            removed_idxs_per_child[right_child].append(idx)
            right_child.tours.remove(tour)

    # 4. adjust gomory cut dictionaries by shifting tour indices to the new respective index (after removing
    # infeasible tours in child nodes)
    adjust_gomory_cut_indices([left_child, right_child], removed_idxs_per_child)

    return left_child, right_child

def generate_children_b_on_tour_counts(node, l, t_tilde):
    """Create child nodes by branching on no. of tours at specific time.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    l: float
        Sum of lambda values of all tours active at time t_tilde
    t_tilde: int
        Time instant to branch on

    Returns
    -------
    left_child: GH_node
        Child node of node with sum of lambdas at time t_tilde enforced to be <= floor(l)
    right_child: GH_node
        Child node of node with sum of lambdas at time t_tilde enforced to be > ceil(l)
    """

    # 1. clone parent node
    left_child = node.clone()
    right_child = node.clone()

    # add branching constraints to t_le and t_gr (left child gets <= inequality, right child >= inequality)
    left_child.t_le[t_tilde] = math.floor(l)
    right_child.t_gr[t_tilde] = math.ceil(l)

    # 2. increase tree depth
    left_child.depth = node.depth + 1
    right_child.depth = node.depth + 1

    return left_child, right_child

def generate_children_b_on_arcs(node, arc, forbidden_tour_idxs_to_remove, forced_tour_idxs_to_remove):
    """Generate child nodes by enforcing/forbidding a given arc.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    arc: tuple
        Arc that is to be forced or forbidden
    forbidden_tour_idxs_to_remove: list
        List of indices of tours that need to be removed when arc is forbidden
    forced_tour_idxs_to_remove: list
        List of indices of tours that need to be removed when arc is forced

    Returns
    -------
    left_child: GH_node
        Child node of node with arc forced
    right_child: GH_node
        Child node of node with arc forbidden

    """
    # 1. clone parent node
    left_child = node.clone()       # arc is forced
    right_child = node.clone()      # arc is forbidden

    # 2. remove infeasible tours from right child node
    removed_idxs_per_child = {right_child: [], left_child: []}  # only contains list of indices that have been removed in each child node
    for idx in forbidden_tour_idxs_to_remove:
        tour = node.tours[idx]
        if idx == 0:    # do not remove the fake tour
            continue
        # remove tour
        removed_idxs_per_child[right_child].append(idx)
        right_child.tours.remove(tour)

    # 3. remove infeasible tours from left child node
    for idx in forced_tour_idxs_to_remove:
        tour = node.tours[idx]
        if idx == 0:    # do not remove the fake tour
            continue
        # remove tour
        removed_idxs_per_child[left_child].append(idx)
        left_child.tours.remove(tour)


    # 4. add arc to the list of forced/forbidden arcs at the left/right child node
    if arc in node.forbidden_arcs or arc in node.forced_arcs: # quick sanity check
        raise Exception(f"Tried branching on an arc that is already forced or forbidden.")
    left_child.forced_arcs.append(arc)
    right_child.forbidden_arcs.append(arc)


    # 4. adjust gomory cut dictionaries by shifting tour indices to the new respective index (after removing infeasible tours in child nodes)
    # note: this is not necessary as branching on arcs is only used by Yuan et al., who do not use any cutting planes.
    if node.gomory_cuts_lhs:
        raise Exception("Branching on arcs not supported when gomory cuts are enabled.")

    return left_child, right_child

def generate_children_b_on_variable(node, fixed_tour, fixed_tour_idx):
    """Create child nodes by branching on a given tour fixed_tour.

    Parameters
    ----------
    node: GH_node
        Current node in the branching tree
    fixed_tour: GH_tour
        Arc that is to be forced or forbidden
    fixed_tour_idx: int
        Index of fixed_tour in current tour set node.tours

    Returns
    -------
    left_child: GH_node
        Child node of node where fixed_tour is forced
    right_child: GH_node
        Child node of node where fixed_tour is forbidden
    """
    # right child -> tour forbidden

    # left tour list -> exclude tours containing tasks performed in one of the fixed tours
    # right tour list -> exclude tours fixed in previous iterations/parent nodes

    # 1. check for errors: if fixed_tour is already in forced_tours or forbidden_tours, there is an error
    for tour in node.forced_tours:
        if tour.get_hash() == fixed_tour.get_hash():
            raise Exception("The tour " + tour.to_string() + " is already forced")

    for tour in node.forbidden_tours:
        if tour.get_hash() == fixed_tour.get_hash():
            raise Exception("The tour " + tour.to_string() + " is forbidden")

    # 2. save forced tasks (tasks in tour that is to be fixed) in list forced_tasks
    forced_tasks = []
    for task in fixed_tour.tasks:
        if task not in forced_tasks:
            forced_tasks.append(task)
        else:  # this should not be possible
            raise Exception("task forced in more than one tour")

    # 3. clone parent node and force/forbid fixed_tour
    left_child = node.clone()
    right_child = node.clone()


    # 3.1 remove fixed_tour from list of available tours at child nodes
    for tour in left_child.tours:
        if tour.get_hash() == fixed_tour.get_hash():
            tour_left_child_to_remove = tour
            break
    left_child.tours.remove(tour_left_child_to_remove)
    right_child.tours.remove(tour_left_child_to_remove)  # tour shares the same index in both child nodes

    # 3.2 remove forced/forbidden tour from all gomory cuts in left and right child
    for i in range(len(left_child.gomory_cuts_lhs)):
        if fixed_tour_idx in left_child.gomory_cuts_lhs[i]:
            # # for left child: right hand side must be adjusted by alpha[forced_tour] as its lambda-value is forced to 1
            left_child.gomory_cuts_rhs[i] -= left_child.gomory_cuts_lhs[i][fixed_tour_idx]
            left_child.gomory_cuts_lhs[i].pop(fixed_tour_idx)
            # for right child: right hand side remains the same as forbidding tour equals lambda[forced_tour] = 0
            right_child.gomory_cuts_lhs[i].pop(fixed_tour_idx)
        # adjust indices of all following tours
        # this procedure is identical for both left and right child. Note that both child nodes have the same gomory cut LHS's.
        new_lhs = {}
        for idx in left_child.gomory_cuts_lhs[i]:
            if idx > fixed_tour_idx:
                new_lhs[idx - 1] = left_child.gomory_cuts_lhs[i][idx]
            else:
                new_lhs[idx] = left_child.gomory_cuts_lhs[i][idx]
        left_child.gomory_cuts_lhs[i] = new_lhs.copy()
        right_child.gomory_cuts_lhs[i] = new_lhs.copy()

    # 3.3 left child: remove all tours that share tasks with fixed_tour and adjust gomory cuts indices
    incompatible_tours_left = []
    incompatible_tour_idxs_left = []
    # 3.3.1 get all incompatible tours and their indices
    for idx in range(len(left_child.tours)):
        tour = left_child.tours[idx]
        if not is_tour_suitable_for_left_child(tour, forced_tasks) and not tour == left_child.tours[0]: # fake tour always at index 0
            incompatible_tours_left.append(tour)
            incompatible_tour_idxs_left.append(idx)
    # 3.3.1 remove incompatible tours
    for tour in incompatible_tours_left:
        left_child.tours.remove(tour)
    # 3.3.2 adjust gomory cut indices
    for i in range(len(left_child.gomory_cuts_lhs)):
        adjustment_per_idx = {}
        for idx in left_child.gomory_cuts_lhs[i]:
            adjustment_per_idx[idx] = 0
        # remove largest index first
        for j in range(len(incompatible_tour_idxs_left) - 1, -1, -1):
            idx_to_remove = incompatible_tour_idxs_left[j]
            # remove index from cut if it has a nontrivial coefficient (i.e. if it is contained in gomory_cuts_lhs[i]
            if idx_to_remove in left_child.gomory_cuts_lhs[i]:
                left_child.gomory_cuts_lhs[i].pop(idx_to_remove)
                adjustment_per_idx.pop(idx_to_remove)
            for idx_greater_j in [key for (key, val) in left_child.gomory_cuts_lhs[i].items() if key > idx_to_remove]:
                adjustment_per_idx[idx_greater_j] += 1
        # create new LHS for gomory cut and store it
        new_cut_lhs = {}
        for idx in adjustment_per_idx:
            new_cut_lhs[idx - adjustment_per_idx[idx]] = left_child.gomory_cuts_lhs[i][idx]
        # 4.2.4 store new LHS for gomory cut
        left_child.gomory_cuts_lhs[i] = new_cut_lhs.copy()

    # 3.4 append fixed tour to forced/forbidden tours
    left_child.forced_tours.append(fixed_tour)
    right_child.forbidden_tours.append(fixed_tour)
    for task in forced_tasks:
        left_child.forced_tasks.append(task)

    # increase node depth
    left_child.depth = node.depth + 1
    right_child.depth = node.depth + 1

    return left_child, right_child

def adjust_gomory_cut_indices(children, removed_idxs_per_child):
    """Adjust gomory cuts at a given node after tours have been removed, e.g., by branching on arcs or on
    task finish times.
    When child nodes are created, at least one tour is removed. GH_node.gomory_cuts_lhs maps the indices of objects
    in node.tours to their coefficients in the gomory cuts. Because the indices of objects in node.tours change when
    objects are removed, the keys of GH_node.gomory_cuts_lhs need to be updated as well. This is done in this function.

    Parameters
    ----------
    children: list
        List of child nodes for which gomory cut coefficients need to be adjusted.
    removed_idxs_per_child: dict
        Maps child nodes to list of indices of objects in child.parent.tours that were removed when creating child

    """
    for child in children:
        # 1. remove coefficients from removed tours from all relevant gomory cuts and adjust all other indices
        for i in range(len(child.gomory_cuts_lhs)):
            adjustment_per_idx = {}
            for idx in child.gomory_cuts_lhs[i]:
                adjustment_per_idx[idx] = 0
            # 2. remove largest index first
            for j in range(len(removed_idxs_per_child[child]) - 1, -1, -1):
                idx_to_remove = removed_idxs_per_child[child][j]
                # remove index from cut if it has a nontrivial coefficient (i.e. if it is contained in gomory_cuts_lhs[i]
                if idx_to_remove in child.gomory_cuts_lhs[i]:
                    child.gomory_cuts_lhs[i].pop(idx_to_remove)
                    adjustment_per_idx.pop(idx_to_remove)       # do not copy index j to new gomory LHS dict as corresponding tour has been removed
                # 3. for each index higher than j: remember that index value  must be reduced by 1
                for idx_greater_j in [key for (key,val) in child.gomory_cuts_lhs[i].items() if key > idx_to_remove]:
                    adjustment_per_idx[idx_greater_j] += 1
            # 4. create new LHS for gomory cut and store it
            new_cut_lhs = {}
            for idx in adjustment_per_idx:
                new_cut_lhs[idx - adjustment_per_idx[idx]] = child.gomory_cuts_lhs[i][idx]
            # 5. store new LHS for gomory cut
            child.gomory_cuts_lhs[i] = new_cut_lhs.copy()

    return


def is_tour_suitable_for_left_child(tour, forced_tasks):
    """Check if tour contains any forced tasks. Used for branching on tours.
    Parameters
    ----------
    tour: GH_tour
        Tour which should be checked
    forced_tasks: list
        List of forced tasks

    Returns
    -------
    is_suitable: bool
        True if tour contains any forced tasks, False otherwise
    """

    for task in tour.tasks:
        if task in forced_tasks:
            return False
    return True

