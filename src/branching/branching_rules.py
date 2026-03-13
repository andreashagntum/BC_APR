"""This script contains functions to compute branching decision. Currently, the following branching decisions are
implemented:
- branching on task finish times
- branching on no. of tours at a given time
- branching on arcs
- branching on variables
"""

import math
from numpy import std, median, mean
from config.config import *

def find_most_std_frac_task_finish_time_alt(node_sol, node_tours):
    """Finds a) the task i such that the no. of distinct finish times of task i of fractional tours is maximal (1st criteria)
    and the std. of finish times of task i of fractional tours covering i is largest (2nd criteria) and
    b) the median worst case finish time of all fractional tours covering task i.
    If for each task, either no fractional tour is found or all tours that visit the task finish at the same time instant,
    this strategy does not procedure a nontrivial branch. In such a case, a None tuple is returned.
    For additional details, see paragraph Branching on Task Finish Times in Section 5.5 of Hagn et al. (2026).

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects

    Returns
    -------
    opt_task: str
        Name of task to branch on
    opt_time_instant: int
        Gamma-quantile case finish time of task to branch on
    tours_to_remove: list
        List of tours that visit opt_task. These tours will have to be removed in exactly one of the to-be-created child
        nodes.
    tour_idxs_to_remove: list
        List of indices in node_tours that correspond to objects in tours_to_remove

    """

    # 1. for each task: count no. of fractional tours covering task
    tours_per_task = {}
    limits_per_task = {}        # key: task name, value: dict containing earliest and latest worst-case finish time of selected tours covering task
    finish_times = {}
    for idx in node_sol:
        if not idx == 0 and node_sol[idx] > eps_global / 100 and node_sol[idx] < 1 - eps_global / 100: # select all tours with fractional lambda value (except fake tour)
            tour = node_tours[idx]
            for task in tour.tasks:
                quantile_finish_time = node_tours[idx].quantile_finish_time[task]
                if task not in finish_times:
                    tours_per_task[task] = []
                    limits_per_task[task] = {"earliest": math.inf, "latest": 0}
                    finish_times[task] = [0, []]
                tours_per_task[task].append(idx)
                finish_times[task][1].append(node_tours[idx].quantile_finish_time[task])
                limits_per_task[task]["earliest"] = min(limits_per_task[task]["earliest"], quantile_finish_time)
                limits_per_task[task]["latest"] = max(limits_per_task[task]["latest"], quantile_finish_time)

    # 2. get unique finish times and sort descending w.r.t no. of different finish times
    for task in finish_times:
        finish_times[task][0] = len(set(finish_times[task][1]))
    # 2.1 sort tasks w.r.t their no. of distinct finish times
    finish_times = sorted(finish_times.items(), key = lambda x: x[1][0], reverse = True)
    # 2.2 if no fractional tours found or all tasks have at most one unique finish time: return empty
    if len(finish_times) == 0 or finish_times[0][1][0] <= 1:
        return None, None, None, None
    # 2.3 get all tasks with max. no. of distinct finish time and sort them w.r.t their standard deviation
    max_no_finish_times = finish_times[0][1][0]
    max_finish_times = [(task, tup) for (task, tup) in finish_times if tup[0] == max_no_finish_times]
    # 2.4 sort tasks w.r.t standard deviation of finish times
    max_finish_times = sorted(max_finish_times, key = lambda x: std(x[1][1]), reverse = True)


    # 3. return task with maximum std. of task finish times of fractional tours and median worst-case finish time
    opt_task = max_finish_times[0][0]
    # get task finish time in all task, then calculate median of these values
    finish_times_task = max_finish_times[0][1][1]
    opt_time_instant = math.floor(median(finish_times_task))
    # if optimal time instant is equal to smallest or largest finish time: use floor(mean(..)) instead
    # this way, cuts can never be generated twice
    if opt_time_instant == min(finish_times_task) or opt_time_instant == max(finish_times_task):
        opt_time_instant = math.floor(mean(finish_times_task))

    # 4. return optimal task, time instant and all tours covering the optimal task
    tours_to_remove = []
    tour_idxs_to_remove = []
    for i in range(len(node_tours)):
        tour = node_tours[i]
        if opt_task in tour.tasks and i != 0:       # index 0 is always the fake tour, we do not remove it to always ensure feasibility
            tours_to_remove.append(tour)
            tour_idxs_to_remove.append(i)

    return opt_task, opt_time_instant, tours_to_remove, tour_idxs_to_remove

def find_most_freq_frac_instant(node_sol, node_tours):
    """Finds the time instant t such that the no. of tours at time t with fractional value lambda is largest.
    If for each time instant t, the sum of all lambda variables of tours active at time t is integer, this strategy
    does not procedure a nontrivial branch. In such a case, a None tuple is returned.

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects

    Returns
    -------
    t: int
        Time instant to branch on
    l: float
        Sum of lambda values of all tours active at time t
    no_tours_at_instant_t: int | None
        No. of tours in current solution that are active at time t, None if no nontrivial branch has been found
    """

    # 1. for each time instant: get no. of tours and the sum of lambdas
    no_frac_tours_per_instant = {}
    sum_var_per_instant = {}
    for i in node_sol:
        if node_sol[i] > eps_global / 10:
            for t in list(range(node_tours[i].leave_time, node_tours[i].quantile_return_time)):
                if t not in sum_var_per_instant:
                    sum_var_per_instant[t] = 0.0
                if t not in no_frac_tours_per_instant:
                    no_frac_tours_per_instant[t] = 0
                l = node_sol[i]
                frac = min(l - math.floor(l), math.ceil(l) - l)
                sum_var_per_instant[t] += node_sol[i]
                if frac > eps_global / 10:
                    no_frac_tours_per_instant[t] += 1


    # 2. remove all time instants for which sum of tour lambdas is not fractional
    frac_var_per_instant = {}
    for t in sum_var_per_instant:
        l = sum_var_per_instant[t]
        frac = min(l - math.floor(l), math.ceil(l) - l)
        if frac < eps_global / 10:
            no_frac_tours_per_instant.pop(t)
        else:       # store fractional value of lambda of tours at time t
            frac_var_per_instant[t] = frac

    # 3. if all time instants have an integer sum of tour lambdas: return empty
    if len(no_frac_tours_per_instant) == 0:
        return None, None, None

    # 4. else: get time instant with largest no. of fractional tours active and most fractional lambda sum
    # 4.1 sort time instants by no. of tours active
    no_frac_tours_per_instant = sorted([i for i in no_frac_tours_per_instant.items()], key = lambda x: x[1], reverse = True)
    # get time instants with largest no. of active tours
    (t, no_tours) = no_frac_tours_per_instant[0]
    # convert data to dict for easier information retrieval
    no_frac_tours_per_instant = {k:v for (k,v) in no_frac_tours_per_instant}

    # 4.2 get all time instants with "no_tours" many tours active
    maximum_tours = [(t,no_frac_tours_per_instant[t], frac_var_per_instant[t], sum_var_per_instant[t])]
    for t_new in no_frac_tours_per_instant:
        if t_new == t:
            continue
        no_frac_tours_new = no_frac_tours_per_instant[t_new]
        if no_frac_tours_new == no_tours:
            maximum_tours.append((t_new, no_frac_tours_new, frac_var_per_instant[t_new], sum_var_per_instant[t_new]))
        else:
            break
    # 4.3 sort all tours with maximum size descending w.r.t their fractionality
    maximum_tours = sorted(maximum_tours, key = lambda x: x[2], reverse = True)

    # 4.4 return data of best time instant
    t, no_tours_at_instant_t, l = maximum_tours[0][0], maximum_tours[0][1], maximum_tours[0][3]

    return l, t, no_tours_at_instant_t

def find_most_frac_arc(node_sol, node_tours):
    """Get all arcs used by node_sol and their value. Return the most fractional arc, its values, and all tours
    using said arc.

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value
    node_tours: list
        Corresponding list of GH_tour objects

    Returns
    -------
    arc_value: float
        Absolute difference between value of the most fractional arc and 0.5 (arc_value < 0.5 implies the arc is fractional)
    most_frac_arc: tuple
        Arc whose lambda value is most fractional
    sol_tours_using_arc: list
        List of GH_tours in node_tours that use most_frac_arc
    forbidden_tour_idxs_to_remove: list
        List of indices of tours that need to be removed when most_frac_arc is forbidden
    forced_tour_idxs_to_remove: list
        List of indices of tours that need to be removed when most_frac_arc is forced
    """

    # 1. get all tours selected by the current solution
    selected_sol = {idx: node_sol[idx] for idx in node_sol if node_sol[idx] > eps_global}
    # 2. get arc usages
    arc_values = {}
    sol_tours_using_arc = {} # keys: arcs, values: indices of all tours part of the current solution that use the arc
    # 2.1 for each selected tour: get all used arcs and add their respective value to arc_values
    for tour_idx in selected_sol.keys():
        tour = node_tours[tour_idx]
        val = node_sol[tour_idx]
        sequence = ["source"] + tour.tasks + ["sink"]       # include depot into task sequence
        for i in range(len(sequence)-1):
            pred = sequence[i]
            succ = sequence[i+1]
            quantile_finish = tour.quantile_finish_time[sequence[i+1]]  # worst-case finish of successor task
            if (pred, succ, quantile_finish) not in arc_values:
                arc_values[(pred, succ, quantile_finish)] = 0
                sol_tours_using_arc[(pred, succ, quantile_finish)] = []
            sol_tours_using_arc[(pred, succ, quantile_finish)].append(tour_idx)
            arc_values[(pred, succ, quantile_finish)] += val

    # 2.2 sort arc_values ascending w.r.t fractionality and return most fractional value
    for arc in arc_values:  # transform arc values to their absolute fractionality value
        arc_values[arc] = abs(0.5 - arc_values[arc])
    arc_values = dict(sorted(arc_values.items(), key = lambda x: x[1]))
    most_frac_arc = list(arc_values.keys())[0]



    # 2.3 get all tours which select the given arc (those need to be removed in the right child node)
    forbidden_tours_to_remove = []
    forbidden_tour_idxs_to_remove = []
    for tour_idx in range(len(node_sol)):
        tour = node_tours[tour_idx]
        sequence = ["source"] + tour.tasks + ["sink"]
        for i in range(len(sequence)-1):
            if (sequence[i], sequence[i+1]) == most_frac_arc[:2] and tour.quantile_finish_time[sequence[i+1]] == most_frac_arc[2]:
                forbidden_tour_idxs_to_remove.append(tour_idx)
                forbidden_tours_to_remove.append(tour)
                break

    # 2.4 get all tours which contain the given successor node of the arc, but do not use the arc (these need to
    # be removed in the left child node)
    forced_tours_to_remove = []
    forced_tour_idxs_to_remove = []
    for tour_idx in range(len(node_sol)):
        tour = node_tours[tour_idx]
        sequence = ["source"] + tour.tasks + ["sink"]
        for i in range(len(sequence)-1):
            # case a) destination matches, but origin does not [not relevant for last arc of each tour]
            if sequence[i+1] == most_frac_arc[1] and sequence[i] != most_frac_arc[0]:
                if sequence[i+1] != "sink":
                        forced_tour_idxs_to_remove.append(tour_idx)
                        forced_tours_to_remove.append(tour)
                        break
            # case b) origin matches, but destination does not [not relevant for first arc of each tour]
            elif sequence[i] == most_frac_arc[0] and sequence[i+1] != most_frac_arc[1]:
                if sequence[i] != "source":
                    forced_tour_idxs_to_remove.append(tour_idx)
                    forced_tours_to_remove.append(tour)
                    break
            # case c) origin and destination match, but worst-case finish does not
            elif sequence[i] == most_frac_arc[0] and sequence[i+1] == most_frac_arc[1]:
                if tour.quantile_finish_time[sequence[i+1]] != most_frac_arc[2]:
                    forced_tour_idxs_to_remove.append(tour_idx)
                    forced_tours_to_remove.append(tour)
                    break

    return (arc_values[most_frac_arc], most_frac_arc, sol_tours_using_arc, forbidden_tour_idxs_to_remove,
            forced_tour_idxs_to_remove)

def find_most_frac_variable(node_sol):
    """Find most fractional tour variable.

    Parameters
    ----------
    node_sol: dict
        Maps tour indices to their lambda value

    Returns
    -------
    best_index: int
        Index of most fractional variable in node_sol
    """

    node_frac = dict((k, abs(v - 0.5)) for (k, v) in node_sol.items() if v > eps_global / 100 and v < 1 - eps_global / 100)
    best_index = max(node_frac, key=node_frac.get)
    return best_index
