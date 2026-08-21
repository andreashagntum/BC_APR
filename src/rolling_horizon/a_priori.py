"""Functions to compute solutions in a rolling horizon framework, assuming no observations are exploited: whenever a segment
is solved and the next segment is treated, any uncertain parameters that have materialized before the new segment's start
are NOT restricted to their actual (observed) value; instead, replanning is performed using all distributional information
provided by the input JSON file.
This approach provides the advantage that planning only has to be done once, and then just applied to every single monte-carlo
sampled scenario of uncertain parameters. Its disadvantage is that planning can not be adjusted to the actually observed
values, potentially losing solution quality by that.
"""
import copy
from src.instance_loader.instance_loader import add_travel_times_per_bin_restricted
from src.rolling_horizon.monte_carlo import simulate, KnowledgeState
import numpy as np
from tqdm import tqdm

def adjust_inst_to_segment(inst, th_segment, travel_times_per_bin, sol_per_seg, lookahead, all_tours):
    """Adjust a (deepcopied version of the input) inst object according to a passed th_segment. The following changes are made:

    - inst.tasks is restricted to the set of tasks whose earliest start lie in th_segment
    - inst.begin_horizon and inst.end_horizon are adjusted based on task's earliest starts and latest extended finish times
    - unused formations are removed
    - time window-related values are restricted to the task subset
    - instants, mappings of bins to instants (and vice versa) are restricted to the considered segment
    - travel times are restricted to the considered segment

    Parameters
    ---------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    th_segment: tuple[int, int]
        First and last instant of the considered segment
    travel_times_per_bin: dict
        Maps time bins and edges to their distribution of probabilities
    sol_per_seg: dict[int, GH_solution]
        Maps prior segments to their GH_solution object
    lookahead: int
        Length of interval whose tasks will be considered during planning of each segment. Tasks that start after the segment
        end will be planned, but their solution will not be implemented and might be overwritten by the decision in the
        subsequent segment.
    all_tours: list[GH_tour]
        List of all tours (finished or unfinished) from previous segments
    """

    seg_start, seg_end = th_segment

    # 0. Get all tasks carried over from tours of prior segments, that are still active at the beginning of the considered segment
    inst.tours_from_prev_segs = [tour for tour in all_tours if (tour.leave_time < seg_start and tour.quantile_return_time > seg_start)]

    # 1. Get all tasks in considered segment
    inst.tasks = [task for task in inst.tasks if (inst.earliest_start[task] >= seg_start and inst.earliest_start[task] < seg_start + lookahead)]
    # 1.1 Also include tasks from tours from previous segments that have been finished or are currently active
    inst.tasks_from_prev_segs = []
    formation_id_per_active_task = {}
    tours_returning_to_depot = [] # list of tours that are active, but are currently returning to the depot
    for i in range(len(inst.tours_from_prev_segs)):
        tour = inst.tours_from_prev_segs[i]
        # 1.1.2 Store all tasks whose subsequent task is not finished yet (i.e., all tasks that are either already
        # finished or are currently being executed)
        curr_active = sorted([(node, qft) for (node, qft) in tour.quantile_finish_time.items() if
                              (qft is not None and qft > seg_start)], key = lambda x: x[1])
        # a) If only active task/node is the sink: do not store this tour as a tour from a previous segment
        # Instead, we adjust the workforce accordingly
        if curr_active[0][0] == "sink":
            tours_returning_to_depot.append(tour)
            continue
        # quick sanity check
        if len(curr_active) < 1:
            raise Exception("At least one task must be active per tour")
        # b) Else: proceed
        active_tasks = [task for task in tour.tasks if (tour.quantile_finish_time[task] is not None and
                                                        tour.quantile_finish_time[task] <= curr_active[0][1])]
        passive_tasks = [task for task in tour.tasks if (tour.quantile_finish_time[task] is None or
                                                         tour.quantile_finish_time[task] > curr_active[0][1])]
        inst.tasks = list(set(inst.tasks).union(set(passive_tasks)))
        inst.tasks_from_prev_segs += active_tasks
        # remove active tasks from inst.tasks: inst.tasks will be used downstream to derive tasks per formation, and
        # active tasks will be added to these lists manually way
        inst.tasks = list(set(inst.tasks).difference(set(active_tasks)))
        for task in active_tasks:
            formation_id_per_active_task[task] = tour.formation_id
        # 1.1.3 Store active tasks sequence in tour object, will be used downstream during pricing network setup
        tour.active_tasks = [task for task in tour.tasks if task in active_tasks]


    # 1.2 Remove tasks from active tours from inst.tasks: their ES might not have been reached yet,
    # but a team is already on their way to the task: in this case, we can't reschedule the ask
    inst.tasks = [task for task in inst.tasks if task not in inst.tasks_from_prev_segs]
    # 1.3 Store all tasks (active and non-active) to inst.all_tasks
    inst.all_tasks = inst.tasks + inst.tasks_from_prev_segs
    # 1.4 Remove tours currently returning to the depot from the list tours from the previous segment
    for tour in tours_returning_to_depot:
        inst.tours_from_prev_segs.remove(tour)
    # 1.5 Update tours from the last segments by removing the tours that are not yet finished and are subject to change
    # in the current segment
    all_tours = [tour for tour in all_tours if (tour not in inst.tours_from_prev_segs and tour.leave_time < seg_start)]

    # quick sanity check
    if len(set(inst.tasks)) != len(inst.tasks):
        raise Exception("Task set contains duplicate tasks after adjusting instant to segment.")

    # 2. Restrict all sets relating to tasks
    # 2.1 Time windows
    inst.earliest_start = {task: inst.earliest_start[task] for task in inst.all_tasks}
    inst.latest_start = {task: inst.latest_start[task] for task in inst.all_tasks}
    inst.earliest_finish = {task: inst.earliest_finish[task] for task in inst.all_tasks}
    inst.latest_finish = {task: inst.latest_finish[task] for task in inst.all_tasks}
    inst.latest_start_viol = {task: inst.latest_start_viol[task] for task in inst.all_tasks}
    inst.latest_finish_viol = {task: inst.latest_finish_viol[task] for task in inst.all_tasks}

    # 2.2 Modes, weights, task locations, and tasks per formation
    inst.modes = {task: inst.modes[task] for task in inst.all_tasks}
    inst.modes_with_domination = {task: inst.modes_with_domination[task] for task in inst.all_tasks}
    inst.weights = {task: inst.weights[task] for task in inst.all_tasks}
    inst.task_locations = {task: inst.task_locations[task] for task in inst.all_tasks}
    # 2.2.1 For tasks of active tours that have already been executed or are currently being executed: only copy single
    # formation
    for (task, formation_id) in formation_id_per_active_task.items():
        assert formation_id in inst.modes[task] or formation_id in inst.modes_with_domination[task] # quick sanity check
        # if formation_id is dominated for task: modes list is empty
        if formation_id in inst.modes[task]:
            inst.modes[task] = {formation_id: inst.modes[task][formation_id]}
        else:
            inst.modes[task] = {}
        # formation_id will always be in modes_with_domination
        inst.modes_with_domination[task] = {formation_id: inst.modes_with_domination[task][formation_id]}


    # 2.3 Tasks per formation: formations with no tasks will be dropped
    inst.tasks_per_formation = {f: list(set(inst.tasks_per_formation[f]).intersection(set(inst.tasks))) for f in inst.tasks_per_formation}
    # 2.3.1 Add tasks from previous segment separately, because inst.tasks_per_formation still contains all formations for those
    for (task, formation_id) in formation_id_per_active_task.items():
        if formation_id in inst.modes[task]: # only add task if mode is not dominated for it
            inst.tasks_per_formation[formation_id].append(task)
    # 2.3.2 Drop formations with no suitable tasks
    to_drop = [f for f in inst.tasks_per_formation if not inst.tasks_per_formation]
    for f in to_drop:
        del inst.tasks_per_formation[f]

    # 2.4 Same for tasks per formation with domination
    inst.tasks_per_formation_with_domination = {f: list(set(inst.tasks_per_formation_with_domination[f]).intersection(set(inst.tasks)))
                                                for f in inst.tasks_per_formation_with_domination}
    for (task, formation_id) in formation_id_per_active_task.items():
        inst.tasks_per_formation_with_domination[formation_id].append(task)
    to_drop_with_domination = [f for f in inst.tasks_per_formation_with_domination if not inst.tasks_per_formation_with_domination]
    for f in to_drop_with_domination:
        del inst.tasks_per_formation_with_domination[f]

    # 2.5 Drop formations entry for all formations that are not suitable for any task, neither normally nor with domination
    for f in set(to_drop).intersection(set(to_drop_with_domination)):
        del inst.formations[f]
        del inst.formations_w_d[f]


    # 3. Restrict time horizon
    # 3.1 Get earliest possible leave time for any task: this will be the horizon begin
    # Initialize list of earliest leaves with leave time of earliest tours from previous segment that is still active
    # (Note: this is necessary to ensure all travel times values accessed during pricing actually exist)
    earliest_leaves = [tour.leave_time for tour in inst.tours_from_prev_segs]
    for task in inst.all_tasks:
        # 3.1.1 Get all time instants that would incur an early arrival
        early_arrivals = [t for t in range(inst.begin_horizon, inst.earliest_start[task] + 1) if
                               t + max(travel_times_per_bin[inst.bin_per_instant[t]][(inst.depot, task)])
                               <= inst.earliest_start[task]]
        # 3.1.2 If any exist: pick the largest one as the earliest leave time that might be optimal
        if early_arrivals:
            earliest_leaves.append(max(early_arrivals))
        # 3.1.3 Else: do not store anything: earliest leave will then be defined by the segment start
    if earliest_leaves:
        inst.begin_horizon = min(earliest_leaves)
    else:
        inst.begin_horizon = seg_start

    # 3.2 Same for end_horizon: get latest possible extended finish time + worst-case travel time and set it as end_horizon
    latest_returns = [seg_end]
    for task in inst.all_tasks:
        latest_return = max([t + max(travel_times_per_bin[inst.bin_per_instant[t % inst.last_inst_of_day]][(task, inst.depot)])
                             for t in range(inst.earliest_finish[task], inst.latest_finish_viol[task] + 1)])
        latest_returns.append(latest_return)
    inst.end_horizon = max(latest_returns)

    # 4. Update time bins and mappings between instant and time bins
    inst.bin_per_instant = {t: inst.bin_per_instant[t % inst.last_inst_of_day] for t in range(inst.begin_horizon, inst.end_horizon + 1)}
    inst.instant_per_bin = {int(b): [] for b in inst.bin_per_instant.values()}
    for t in inst.bin_per_instant:
        inst.instant_per_bin[int(inst.bin_per_instant[t])].append(t)
    inst.instant_per_bin = {b: list(sorted(inst.instant_per_bin[b])) for b in inst.instant_per_bin}
    inst.bins = list(inst.instant_per_bin.keys())
    inst.instants = [t for t in range(inst.begin_horizon, inst.end_horizon + 1)]

    # 5. Transfer restricted travel time bins
    add_travel_times_per_bin_restricted(inst, travel_times_per_bin, inst.bins)

    # 6. Overwrite inst.tasks with inst.all_tasks and delete inst.all_tasks
    inst.tasks = inst.all_tasks
    del inst.all_tasks

    # 7. Return tours returning to the depot at time seg_start: their workforce will be substracted from the available
    # workforce
    # + Return all tours that have finished before current segment start
    return tours_returning_to_depot, all_tours

def restrict_workforce(inst, inst_seg, th_segments, seg_idx,
                       all_tours, tours_returning_to_depot):
    """ Restrict workforce to considered planning horizon
    Note: we no longer adjust the workforce according to the solutions from the previous segments because
    workers occupied by tours from the previous segments that are still active during the new segment's time horizon
    will be force-occupied downstream anyway: for each existing tour, an artificial label in the pricing network
    will be created with a matching leave time, formation, and task sequence (up until the last task that was
    finished before the time horizon). This tour then occupies the exact number of workers that it also occupied
    in the solution of the last segment.
    There are two exceptions for this:
    A) tours that are returning to the depot and have not returned yet at the segment start:  We need to adjust the
    available workforce according to their occupancy
    B) tours that have already returned to the depot between [inst_seg.begin_horizon, seg_start]

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    inst_seg: instance_loader.Instance
        instance_loader.Instance object restrict to the current segment
    th_segments: list[(int, int)]
       Tuples indicating the start and end of a segment (both time instants included)
    seg_idx: int
       Index of current segment in th_segments
    all_tours: list[GH_tour]
        List of conducted tours
    tours_returning_to_depot: list[GH_tour]
        List of all tours that are returning to the depot at or after segment start
    """

    inst_seg.workers = copy.deepcopy(inst.workers)
    inst_seg.workers = {k: {t: v for (t, v) in inst_seg.workers[k].items() if
                            (t >= inst_seg.begin_horizon and t <= inst_seg.end_horizon)}
                        for k in inst_seg.workers}
    # Case A)
    for tour in tours_returning_to_depot:
        for k in tour.skill_comp_cnt:
            for t in range(tour.leave_time, tour.quantile_return_time):
                if t in inst_seg.workers[k]:
                    inst_seg.workers[k][t] -= tour.skill_comp_cnt[k]
    # Case B)
    for tour in all_tours:
        if tour.quantile_return_time >= inst_seg.begin_horizon and tour.quantile_return_time <= th_segments[seg_idx][0]:
            for k in tour.skill_comp_cnt:
                for t in range(tour.leave_time, tour.quantile_return_time):
                    if t in inst_seg.workers[k]:
                        inst_seg.workers[k][t] -= tour.skill_comp_cnt[k]
                        for kk in [kk for kk in inst_seg.workers_w_d if kk <= k]:
                            inst_seg.workers_w_d[kk][t] -= tour.skill_comp_cnt[k]
    inst_seg.workers_w_d = {k: {t: v for (t, v) in inst_seg.workers_w_d[k].items() if
                                (t >= inst_seg.begin_horizon and t <= inst_seg.end_horizon)}
                            for k in inst_seg.workers_w_d}

    return


def apply_solution_to_monte_carlo(all_tours, inst, travel_times_per_bin, rng, sample_cnt = 1):
    """Apply a solution to (one or multiple) monte-carlo samples.
    Repeatedly samples travel times, service times, and task execution times and applies the given set of tours to it.


    """
    obj_per_sample = []
    for i in tqdm(range(sample_cnt), "Trajectories:"):
        # 1. Deepcopy instance and tours
        inst_seg = copy.deepcopy(inst)
        seg_tours = copy.deepcopy(all_tours)

        # 2. Fetch service and task execution times
        for attr in ["sampled_modes", "sampled_modes_with_domination", "sampled_earliest_start", "sampled_latest_finish",
                     "sampled_latest_finish_viol", "sampled_earliest_finish", "sampled_latest_start", "sampled_latest_start_viol"]:
            setattr(inst_seg, attr, getattr(inst_seg, f"multi_{attr}")[i])

        # 3. Reset sampled travel times
        inst_seg.sampled_tts = {b: {} for b in inst.bins}

        # 4. Definy dummy knowledge state
        knowledge_state = KnowledgeState(inst_seg.begin_horizon)

        # 5. Simulate
        seg_tours = simulate(inst_seg, inst_seg, (np.inf, np.inf), travel_times_per_bin,
                             rng, seg_tours, knowledge_state, final_run = True)
        # 6. Store results
        obj_per_sample.append((sum([tour.task_cost_dict[task] for tour in seg_tours for task in tour.tasks]),
                              sum([tour.net_task_cost_dict[task] for tour in seg_tours for task in tour.tasks])))



    print(f"Done. Objective value summary:")
    min_val = min(obj_per_sample, key = lambda x: x[1]) # sort based on net value
    max_val = max(obj_per_sample, key = lambda x: x[1]) # sort based on net value
    avg_val = (np.mean([v[0] for v in obj_per_sample]), np.mean([v[1] for v in obj_per_sample]))
    print(f"Minimum objective value: {min_val[0]} (net {min_val[1]})")
    print(f"Maximum objective value: {max_val[0]} (net {max_val[1]})")
    print(f"Average objective value: {avg_val[0]} (net {avg_val[1]})")

    return