"""Utility functions used for solving instances with a rolling horizon approach.
"""
from src.instance_loader.instance_loader import add_travel_times_per_bin_restricted

def get_segments(inst, segment_length):
    """Segment time horizon into segments of length 'segment_length'. Last segment might be shorter than the others.

    Parameters
    ----------
    inst: instance_loader.Instance
        Contains all necessary instance data read from input files.
    segment_length: int
        Length of each segment (in time steps) during rolling horizon separation

    Returns
    -------
    th_segments: list[(int, int)]
       Tuples indicating the start and end of a segment (both time instants included)
    """
    th_segments = []
    start = inst.begin_horizon
    while start <= inst.end_horizon:
        end = min(start + segment_length - 1, inst.end_horizon)
        th_segments.append((start, end))
        start += segment_length

    return th_segments


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
    active_tours_from_prev_segs = []
    for seg_idx in sol_per_seg:
        if sol_per_seg[seg_idx]: # Skip segments with None solution (=no tasks contained in segment)
            sol_seg = sol_per_seg[seg_idx]
            # 0.1 Get tours that are still active at the segment start
            if sol_seg.optimal is not None:
                active_tours_from_prev_segs = [tour for tour in sol_seg.optimal.tours if (tour.leave_time < seg_start and tour.quantile_return_time > seg_start)]
            if sol_seg.heuristic is not None:
                active_tours_from_prev_segs = [tour for tour in sol_seg.heuristic.tours if (tour.leave_time < seg_start and tour.quantile_return_time > seg_start)]
    inst.tours_from_prev_segs = active_tours_from_prev_segs

    # 1. Get all tasks in considered segment
    inst.tasks = [task for task in inst.tasks if (inst.earliest_start[task] >= seg_start and inst.earliest_start[task] < seg_start + lookahead)]
    # 1.1 Also include tasks from tours from previous segments that have been finished or are currently active
    inst.tasks_from_prev_segs = []
    formation_id_per_active_task = {}
    tours_returning_to_depot = [] # list of tours that are active, but are currently returning to the depot
    for i in range(len(inst.tours_from_prev_segs)):
        tour = inst.tours_from_prev_segs[i]
        # 1.1.1 Skip tours that have not left the depot yet: we can freely readjust them
        if tour.leave_time >= seg_start:
            continue
        # 1.1.2 Get sorted list of quantile finish times
        tasks_quantile_finish_times = sorted(tour.quantile_finish_time.items(), key = lambda x: x[0])
        # 1.1.3 Store all tasks whose subsequent task is not finished yet (i.e., all tasks that are either already finished or currently being executed)
        curr_active = min([(node, qft) for (node, qft) in tasks_quantile_finish_times if qft > seg_start], key = lambda x: x[1])
        # a) If only active task/node is the sink: do not store this tour as a tour from a previous segment
        # Instead, we adjust the workforce accordingly
        if len(curr_active) == 1 and curr_active[0][0] == "sink":
            tours_returning_to_depot.append(i)
            continue
        # b) Else: proceed
        active_tasks = [task for task in tour.tasks if tour.quantile_finish_time[task] <= curr_active[1]]
        inst.tasks_from_prev_segs += active_tasks
        for task in active_tasks:
            formation_id_per_active_task[task] = tour.formation_id
        # 1.1.4 Store active tasks sequence in tour object, will be used downstream during pricing network setup
        tour.active_tasks = [task for task in tour.tasks if task in active_tasks]
    # 1.2 Remove tasks from active tours from inst.tasks: their ES might not have been reached yet,
    # but a team is already on their way to the task: in this case, we can't reschedule the ask
    inst.tasks = [task for task in inst.tasks if task not in inst.tasks_from_prev_segs]
    # 1.3 Store all tasks (active and non-active) to inst.all_tasks
    inst.all_tasks = inst.tasks + inst.tasks_from_prev_segs
    # 1.4 Remove tours currently returning to the depot from the list tours from the previous segment
    for i in sorted(tours_returning_to_depot, reverse = True):
        inst.tours_from_prev_segs.pop(i)
    # 1.5 Update tours from the last segments by removing the tours that are not yet finished and are subject to change
    # in the current segment
    all_tours = [tour for tour in all_tours if (tour not in inst.tours_from_prev_segs and tour.leave_time < seg_start)]

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
    earliest_leaves = []
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
        latest_return = max([t + max(travel_times_per_bin[inst.bin_per_instant[t]][(task, inst.depot)])
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
